from optuna.integration import XGBoostPruningCallback
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import numpy as np
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import KFold
import torch
import optuna

feature_x1 = ['AREA', 'BUILD_1ST_NUM', 'FLOOR', 'ROAD_NAME', 'GU',
              'DONG', 'DONG_CPX_NME', 'YEAR', 'MONTH', 'MONTH_SIN', 'MONTH_COS']
target = 'AMOUNT'

feature_x2 = []
feature_x3 = []


def run_tabnet_optuna(X_df, y_df, final_test, run_time=0.5):  # 0.5 --> 30분
    X = X_df.copy()
    y = y_df.copy()

    def Objective(trial):
        mask_type = trial.suggest_categorical("mask_type", ["entmax", "sparsemax"])
        n_da = trial.suggest_int("n_da", 8, 64, step=4)
        n_steps = trial.suggest_int("n_steps", 1, 3, step=1)
        gamma = trial.suggest_float("gamma", 1., 1.4, step=0.2)
        n_shared = trial.suggest_int("n_shared", 1, 3)
        lambda_sparse = trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True)
        tabnet_params = dict(n_d=n_da, n_a=n_da, n_steps=n_steps, gamma=gamma,
                             lambda_sparse=lambda_sparse, optimizer_fn=torch.optim.Adam,
                             optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                             mask_type=mask_type, n_shared=n_shared,
                             scheduler_params=dict(mode="min",
                                                   patience=trial.suggest_int("patienceScheduler", low=3, high=10),
                                                   # changing sheduler patience to be lower than early stopping patience
                                                   min_lr=1e-5,
                                                   factor=0.5),
                             scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                             verbose=1,
                             )  # early stopping
        kf = KFold(n_splits=5, random_state=42, shuffle=True)
        CV_score_array = []
        for train_index, test_index in kf.split(X):
            X_train, X_valid = X[train_index], X[test_index]
            y_train, y_valid = y[train_index], y[test_index]
            regressor = TabNetRegressor(**tabnet_params)
            regressor.fit(X_train=X_train, y_train=y_train,
                          eval_set=[(X_valid, y_valid)],
                          patience=trial.suggest_int("patience", low=15, high=30),
                          max_epochs=trial.suggest_int('epochs', 1000, 5000),
                          eval_metric=['rmse'])
            CV_score_array.append(regressor.best_cost)
        avg = np.mean(CV_score_array)
        return avg

    study = optuna.create_study(direction="minimize", study_name='TabNet optimization')
    study.optimize(Objective, timeout=run_time * 60)  # 5 hours

    TabNet_params = study.best_params

    final_params = dict(n_d=TabNet_params['n_da'], n_a=TabNet_params['n_da'], n_steps=TabNet_params['n_steps'],
                        gamma=TabNet_params['gamma'],
                        lambda_sparse=TabNet_params['lambda_sparse'], optimizer_fn=torch.optim.Adam,
                        optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                        mask_type=TabNet_params['mask_type'], n_shared=TabNet_params['n_shared'],
                        scheduler_params=dict(mode="min",
                                              patience=TabNet_params['patienceScheduler'],
                                              min_lr=1e-5,
                                              factor=0.5),
                        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                        verbose=1,
                        )
    epochs = TabNet_params['epochs']

    regressor = TabNetRegressor(**final_params)
    regressor.fit(X_train=X, y_train=y,
                  patience=TabNet_params['patience'], max_epochs=epochs,
                  eval_metric=['rmse'])

    X_test = final_test[feature_x1].values
    pred = regressor.predict(X_test)
    y_pred = np.expm1(pred)
    y_test = final_test['AMOUNT'].values
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return X_test, y_test, y_pred, rmse