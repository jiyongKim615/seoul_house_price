from optuna.integration import XGBoostPruningCallback
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from train.train_data import *
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import KFold
import torch
from preprocess.preprocess_all_in_one import get_all_feature_gen_macro
import optuna


FS = (14, 6)  # figure size
RS = 124  # random state
N_JOBS = 8  # number of parallel threads

# repeated K-folds
N_SPLITS = 10
N_REPEATS = 1

# Optuna
N_TRIALS = 100
MULTIVARIATE = True

# XGBoost
EARLY_STOPPING_ROUNDS = 100


def run_tabnet_with_hyperparameter_tuning(train_test_ratio=80, run_time=0.5):  # 0.5 --> 30분
    X_df, y_df, final_test = get_tabnet_final_df(train_test_ratio=train_test_ratio)
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
                             verbose=0,
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

    X_test = final_test[feature_x].values
    pred = regressor.predict(X_test)
    y_pred = np.expm1(pred)
    y_test = final_test['AMOUNT'].values
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return X_test, y_test, y_pred, rmse


def run_xgboost_optuna(X_train, y_train, X_val, y_val):
    def objective(trial):
        # XGBoost parameters
        params = {
            "verbosity": 0,  # 0 (silent) - 3 (debug)
            "objective": "reg:squarederror",
            "n_estimators": 1000,
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.05),
            "colsample_bytree": trial.suggest_loguniform("colsample_bytree", 0.2, 0.6),
            "subsample": trial.suggest_loguniform("subsample", 0.4, 0.8),
            "alpha": trial.suggest_loguniform("alpha", 0.01, 10.0),
            "lambda": trial.suggest_loguniform("lambda", 1e-8, 10.0),
            "gamma": trial.suggest_loguniform("lambda", 1e-8, 10.0),
            "min_child_weight": trial.suggest_loguniform("min_child_weight", 10, 100),
            "seed": 42,
            "n_jobs": -1,
        }

        model = XGBRegressor(**params)
        pruning_callback = XGBoostPruningCallback(trial, "validation_0-rmse")

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=True)

        preds = model.predict(X_val)

        rmse = mean_squared_error(y_val, preds, squared=False)

        return rmse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5)
    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)
    return study


