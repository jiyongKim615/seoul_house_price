from catboost import CatBoostRegressor
from optuna.integration import XGBoostPruningCallback
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import lightgbm as lgb
import numpy as np
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import KFold, train_test_split
import torch
import optuna


target = 'AMOUNT'


def run_tabnet_optuna(X_df, y_df, X_test, y_test):  # 0.5 --> 30분
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
                          max_epochs=trial.suggest_int('epochs', 100, 200),
                          eval_metric=['rmse'])
            CV_score_array.append(regressor.best_cost)
        avg = np.mean(CV_score_array)
        return avg

    study = optuna.create_study(direction="minimize", study_name='TabNet optimization')
    # study.optimize(Objective, timeout=run_time * 60)  # 5 hours
    study.optimize(Objective, n_trials=10)  # 5 hours

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

    pred = regressor.predict(X_test)
    exp_y_pred = np.expm1(pred)
    exp_y_test = np.expm1(y_test)
    rmse_non = mean_squared_error(y_test, pred, squared=False)
    rmse_exp = mean_squared_error(exp_y_test, exp_y_pred, squared=False)
    return X_test, y_test, pred, rmse_non, rmse_exp


def run_xgboost_optuna(X_df, y_df, X_test, y_test):
    X = X_df.copy()
    y = y_df.copy()

    def objective(trial):
        param = {
            'max_depth': trial.suggest_int('max_depth', 1, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.01, 1.0),
            'subsample': trial.suggest_float('subsample', 0.01, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0),
            'random_state': trial.suggest_int('random_state', 1, 1000)
        }
        model = XGBRegressor(**param)
        model.fit(X_df, y_df)
        y_pred = model.predict(X_test)
        rmse_xgb = mean_squared_error(y_test, y_pred, squared=False)
        return rmse_xgb

    study = optuna.create_study(direction='minimize', study_name='regression')
    study.optimize(objective, n_trials=10)

    xgb_params = study.best_params
    kf = KFold(n_splits=5, random_state=42, shuffle=True)

    oof_preds = np.zeros((X.shape[0],))
    preds = 0
    model_fi = 0
    mean_rmse = 0

    for num, (train_id, valid_id) in enumerate(kf.split(X)):
        X_train, X_valid = X[train_id], X[valid_id]
        y_train, y_valid = y[train_id], y[valid_id]

        model = XGBRegressor(**xgb_params)
        model.fit(X_train, y_train,
                  verbose=False,
                  eval_set=[(X_train, y_train), (X_valid, y_valid)],
                  eval_metric="rmse",
                  early_stopping_rounds=250)

        # Mean of the predictions
        preds += model.predict(X_test) / 5  # Splits

        # Mean of feature importance
        model_fi += model.feature_importances_ / 5  # splits

        # Out of Fold predictions
        oof_preds[valid_id] = model.predict(X_valid)
        fold_rmse = np.sqrt(mean_squared_error(y_valid, oof_preds[valid_id]))
        print(f"Fold {num} | RMSE: {fold_rmse}")

        mean_rmse += fold_rmse / 5

    print(f"\nOverall RMSE: {mean_rmse}")

    exp_y_pred = np.expm1(preds)
    exp_y_test = np.expm1(y_test)
    rmse_non = mean_squared_error(y_test, preds, squared=False)
    rmse_exp = mean_squared_error(exp_y_test, exp_y_pred, squared=False)
    return X_test, y_test, preds, rmse_non, rmse_exp


def run_lightgbm_optuna(X_df, y_df, X_test, y_test):
    X = X_df.copy()
    y = y_df.copy()

    def objective(trial):
        param = {
            'objective': 'regression',
            'verbose': -1,
            'metric': 'rmse',
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_loguniform("learning_rate", 1e-8, 1e-2),
            'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_loguniform('subsample', 0.4, 1),
        }

        model = lgb.LGBMRegressor(**param)
        model = model.fit(X_df, y_df)
        y_pred = model.predict(X_test)
        rmse_lgb = mean_squared_error(y_test, y_pred, squared=False)
        return rmse_lgb

    study = optuna.create_study(direction='minimize', study_name='regression')
    study.optimize(objective, n_trials=10)

    lgb_params = study.best_params
    kf = KFold(n_splits=5, random_state=42, shuffle=True)

    oof_preds = np.zeros((X.shape[0],))
    preds = 0
    model_fi = 0
    mean_rmse = 0

    for num, (train_id, valid_id) in enumerate(kf.split(X)):
        X_train, X_valid = X[train_id], X[valid_id]
        y_train, y_valid = y[train_id], y[valid_id]

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(X_train, y_train,
                  verbose=False,
                  eval_set=[(X_train, y_train), (X_valid, y_valid)],
                  eval_metric="rmse",
                  early_stopping_rounds=250)

        # Mean of the predictions
        preds += model.predict(X_test) / 5  # Splits

        # Mean of feature importance
        model_fi += model.feature_importances_ / 5  # splits

        # Out of Fold predictions
        oof_preds[valid_id] = model.predict(X_valid)
        fold_rmse = np.sqrt(mean_squared_error(y_valid, oof_preds[valid_id]))
        print(f"Fold {num} | RMSE: {fold_rmse}")

        mean_rmse += fold_rmse / 5

    print(f"\nOverall RMSE: {mean_rmse}")

    exp_y_pred = np.expm1(preds)
    exp_y_test = np.expm1(y_test)
    rmse_non = mean_squared_error(y_test, preds, squared=False)
    rmse_exp = mean_squared_error(exp_y_test, exp_y_pred, squared=False)
    return X_test, y_test, preds, rmse_non, rmse_exp


def run_catboost_optuna(X_df, y_df, X_test, y_test):
    X = X_df.copy()
    y = y_df.copy()

    def objective(trial):
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
        param = {
            'loss_function': 'RMSE',
            "task_type": "GPU",
            'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-3, 10.0),
            'max_bin': trial.suggest_int('max_bin', 200, 400),
            # 'rsm': trial.suggest_uniform('rsm', 0.3, 1.0),
        #    'subsample': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
            'learning_rate': trial.suggest_uniform('learning_rate', 0.006, 0.018),
            'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
            'max_depth': trial.suggest_categorical('max_depth', [5, 7, 9, 11, 13, 15]),
            'random_state': trial.suggest_categorical('random_state', [2020]),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 300),
        }
        model = CatBoostRegressor(**param)

        model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=200, verbose=False)

        preds = model.predict(test_x)

        rmse = mean_squared_error(test_y, preds, squared=False)

        return rmse

    study = optuna.create_study(direction='minimize', study_name='regression')
    study.optimize(objective, n_trials=10)

    cat_params = study.best_params
    kf = KFold(n_splits=5, random_state=42, shuffle=True)

    oof_preds = np.zeros((X.shape[0],))
    preds = 0
    model_fi = 0
    mean_rmse = 0

    for num, (train_id, valid_id) in enumerate(kf.split(X)):
        X_train, X_valid = X[train_id], X[valid_id]
        y_train, y_valid = y[train_id], y[valid_id]

        model = CatBoostRegressor(**cat_params, eval_metric='RMSE')
        model.fit(X_train, y_train,
                  verbose=False,
                  eval_set=[(X_train, y_train), (X_valid, y_valid)],
                  early_stopping_rounds=250)

        # Mean of the predictions
        preds += model.predict(X_test) / 5  # Splits

        # Mean of feature importance
        model_fi += model.feature_importances_ / 5  # splits

        # Out of Fold predictions
        oof_preds[valid_id] = model.predict(X_valid)
        fold_rmse = np.sqrt(mean_squared_error(y_valid, oof_preds[valid_id]))
        print(f"Fold {num} | RMSE: {fold_rmse}")

        mean_rmse += fold_rmse / 5

    print(f"\nOverall RMSE: {mean_rmse}")
    exp_y_pred = np.expm1(preds)
    exp_y_test = np.expm1(y_test)
    rmse_non = mean_squared_error(y_test, preds, squared=False)
    rmse_exp = mean_squared_error(exp_y_test, exp_y_pred, squared=False)
    return X_test, y_test, preds, rmse_non, rmse_exp, cat_params
