import numpy as np
from optuna.integration import XGBoostPruningCallback
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from optuna import create_study
import optuna
import pandas as pd
from tqdm import tqdm

from preprocess.preprocess_all_in_one import get_all_feature_gen_macro

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

all_common_feature = ['DATE',
                      'SALE_RATE',
                      'JEONSE_RATE',
                      'UNDERVALUE_JEONSE',
                      'AREA',
                      'FLOOR',
                      'GU_DONG_AMOUNT_MEAN',
                      'GU_DONG_AMOUNT_MEDIAN',
                      'GU_DONG_AMOUNT_SKEW',
                      'GU_DONG_AMOUNT_MIN',
                      'GU_DONG_AMOUNT_MAX',
                      'GU_DONG_AMOUNT_MAD',
                      'COMPLEX_NAME_AMOUNT_MEAN',
                      'COMPLEX_NAME_AMOUNT_MEDIAN',
                      'COMPLEX_NAME_AMOUNT_SKEW',
                      'COMPLEX_NAME_AMOUNT_MIN',
                      'COMPLEX_NAME_AMOUNT_MAX',
                      'COMPLEX_NAME_AMOUNT_MAD',
                      'COMPLEX_NAME',
                      'REGION_CODE',
                      'INCOME_PIR',
                      'SALE_CONSUMER_FLAG',
                      'FINAL_KHAI',
                      'SALE_OVER_JEONSE',
                      'SUPPLY_DEMAND',
                      'HOUSE_OCCUPANCY',
                      'HOUSE_UNSOLD',
                      'KOR_VALUE',
                      'EU_VALUE',
                      'CN_VALUE',
                      'USA_VALUE',
                      'INTEREST_RATE',
                      'KOREA_IR',
                      'AMOUNT']

feature_x = ['SALE_RATE',
             'JEONSE_RATE',
             'UNDERVALUE_JEONSE',
             'AREA',
             'FLOOR',
             'GU_DONG_AMOUNT_MEAN',
             'GU_DONG_AMOUNT_MEDIAN',
             'GU_DONG_AMOUNT_SKEW',
             'GU_DONG_AMOUNT_MIN',
             'GU_DONG_AMOUNT_MAX',
             'GU_DONG_AMOUNT_MAD',
             'COMPLEX_NAME_AMOUNT_MEAN',
             'COMPLEX_NAME_AMOUNT_MEDIAN',
             'COMPLEX_NAME_AMOUNT_SKEW',
             'COMPLEX_NAME_AMOUNT_MIN',
             'COMPLEX_NAME_AMOUNT_MAX',
             'COMPLEX_NAME_AMOUNT_MAD',
             'COMPLEX_NAME',
             'REGION_CODE',
             'INCOME_PIR',
             'SALE_CONSUMER_FLAG',
             'FINAL_KHAI',
             'SALE_OVER_JEONSE',
             'SUPPLY_DEMAND',
             'HOUSE_OCCUPANCY',
             'HOUSE_UNSOLD',
             'KOR_VALUE',
             'EU_VALUE',
             'CN_VALUE',
             'USA_VALUE',
             'INTEREST_RATE',
             'KOREA_IR']

target = 'AMOUNT'


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

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)

        preds = model.predict(X_val)

        rmse = mean_squared_error(y_val, preds, squared=False)

        return rmse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5)
    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)
    return study


