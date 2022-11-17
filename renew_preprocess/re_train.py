from renew_preprocess.re_train_utils import *


def get_tuning_optuna(X_df, y_df, X_test, y_test, model='tabnet'):
    """
    :param model: tabnet, xgboost, catboost, lightgbm
    :return: X_test, y_test, y_pred, rmse_final
    """

    if model == 'tabnet':
        X_test, y_test, y_pred, rmse_final = run_tabnet_optuna(X_df, y_df, X_test, y_test)
    elif model == 'xgboost':
        X_test, y_test, y_pred, rmse_final = run_xgboost_optuna(X_df, y_df, X_test, y_test)
    elif model == 'lightgbm':
        X_test, y_test, y_pred, rmse_final = run_lightgbm_optuna(X_df, y_df, X_test, y_test)
    else:  # catboost
        X_test, y_test, y_pred, rmse_final = run_catboost_optuna(X_df, y_df, X_test, y_test)

    return X_test, y_test, y_pred, rmse_final
