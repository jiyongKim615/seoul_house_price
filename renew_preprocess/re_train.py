from renew_preprocess.re_train_utils import *


def get_tuning_optuna(model='tabnet', feature_select='raw'):
    """
    :param model: tabnet, xgboost, catboost, lightgbm
    :return: X_test, y_test, y_pred, rmse_final
    """

    if model == 'tabnet':
        X_test, y_test, y_pred, rmse_final = run_tabnet_optuna()
    elif model == 'xgboost':
        X_test, y_test, y_pred, rmse_final = run_xgboost_optuna()
    elif model == 'lightgbm':
        X_test, y_test, y_pred, rmse_final = run_lightgbm_optuna()
    else:  # catboost
        X_test, y_test, y_pred, rmse_final = run_catboost_optuna()

    return X_test, y_test, y_pred, rmse_final
