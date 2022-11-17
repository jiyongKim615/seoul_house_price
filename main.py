from preprocess.target_encoding import TargetEncode
from renew_preprocess.re_house_eco_fe_utils import preprocess_gu_feature
from renew_preprocess.re_preprocess import get_raw_preprocess_df
from renew_preprocess.re_preprocess_utils import get_raw_house_data
from renew_preprocess.re_train import get_tuning_optuna
from renew_preprocess.re_train_utils import run_tabnet_optuna
from train.hyperparameter_tuning import run_tabnet_with_hyperparameter_tuning
from train.ploting import plot_train_val_metric
from train.train_data import redefine_train_df_test_df, get_final_df
get_tuning_optuna
if __name__ == '__main__':
    print()
