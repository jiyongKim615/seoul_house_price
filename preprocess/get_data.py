from glob import glob
import os
import pandas as pd

file_path = '/Users/jiyongkim/Downloads/seoul_price_data/'
file_name_os = os.listdir(file_path)
file_name_glob = glob(file_path+'*')


def get_raw_house_data():
    df_list = []
    for fn in file_name_os:
        df_list.append(
            pd.read_csv(file_path + fn,
                        header=None,
                        skiprows=16,
                        names=['시군구', '번지', '본번', '부번', '단지명', '전용면적(㎡)', \
                               '계약년월', '계약일', '거래금액(만원)', '층', \
                               '건축년도', '도로명', '해제사유발생일', '거래유형', '중개사소재지'],
                        encoding='cp949'))

    list_data = pd.concat(df_list)
    return list_data


def get_feature_gen():
    return None


def get_train_test():
    raw_df = concat_all_data()
    df = preprocess_feature_gen()
    train_df, test_df = split_data()

    return train_df, test_df