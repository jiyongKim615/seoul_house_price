import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, skew  # for some statistics
from scipy import stats

file_path = '/Users/jiyongkim/Downloads/seoul_price_data/'
file_name_os = os.listdir(file_path)


def get_raw_house_data():
    df_list = []
    for fn in file_name_os:
        df_list.append(
            pd.read_csv(file_path + fn,
                        header=None,
                        skiprows=16,
                        names=['시군구', '번지', '본번', '부번', '단지명', '전용면적(㎡)',
                               '계약년월', '계약일', '거래금액(만원)', '층',
                               '건축년도', '도로명', '해제사유발생일', '거래유형', '중개사소재지'],
                        encoding='cp949'))

    list_data = pd.concat(df_list)
    return list_data


def get_missing_data_percentage(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data


def get_rename(df):
    df.rename(columns={'시군구': 'CITY_DISTRICT'}, inplace=True)
    df.rename(columns={'번지': 'ADDRESS'}, inplace=True)
    df.rename(columns={'본번': 'BUILD_1ST_NUM'}, inplace=True)
    df.rename(columns={'부번': 'BUILD_2ND_NUM'}, inplace=True)
    df.rename(columns={'단지명': 'COMPLEX_NAME'}, inplace=True)
    df.rename(columns={'전용면적(㎡)': 'AREA'}, inplace=True)
    df.rename(columns={'계약년월': 'CONTRACT_YEAR_MONTH'}, inplace=True)
    df.rename(columns={'계약일': 'CONTRACT_DAY'}, inplace=True)
    df.rename(columns={'거래금액(만원)': 'AMOUNT'}, inplace=True)
    df.rename(columns={'층': 'FLOOR'}, inplace=True)
    df.rename(columns={'건축년도': 'CONSTRUCTION_YEAR'}, inplace=True)
    df.rename(columns={'도로명': 'ROAD_NAME'}, inplace=True)
    df.rename(columns={'해제사유발생일': 'TERMINATION_DATE'}, inplace=True)
    df.rename(columns={'거래유형': 'TRANSACTION_TYPE'}, inplace=True)
    df.rename(columns={'중개사소재지': 'BROKER_LOCATION'}, inplace=True)
    return df


def get_amount_int(train_df_copy):
    # 타겟 AMOUNT 처리 및 활용
    train_df_copy['AMOUNT'] = train_df_copy.AMOUNT.str.replace(',', '').astype('int64')
    return train_df_copy


def get_gu_dong_fe(df_copy):
    city_district_df = df_copy['CITY_DISTRICT'].str.split(' ', expand=True)
    city_district_df.rename(columns={0: 'CITY'}, inplace=True)
    city_district_df.rename(columns={1: 'GU'}, inplace=True)
    city_district_df.rename(columns={2: 'DONG'}, inplace=True)
    df_copy['GU'] = city_district_df['GU']
    df_copy['DONG'] = city_district_df['DONG']
    return df_copy


def get_dong_cpx_nme_fe(df_copy):
    df_copy['SPACE'] = ' '
    # complex_name과 합치기
    df_copy['DONG_CPX_NME'] = df_copy['DONG'] + df_copy['SPACE'] + df_copy['COMPLEX_NAME']
    df_copy.drop('SPACE', axis=1)
    return df_copy


def get_contract_date_preprocess(df):
    # 앞에 0을 채우면서 문자열로 바꿀 것
    df['CONTRACT_YEAR_MONTH'] = df['CONTRACT_YEAR_MONTH'].astype(str).str.zfill(2)
    df['CONTRACT_DAY'] = df['CONTRACT_DAY'].astype(str).str.zfill(2)
    df['DATE'] = df['CONTRACT_YEAR_MONTH'] + df['CONTRACT_DAY']
    df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d')
    return df


def get_road_name_first_only(df):
    road_name_df = df['ROAD_NAME'].str.split(' ', expand=True)
    road_name_df.rename(columns={0: 'ROAD_NAME'}, inplace=True)
    df['ROAD_NAME'] = road_name_df['ROAD_NAME']
    return df


def create_feature_time(df, col='DATE'):
    df = get_contract_date_preprocess(df)
    df[col] = pd.to_datetime(df[col])
    df["YEAR"] = df[col].dt.year
    df["MONTH"] = df[col].dt.month
    df['MONTH_SIN'] = np.sin(2 * np.pi * df["MONTH"] / 12)
    df['MONTH_COS'] = np.cos(2 * np.pi * df["MONTH"] / 12)

    return df


def get_log_transform(df, target):
    # Target 변수의 로그변환
    # We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
    df_copy = df.copy()
    df_copy[target] = np.log1p(df[target])

    # Check the new distribution
    sns.distplot(df_copy[target], fit=norm)

    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(df_copy[target])
    print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    # Now plot the distribution
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
               loc='best')
    plt.ylabel('Frequency')
    plt.title('Target distribution')

    # Get also the QQ-plot
    fig = plt.figure()
    res = stats.probplot(df_copy[target], plot=plt)
    plt.show()

    return df_copy


def get_train_test_df(df_copy):
    cond = (df_copy['YEAR'] == 2022) & (df_copy['MONTH'] == 10)
    test_df = df_copy[cond]
    cond_train = (df_copy['YEAR'] != 2022) | (df_copy['MONTH'] != 10)
    train_df = df_copy[cond_train]
    train_df_cpx_name_lst = train_df['COMPLEX_NAME']
    test_df_cpx_nme_lst = test_df['COMPLEX_NAME'].unique().tolist()
    SetList1 = set(train_df_cpx_name_lst)
    SetList2 = set(test_df_cpx_nme_lst)
    except_lst = list(SetList2.difference(SetList1))

    for i in except_lst:
        test_df = test_df[test_df['COMPLEX_NAME'] != i]

    return train_df, test_df





