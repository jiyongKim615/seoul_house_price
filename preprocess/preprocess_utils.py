from glob import glob
import os
import pandas as pd
from pytimekr import pytimekr
import warnings
from tqdm import tqdm
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

file_path = '/Users/jiyongkim/Downloads/seoul_price_data/'
file_name_os = os.listdir(file_path)
file_name_glob = glob(file_path + '*')
target = 'AMOUNT'


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


def get_gen_ml_train_test_df():
    df = get_raw_house_data()
    df = get_rename(df)
    train_df = df.iloc[:206477, :]
    test_df = df.iloc[206477:, :]
    train_df = get_amount_int(train_df)
    test_df = get_amount_int(test_df)
    return train_df, test_df


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


def get_groupby_target_stats(train_df_copy, groupby_lst, target):
    gu_dong_mean = train_df_copy.groupby(groupby_lst)[target].mean().reset_index()
    gu_dong_median = train_df_copy.groupby(groupby_lst)[target].median().reset_index()
    gu_dong_skew = train_df_copy.groupby(groupby_lst)[target].skew().reset_index()
    gu_dong_min = train_df_copy.groupby(groupby_lst)[target].min().reset_index()
    gu_dong_max = train_df_copy.groupby(groupby_lst)[target].max().reset_index()
    gu_dong_mad = train_df_copy.groupby(groupby_lst)[target].mad().reset_index()
    return gu_dong_mean, gu_dong_median, gu_dong_skew, gu_dong_min, gu_dong_max, gu_dong_mad


def merge_stats_to_df(train_df_copy, mean_df, median_df, skew_df, min_df, max_df, mad_df, groupby_lst):
    train_df_copy = pd.merge(train_df_copy, mean_df, on=groupby_lst)
    train_df_copy = pd.merge(train_df_copy, median_df, on=groupby_lst)
    train_df_copy = pd.merge(train_df_copy, skew_df, on=groupby_lst)
    train_df_copy = pd.merge(train_df_copy, min_df, on=groupby_lst)
    train_df_copy = pd.merge(train_df_copy, max_df, on=groupby_lst)
    train_df_copy = pd.merge(train_df_copy, mad_df, on=groupby_lst)
    return train_df_copy


# module
def get_city_district_target_stats(train_df):
    """
    1. 서울특별시 강남구 개포동 --> 빈 공백 베이스로 나눠 --> CITY, GU, DONG으로 나눌 것
    2. 각 세분화된 피처에 대한 target과의 통계값 산출
    3. 원래 데이터에 병합
    """
    train_df_copy = train_df.copy()
    city_district_df = train_df_copy['CITY_DISTRICT'].str.split(' ', expand=True)
    city_district_df.rename(columns={0: 'CITY'}, inplace=True)
    city_district_df.rename(columns={1: 'GU'}, inplace=True)
    city_district_df.rename(columns={2: 'DONG'}, inplace=True)

    train_df_copy['CITY'] = city_district_df['CITY']
    train_df_copy['GU'] = city_district_df['GU']
    train_df_copy['DONG'] = city_district_df['DONG']

    gu_dong_mean, gu_dong_median, gu_dong_skew, gu_dong_min, \
    gu_dong_max, gu_dong_mad = \
        get_groupby_target_stats(train_df_copy, ['GU', 'DONG'], 'AMOUNT')

    # rename feature --> stats
    gu_dong_mean.rename(columns={'AMOUNT': 'GU_DONG_AMOUNT_MEAN'}, inplace=True)
    gu_dong_median.rename(columns={'AMOUNT': 'GU_DONG_AMOUNT_MEDIAN'}, inplace=True)
    gu_dong_skew.rename(columns={'AMOUNT': 'GU_DONG_AMOUNT_SKEW'}, inplace=True)
    gu_dong_min.rename(columns={'AMOUNT': 'GU_DONG_AMOUNT_MIN'}, inplace=True)
    gu_dong_max.rename(columns={'AMOUNT': 'GU_DONG_AMOUNT_MAX'}, inplace=True)
    gu_dong_mad.rename(columns={'AMOUNT': 'GU_DONG_AMOUNT_MAD'}, inplace=True)

    # merge
    train_df_copy = pd.merge(train_df_copy, gu_dong_mean, on=['GU', 'DONG'])
    train_df_copy = pd.merge(train_df_copy, gu_dong_median, on=['GU', 'DONG'])
    train_df_copy = pd.merge(train_df_copy, gu_dong_skew, on=['GU', 'DONG'])
    train_df_copy = pd.merge(train_df_copy, gu_dong_min, on=['GU', 'DONG'])
    train_df_copy = pd.merge(train_df_copy, gu_dong_max, on=['GU', 'DONG'])
    train_df_copy = pd.merge(train_df_copy, gu_dong_mad, on=['GU', 'DONG'])
    return train_df_copy


def get_complex_name_target_stats(train_df, groupby_lst):
    # groupby_stats
    mean_df, median_df, skew_df, min_df, max_df, mad_df = \
        get_groupby_target_stats(train_df, groupby_lst, target)
    # rename feature
    mean_df.rename(columns={'AMOUNT': 'COMPLEX_NAME_AMOUNT_MEAN'}, inplace=True)
    median_df.rename(columns={'AMOUNT': 'COMPLEX_NAME_AMOUNT_MEDIAN'}, inplace=True)
    skew_df.rename(columns={'AMOUNT': 'COMPLEX_NAME_AMOUNT_SKEW'}, inplace=True)
    min_df.rename(columns={'AMOUNT': 'COMPLEX_NAME_AMOUNT_MIN'}, inplace=True)
    max_df.rename(columns={'AMOUNT': 'COMPLEX_NAME_AMOUNT_MAX'}, inplace=True)
    mad_df.rename(columns={'AMOUNT': 'COMPLEX_NAME_AMOUNT_MAD'}, inplace=True)
    # merge
    train_df = merge_stats_to_df(train_df, mean_df, median_df, skew_df, min_df, max_df, mad_df, groupby_lst)
    return train_df


def get_date_preprocess(df):
    df = get_contract_date_preprocess(df)
    df = get_weekend_week_holiday_info(df)
    return df


def get_contract_date_preprocess(df):
    # 앞에 0을 채우면서 문자열로 바꿀 것
    df['CONTRACT_YEAR_MONTH'] = df['CONTRACT_YEAR_MONTH'].astype(str).str.zfill(2)
    df['CONTRACT_DAY'] = df['CONTRACT_DAY'].astype(str).str.zfill(2)
    df['DATE'] = df['CONTRACT_YEAR_MONTH'] + df['CONTRACT_DAY']
    df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d')
    return df


def get_weekend_week_holiday_info(df):
    kr_holidays = pytimekr.holidays(year=2022)
    df['HOLIDAY_FLAG'] = \
        df.DATE.apply(lambda x: 1 if x in kr_holidays else 0)
    df['WEEK_INFO'] = df['DATE'].dt.dayofweek
    df["WEEKEND_FLAG"] = df['WEEK_INFO'] > 4

    return df


def get_diff_year_construct_contract(train_df):
    diff_year = (train_df['DATE'].dt.year - train_df['CONSTRUCTION_YEAR']).astype(int)
    train_df['DIFF_YEAR_CONSTRUCT_CONTRACT'] = diff_year
    # 한계: 직전 계약과의 차이가 아닌 건축한 시가와의 차이므로 여기에 계약간 기간 차이를 확인하는데 한계가 있음.
    return train_df


def drop_columns(df, drop_column_lst):
    for drop in drop_column_lst:
        df.drop(drop, axis=1, inplace=True)

    return df


def get_diff_one_month_house_price(df_copy):
    # 직전 한 달 같은 집에 대한 차이
    yrmonth_c_amt_df = df_copy.groupby(['COMPLEX_NAME', 'CONTRACT_YEAR_MONTH']).mean()['AMOUNT'].reset_index()

    yrmonth_c_amt_df.rename(columns={'AMOUNT': 'MEAN_YEAR_MONTH_AMOUNT'}, inplace=True)

    un_lst = yrmonth_c_amt_df['COMPLEX_NAME'].unique().tolist()

    all_complex_lst = []
    for complex_name in tqdm(un_lst, mininterval=0.01):
        temp = yrmonth_c_amt_df[yrmonth_c_amt_df['COMPLEX_NAME'] == complex_name]
        temp['COMPLEX_NAME_AMOUNT_SHIFT'] = temp['MEAN_YEAR_MONTH_AMOUNT'].shift(1)
        diff_complex_amount = temp['MEAN_YEAR_MONTH_AMOUNT'] - temp['COMPLEX_NAME_AMOUNT_SHIFT']
        temp['CHANGE_AMOUNT_COMPLEX'] = diff_complex_amount
        all_complex_lst.append(temp)

    final_df = pd.concat(all_complex_lst)
    # CHANGE_AMOUNT_COMPLEX nan은 비교 대상의 데이터가 없으므로 0으로 처리할 것
    final_df['CHANGE_AMOUNT_COMPLEX'] = final_df['CHANGE_AMOUNT_COMPLEX'].fillna(0)
    merge_df = pd.merge(df_copy,
                        final_df[['COMPLEX_NAME', 'CONTRACT_YEAR_MONTH', 'CHANGE_AMOUNT_COMPLEX']],
                        on=['COMPLEX_NAME', 'CONTRACT_YEAR_MONTH'])
    return merge_df


def filter_feature_lst(train_df):
    train_feature_lst = ['AREA', 'FLOOR', 'GU_DONG_AMOUNT_MEAN',
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
                         'DIFF_YEAR_CONSTRUCT_CONTRACT',
                         'CHANGE_AMOUNT_COMPLEX']

    train_df = train_df[train_feature_lst]
    return train_df


def preprocess_fe_existing(train_df):
    ## 기존 특징 전처리
    # CITY_DISTRICT 처리
    train_df = get_city_district_target_stats(train_df)
    # ADDRESS 처리, BUILD_1ST_NUM 처리, BUILD_2ND_NUM 처리 --> COMPLEX_NAME 처리
    train_df = get_complex_name_target_stats(train_df, ['COMPLEX_NAME', 'ADDRESS'])
    # AREA 처리 & FLOOR 처리 --> 연속형
    # CONTRACT_YEAR_MONTH 처리, CONTRACT_DAY 처리 --> 합치고 년/월/일/주말/공휴일 변수로 바꿀 것
    train_df = get_date_preprocess(train_df)
    # CONSTRUCTION_YEAR 처리 --> 얼마나 오래되었는지 변수 생성 --> 원래 변수 삭제
    train_df = get_diff_year_construct_contract(train_df)
    # ROAD_NAME 처리 --> 주변 도로 조사 --> 공원으로 가는건지 아닌지 범주 변수 생성 --> 보류(일단 삭제)
    # TERMINATION_DATE 처리 --> 삭제
    # TRANSACTION_TYPE 처리 --> 삭제
    # BROKER_LOCATION 처리 --> 삭제
    train_df = drop_columns(train_df, ['ROAD_NAME', 'TERMINATION_DATE',
                                       'TRANSACTION_TYPE', 'BROKER_LOCATION'])
    # 같은 집값 한 달 전 데이터 비교
    train_df = get_diff_one_month_house_price(train_df)

    return train_df




