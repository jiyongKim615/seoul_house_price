import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, skew  # for some statistics
from scipy import stats
from renew_preprocess.re_house_eco_fe_utils import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
# scaling metric: https://www.kaggle.com/code/pythonafroz/standardization-normalization-techniques

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


def get_train_df_groupby_target_stats(train_df_copy, groupby_lst, target):
    gu_dong_mean = train_df_copy.groupby(groupby_lst)[target].mean().reset_index()
    gu_dong_median = train_df_copy.groupby(groupby_lst)[target].median().reset_index()
    gu_dong_skew = train_df_copy.groupby(groupby_lst)[target].skew().reset_index()
    gu_dong_min = train_df_copy.groupby(groupby_lst)[target].min().reset_index()
    gu_dong_max = train_df_copy.groupby(groupby_lst)[target].max().reset_index()
    gu_dong_mad = train_df_copy.groupby(groupby_lst)[target].mad().reset_index()
    return gu_dong_mean, gu_dong_median, gu_dong_skew, gu_dong_min, gu_dong_max, gu_dong_mad


def get_groupby_fe_utils(train_df):
    gu_dong_mean, gu_dong_median, gu_dong_skew, gu_dong_min, \
    gu_dong_max, gu_dong_mad = \
        get_train_df_groupby_target_stats(train_df, ['GU', 'DONG'], 'AMOUNT')

    # rename feature --> stats
    gu_dong_mean.rename(columns={'AMOUNT': 'GU_DONG_AMOUNT_MEAN'}, inplace=True)
    gu_dong_median.rename(columns={'AMOUNT': 'GU_DONG_AMOUNT_MEDIAN'}, inplace=True)
    gu_dong_skew.rename(columns={'AMOUNT': 'GU_DONG_AMOUNT_SKEW'}, inplace=True)
    gu_dong_min.rename(columns={'AMOUNT': 'GU_DONG_AMOUNT_MIN'}, inplace=True)
    gu_dong_max.rename(columns={'AMOUNT': 'GU_DONG_AMOUNT_MAX'}, inplace=True)
    gu_dong_mad.rename(columns={'AMOUNT': 'GU_DONG_AMOUNT_MAD'}, inplace=True)

    return gu_dong_mean, gu_dong_median, gu_dong_skew, gu_dong_min, gu_dong_max, gu_dong_mad


def get_groupby_fe(train_df, test_df):
    gu_dong_mean, gu_dong_median, gu_dong_skew, gu_dong_min, gu_dong_max, gu_dong_mad = \
        get_groupby_fe_utils(train_df)

    train_df_copy = train_df.copy()
    test_df_copy = test_df.copy()
    train_df_copy = pd.merge(train_df_copy, gu_dong_mean, on=['GU', 'DONG'])
    train_df_copy = pd.merge(train_df_copy, gu_dong_median, on=['GU', 'DONG'])
    train_df_copy = pd.merge(train_df_copy, gu_dong_skew, on=['GU', 'DONG'])
    train_df_copy = pd.merge(train_df_copy, gu_dong_min, on=['GU', 'DONG'])
    train_df_copy = pd.merge(train_df_copy, gu_dong_max, on=['GU', 'DONG'])
    train_df_copy = pd.merge(train_df_copy, gu_dong_mad, on=['GU', 'DONG'])

    test_df_copy = pd.merge(test_df_copy, gu_dong_mean, on=['GU', 'DONG'])
    test_df_copy = pd.merge(test_df_copy, gu_dong_median, on=['GU', 'DONG'])
    test_df_copy = pd.merge(test_df_copy, gu_dong_skew, on=['GU', 'DONG'])
    test_df_copy = pd.merge(test_df_copy, gu_dong_min, on=['GU', 'DONG'])
    test_df_copy = pd.merge(test_df_copy, gu_dong_max, on=['GU', 'DONG'])
    test_df_copy = pd.merge(test_df_copy, gu_dong_mad, on=['GU', 'DONG'])

    return train_df_copy, test_df_copy


# 지하철 거리 피처 생성
def get_subway_dist(add_df):
    subway_dist_df = pd.read_csv('test_df_cpx_name_subway.csv', index_col=0)
    merge_df = pd.merge(subway_dist_df, add_df, on=['DONG', 'COMPLEX_NAME'])
    return merge_df


def get_macro_eco_feature_utils():
    us_ir_df = pd.read_csv('/Users/jiyongkim/Downloads/interest_rate.csv')
    us_ir_df['DATE_NEW'] = pd.to_datetime(us_ir_df['DATE']).dt.strftime('%Y-%m')
    korea_ir_df = pd.read_csv('/Users/jiyongkim/Downloads/korea_ir.csv')
    korea_ir_df['DATE_NEW'] = pd.to_datetime(korea_ir_df['DATE']).dt.strftime('%Y-%m')
    # composite leading indicators
    cli_df = pd.read_csv('/Users/jiyongkim/Downloads/composite_leading_indicators.csv')
    cli_df['TIME'] = pd.to_datetime(cli_df['TIME'])
    cli_df['DATE_NEW'] = pd.to_datetime(cli_df['TIME']).dt.strftime('%Y-%m')
    # preprocess (미국, 한국, 주요 유럽국, 중국)
    # Korea, OECD - Europe, China (People's Republic of), United States
    temp_korea = cli_df[cli_df['Country'] == 'Korea']
    temp_eu = cli_df[cli_df['Country'] == 'OECD - Europe']
    temp_china = cli_df[cli_df['Country'] == "China (People's Republic of)"]
    temp_usa = cli_df[cli_df['Country'] == 'United States']

    us_ir_df = us_ir_df[['INTEREST_RATE', 'DATE_NEW']]
    korea_ir_df = korea_ir_df[['KOREA_IR', 'DATE_NEW']]
    temp_korea_filter = temp_korea[['Value', 'DATE_NEW']]
    temp_eu_filter = temp_eu[['Value', 'DATE_NEW']]
    temp_china_filter = temp_china[['Value', 'DATE_NEW']]
    temp_usa_filter = temp_usa[['Value', 'DATE_NEW']]

    return us_ir_df, korea_ir_df, temp_korea_filter, temp_eu_filter, temp_china_filter, temp_usa_filter


def rename_macro_eco_utils(temp_korea_filter, temp_eu_filter, temp_china_filter, temp_usa_filter):
    # temp_korea_filter.rename(columns={'TIME': 'DATE'}, inplace=True)
    temp_korea_filter.rename(columns={'Value': 'KOR_VALUE'}, inplace=True)
    # temp_eu_filter.rename(columns={'TIME': 'DATE'}, inplace=True)
    temp_eu_filter.rename(columns={'Value': 'EU_VALUE'}, inplace=True)
    # temp_china_filter.rename(columns={'TIME': 'DATE'}, inplace=True)
    temp_china_filter.rename(columns={'Value': 'CN_VALUE'}, inplace=True)
    # temp_usa_filter.rename(columns={'TIME': 'DATE'}, inplace=True)
    temp_usa_filter.rename(columns={'Value': 'USA_VALUE'}, inplace=True)

    return temp_korea_filter, temp_eu_filter, temp_china_filter, temp_usa_filter


def get_merge_macro_eco_features(final_df):
    final_df['DATE_NEW'] = pd.to_datetime(final_df['DATE']).dt.strftime('%Y-%m')
    us_ir_df, korea_ir_df, temp_korea_filter, temp_eu_filter, temp_china_filter, temp_usa_filter = \
        get_macro_eco_feature_utils()

    temp_korea_filter, temp_eu_filter, temp_china_filter, temp_usa_filter = rename_macro_eco_utils(temp_korea_filter,
                                                                                                   temp_eu_filter,
                                                                                                   temp_china_filter,
                                                                                                   temp_usa_filter)

    final_df_merge = pd.merge(final_df, us_ir_df, on=['DATE_NEW'])
    final_df_merge = pd.merge(final_df_merge, korea_ir_df, on=['DATE_NEW'])
    final_df_merge = pd.merge(final_df_merge, temp_korea_filter, on=['DATE_NEW'])
    final_df_merge = pd.merge(final_df_merge, temp_eu_filter, on=['DATE_NEW'])
    final_df_merge = pd.merge(final_df_merge, temp_china_filter, on=['DATE_NEW'])
    final_df_merge = pd.merge(final_df_merge, temp_usa_filter, on=['DATE_NEW'])

    return final_df_merge


def get_house_eco_fe_utils():
    concat_index_cumsum_final = get_final_jeonse_price_index_df()
    final_pir_df = get_final_pir_index_df()
    final_consumer_price_df = get_consumer_price_index()
    t_khai_df = get_final_khai_df()
    all_sale_over_jeonse_df = get_sale_over_jeonse_df()
    new_t_supply_demand_df = get_supply_demand_index()
    all_region_house_occupancy_df = get_house_occupancy_df()
    house_unsold_df = get_house_unsold_df()

    return concat_index_cumsum_final, final_pir_df, final_consumer_price_df, t_khai_df, \
           all_sale_over_jeonse_df, new_t_supply_demand_df, all_region_house_occupancy_df, house_unsold_df


def get_merge_house_eco_fe(df_copy):
    concat_index_cumsum_final, final_pir_df, final_consumer_price_df, \
    t_khai_df, all_sale_over_jeonse_df, new_t_supply_demand_df, \
    all_region_house_occupancy_df, house_unsold_df = get_house_eco_fe_utils()

    final_df_copy = merge_index_cumsum(df_copy, concat_index_cumsum_final)
    pir_all_df = merge_pir_df(final_df_copy, final_pir_df)
    consumer_all_df = merge_consumer_df(pir_all_df, final_consumer_price_df)
    khai_all_df = merge_khai_df(t_khai_df, consumer_all_df)
    sale_over_jeonse_df = merge_sale_over_jeonse(khai_all_df, all_sale_over_jeonse_df)
    all_supply_df = merge_supply_demand_df(sale_over_jeonse_df, new_t_supply_demand_df)
    house_occu_all_df = merge_house_occu(all_supply_df, all_region_house_occupancy_df)
    unsold_all_df = merge_house_unsold(house_occu_all_df, house_unsold_df)
    return unsold_all_df


def get_park_df_utils(df_copy):
    park_fold = '/Users/jiyongkim/Documents/not_finish/house_price/park_data_house_study/'
    park_df = pd.read_csv(park_fold + 'park.csv')
    park_seoul_df = park_df[park_df['city'] == '서울특별시']

    park_seoul_df.rename(columns={'gu': 'GU'}, inplace=True)
    park_seoul_df.rename(columns={'dong': 'DONG'}, inplace=True)
    park_seoul_df.rename(columns={'park_name': 'PARK_NAME'}, inplace=True)

    park_seoul_df_filter = park_seoul_df[['GU', 'DONG', 'PARK_NAME']]
    merge_df = pd.merge(df_copy, park_seoul_df_filter, on=['GU', 'DONG'], how='left')
    return merge_df


def get_scaling_method(method):
    # standard, minmax, robust
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MaxAbsScaler()
    else:
        scaler = RobustScaler()

    return scaler
