import pandas as pd
from tqdm import tqdm


def get_region_key_value(region):
    if region == '종로구':
        return 101
    elif region == '중구':
        return 102
    elif region == '용산구':
        return 103
    elif region == '성동구':
        return 104
    elif region == '광진구':
        return 105
    elif region == '동대문구':
        return 106
    elif region == '중랑구':
        return 107
    elif region == '성북구':
        return 108
    elif region == '강북구':
        return 109
    elif region == '도봉구':
        return 110
    elif region == '노원구':
        return 111
    elif region == '은평구':
        return 112
    elif region == '서대문구':
        return 113
    elif region == '마포구':
        return 114
    elif region == '양천구':
        return 201
    elif region == '강서구':
        return 202
    elif region == '구로구':
        return 203
    elif region == '금천구':
        return 204
    elif region == '영등포구':
        return 205
    elif region == '동작구':
        return 206
    elif region == '관악구':
        return 207
    elif region == '서초구':
        return 208
    elif region == '강남구':
        return 209
    elif region == '송파구':
        return 210
    elif region == '강동구':
        return 211
    else:
        return 300


def preprocess_gu_feature(test_df):
    region_lst = test_df['GU'].unique().tolist()
    all_region_lst = []
    for region_name in tqdm(region_lst, mininterval=0.01):
        test = test_df[test_df['GU'] == region_name]
        test = test.copy()
        key_value_region = get_region_key_value(region_name)
        test['REGION_CODE'] = key_value_region
        all_region_lst.append(test)
    all_test_df = pd.concat(all_region_lst)
    return all_test_df


def get_price_index_rate_df_float(sale_price_index_rate):
    sale_price_index_rate = sale_price_index_rate[:33]
    region_lst = sale_price_index_rate['지역명'].unique().tolist()
    all_region_lst = []
    for region_name in tqdm(region_lst, mininterval=0.01):
        test = sale_price_index_rate[sale_price_index_rate['지역명'] == region_name]
        test = test.copy()
        key_value_region = get_region_key_value(region_name)
        test['지역명'] = key_value_region
        test.drop(['2017-10', '2017-11', '2017-12'], axis=1, inplace=True)
        columns_lst = test.columns.tolist()
        test[columns_lst] = test[columns_lst].apply(pd.to_numeric)
        all_region_lst.append(test)
    all_region_price_index_rate = pd.concat(all_region_lst)
    return all_region_price_index_rate


def get_sale_price_cum_sum_index():
    sale_price_index_rate = pd.read_csv('/Users/jiyongkim/Downloads/monthly_sale_price_index.csv', encoding='UTF8')
    all_region_price_index_rate_df = \
        get_price_index_rate_df_float(sale_price_index_rate)
    seoul_df = all_region_price_index_rate_df[all_region_price_index_rate_df['지역명'] != 300]
    region_lst = seoul_df['지역명'].unique().tolist()
    # 2018/01 기준 전월 매매 가격 증감율에 대한 누적 피처 생성
    all_region_lst_data_lst = []
    for region_single in region_lst:
        temp = seoul_df[seoul_df['지역명'] == region_single]
        temp = temp.transpose().reset_index()
        region_code = temp.iloc[0, 1]
        temp['REGION_CODE'] = region_code
        temp.columns = ['DATE', 'SALE_RATE', 'REGION_CODE']
        temp.drop(0, axis=0, inplace=True)
        temp['CUM_SUM_RATE'] = temp['SALE_RATE'].cumsum()
        all_region_lst_data_lst.append(temp)

    all_region_lst_df = pd.concat(all_region_lst_data_lst)
    return all_region_lst_df


def get_jeonse_price_cum_sum_index_df():
    jeonse_sale_price_index_rate = pd.read_csv('/Users/jiyongkim/Downloads/jeonse_sale_price_index.csv',
                                               encoding='UTF8')
    all_region_price_index_rate_df = \
        get_price_index_rate_df_float(jeonse_sale_price_index_rate)
    seoul_df = all_region_price_index_rate_df[all_region_price_index_rate_df['지역명'] != 300]
    region_lst = seoul_df['지역명'].unique().tolist()
    # 2018/01 기준 전월 매매 가격 증감율에 대한 누적 피처 생성
    all_region_lst_data_lst = []
    for region_single in region_lst:
        temp = seoul_df[seoul_df['지역명'] == region_single]
        temp = temp.transpose().reset_index()
        region_code = temp.iloc[0, 1]
        temp['REGION_CODE'] = region_code
        temp.columns = ['DATE', 'JEONSE_RATE', 'REGION_CODE']
        temp.drop(0, axis=0, inplace=True)
        temp['CUM_SUM_RATE'] = temp['JEONSE_RATE'].cumsum()
        all_region_lst_data_lst.append(temp)

    all_region_lst_df = pd.concat(all_region_lst_data_lst)
    return all_region_lst_df


def get_final_jeonse_price_index_df():
    sale_df = get_sale_price_cum_sum_index()
    jeonse_index_df = get_jeonse_price_cum_sum_index_df()
    concat_index_cumsum = \
        pd.merge(sale_df, jeonse_index_df, on=['DATE', 'REGION_CODE'])
    # Undervalued index compared to Jeonse
    # 1. sale_df jeonse_index_df 병합
    # 2. UNDERVALUE_JEONSE 생성 (전세 누적 증감률 - 매매 누적증감률)
    concat_index_cumsum['UNDERVALUE_JEONSE'] = \
        concat_index_cumsum['CUM_SUM_RATE_y'] - concat_index_cumsum['CUM_SUM_RATE_x']
    concat_index_cumsum_final = \
        concat_index_cumsum[['DATE', 'REGION_CODE', 'SALE_RATE', 'JEONSE_RATE', 'UNDERVALUE_JEONSE']]
    return concat_index_cumsum_final


def get_mean_sale_df():
    # 평균 매매 가격 데이터 불러오기
    mean_sale_df = pd.read_csv('/Users/jiyongkim/Downloads/mean_sale_price_df.csv', encoding='UTF-8')
    t_mean_sale_df = mean_sale_df.transpose().reset_index()
    t_mean_sale_df = t_mean_sale_df.iloc[:, [0, 3, 4]]
    t_mean_sale_df.rename(columns={2: 'GANGBUK'}, inplace=True)
    t_mean_sale_df.rename(columns={3: 'GANGNAM'}, inplace=True)
    t_mean_sale_df.drop([0], axis=0, inplace=True)
    t_mean_sale_df[['YEAR', 'MONTH']] = t_mean_sale_df['index'].str.split("-", expand=True)

    g_t_mean_sale_df = t_mean_sale_df.groupby('YEAR').mean().reset_index()
    g_t_mean_sale_df.drop([0], inplace=True)
    g_t_mean_sale_df.drop(columns=['MONTH'], inplace=True)
    return g_t_mean_sale_df


def get_mean_income_df():
    # 처분가능 평득 소득
    final_mean_income_df = pd.DataFrame(columns=['YEAR', 'MEAN_INCOME'])
    final_mean_income_df.loc[0] = [2018, 3071]
    final_mean_income_df.loc[1] = [2018, 2940]
    final_mean_income_df.loc[2] = [2018, 3020]
    final_mean_income_df.loc[3] = [2018, 2991]
    final_mean_income_df.loc[4] = [2019, 3358]
    final_mean_income_df.loc[5] = [2019, 3335]
    final_mean_income_df.loc[6] = [2019, 3383]
    final_mean_income_df.loc[7] = [2019, 3462]
    final_mean_income_df.loc[8] = [2020, 3482]
    final_mean_income_df.loc[9] = [2020, 3519]
    final_mean_income_df.loc[10] = [2020, 3519]
    final_mean_income_df.loc[11] = [2020, 3542]
    final_mean_income_df.loc[12] = [2021, 3511]
    final_mean_income_df.loc[13] = [2021, 3454]
    final_mean_income_df.loc[14] = [2021, 3773]
    final_mean_income_df.loc[15] = [2021, 3783]
    final_mean_income_df.loc[16] = [2022, 3860]
    final_mean_income_df.loc[17] = [2022, 3943]

    final_mean_income_df_copy = final_mean_income_df.groupby('YEAR').mean().reset_index()
    return final_mean_income_df_copy


def get_median_sale_df():
    # 중위 매매 가격 데이터 불러오기
    median_sale_df = pd.read_csv('/Users/jiyongkim/Downloads/median_sale_price_df.csv', encoding='UTF-8')
    t_median_sale_df = median_sale_df.transpose().reset_index()
    t_median_sale_df = t_median_sale_df.iloc[:, [0, 3, 4]]
    t_median_sale_df.rename(columns={2: 'GANGBUK'}, inplace=True)
    t_median_sale_df.rename(columns={3: 'GANGNAM'}, inplace=True)

    t_median_sale_df.drop([0], axis=0, inplace=True)
    t_median_sale_df[['YEAR', 'MONTH']] = t_median_sale_df['index'].str.split("-", expand=True)
    g_t_median_sale_df = t_median_sale_df.groupby('YEAR').mean().reset_index()
    g_t_median_sale_df.drop([0], inplace=True)
    g_t_median_sale_df.drop(columns=['MONTH'], inplace=True)
    return g_t_median_sale_df


def get_median_income_df():
    # 중위 가처분 소득 데이터 불러오기
    median_income = \
        pd.read_csv('/Users/jiyongkim/Downloads/median_income.csv', encoding='UTF-8')
    t_median_income_df = median_income.transpose().reset_index()
    # 4인 가구를 기준으로 할 것
    t_median_income_final_df = t_median_income_df.iloc[4:9, [2, 6]]
    t_median_income_final_df.rename(columns={1: 'YEAR'}, inplace=True)
    t_median_income_final_df.rename(columns={5: 'MEDIAN_INCOME'}, inplace=True)
    t_median_income_final_df['MEDIAN_INCOME'] = t_median_income_final_df.MEDIAN_INCOME.str.replace(',', '').astype(
        'int64')
    return t_median_income_final_df


def get_final_pir_index_df():
    g_t_mean_sale_df = get_mean_sale_df()
    final_mean_income_df_copy = get_mean_income_df()

    g_t_mean_sale_df = g_t_mean_sale_df.astype({'YEAR': 'int'})
    final_mean_income_df_copy = final_mean_income_df_copy.astype({'YEAR': 'int'})

    mean_pir_df = pd.merge(g_t_mean_sale_df, final_mean_income_df_copy, on=['YEAR'])
    mean_pir_df['GANGBUK_MEAN_PIR'] = mean_pir_df['GANGBUK'] / mean_pir_df['MEAN_INCOME']
    mean_pir_df['GANGNAM_MEAN_PIR'] = mean_pir_df['GANGNAM'] / mean_pir_df['MEAN_INCOME']

    g_t_median_sale_df = get_median_sale_df()

    t_median_income_final_df = get_median_income_df()

    # 중위 PIR 산출
    median_pir_df = pd.merge(g_t_median_sale_df, t_median_income_final_df, on=['YEAR'])

    median_pir_df['GANGBUK_MEDIAN_PIR'] = 1000 * median_pir_df['GANGBUK'] / median_pir_df['MEDIAN_INCOME']
    median_pir_df['GANGNAM_MEDIAN_PIR'] = 1000 * median_pir_df['GANGNAM'] / median_pir_df['MEDIAN_INCOME']

    # 최종 PIR (앙상블 0.5씩)
    mean_pir_df = mean_pir_df.astype({'YEAR': 'int'})
    median_pir_df = median_pir_df.astype({'YEAR': 'int'})
    final_pir_df = pd.merge(mean_pir_df, median_pir_df, on=['YEAR'])
    final_pir_df['FINAL_GANGBUK_PIR'] = \
        final_pir_df['GANGBUK_MEAN_PIR'] * 0.5 + final_pir_df['GANGBUK_MEDIAN_PIR'] * 0.5
    final_pir_df['FINAL_GANGNAM_PIR'] = \
        final_pir_df['GANGNAM_MEAN_PIR'] * 0.5 + final_pir_df['GANGNAM_MEDIAN_PIR'] * 0.5

    # 날짜, 지역코드, FINAL_PIR 데이터프레임 생성
    final_pir_df = final_pir_df[['YEAR', 'FINAL_GANGBUK_PIR', 'FINAL_GANGNAM_PIR']]
    return final_pir_df


def get_final_khai_df():
    khai_df = pd.read_csv('/Users/jiyongkim/Downloads/k_hai.csv', encoding='UTF8')
    t_khai_df = khai_df.transpose().reset_index()
    t_khai_df.rename(columns={0: 'GANGBUK_KHAI'}, inplace=True)
    t_khai_df.rename(columns={1: 'GANGNAM_KHAI'}, inplace=True)
    t_khai_df.drop([0], inplace=True)
    t_khai_df[['YEAR', 'MONTH']] = t_khai_df['index'].str.split("-", expand=True)
    # g_t_khai_df = t_khai_df.groupby('YEAR').mean().reset_index()
    # g_t_khai_df.drop(columns=['MONTH'], inplace=True)
    return t_khai_df


def preprocess_sale_over_jeonse_df():
    # 전세가율이란 전세가격 비율의 줄임말로 전세가격을 매매가격으로 나눈 것이다.
    sale_over_jeonse_df = pd.read_csv('/Users/jiyongkim/Downloads/sale_over_jeonse.csv',
                                      encoding='UTF8')
    sale_over_jeonse_df = sale_over_jeonse_df[:33]
    region_lst = sale_over_jeonse_df['지역명'].unique().tolist()
    all_region_lst = []
    for region_name in tqdm(region_lst, mininterval=0.01):
        test = sale_over_jeonse_df[sale_over_jeonse_df['지역명'] == region_name]
        test = test.copy()
        key_value_region = get_region_key_value(region_name)
        test['지역명'] = key_value_region
        test.drop(['2017-10', '2017-11', '2017-12'], axis=1, inplace=True)
        columns_lst = test.columns.tolist()
        test[columns_lst] = test[columns_lst].apply(pd.to_numeric)
        all_region_lst.append(test)
    all_region_sale_over_jeonse = pd.concat(all_region_lst)
    return all_region_sale_over_jeonse


def get_sale_over_jeonse_df():
    all_region_sale_over_jeonse = preprocess_sale_over_jeonse_df()
    seoul_df = all_region_sale_over_jeonse[all_region_sale_over_jeonse['지역명'] != 300]
    region_lst = seoul_df['지역명'].unique().tolist()
    # 2018/01 기준 전월 매매 가격 증감율에 대한 누적 피처 생성
    all_region_lst_data_lst = []
    for region_single in region_lst:
        temp = seoul_df[seoul_df['지역명'] == region_single]
        temp = temp.transpose().reset_index()
        region_code = temp.iloc[0, 1]
        temp['REGION_CODE'] = region_code
        temp.columns = ['DATE', 'SALE_OVER_JEONSE', 'REGION_CODE']
        temp.drop(0, axis=0, inplace=True)
        all_region_lst_data_lst.append(temp)
    all_region_lst_df = pd.concat(all_region_lst_data_lst)
    return all_region_lst_df


def get_consumer_price_index():
    concat_index_cumsum_final = get_final_jeonse_price_index_df()
    consumer_price_index_df = pd.read_csv('/Users/jiyongkim/Downloads/consumer_price_index_df.csv', encoding='UTF8')
    consumer_price_index_df = consumer_price_index_df.dropna()
    final_consumer_price_df = pd.merge(consumer_price_index_df, concat_index_cumsum_final, on=['DATE'])
    final_consumer_price_df['SALE_CONSUMER_FLAG'] = 0
    cond = final_consumer_price_df['SALE_RATE'] > final_consumer_price_df['CONSUMER_PRICE_RATE']
    index_lst = final_consumer_price_df[cond].index.tolist()
    for i in index_lst:
        final_consumer_price_df.iloc[i, -1] = 1
    final_consumer_price_df = final_consumer_price_df[['DATE', 'REGION_CODE', 'SALE_CONSUMER_FLAG']]
    return final_consumer_price_df


def get_supply_demand_index():
    """
    도심권 : 중구, 종로구, 용산구
    동남권 : 강남, 서초, 송파, 강동
    서남권 : 양천, 강서, 구로, 금천, 영등포, 동작, 관악
    동북권 : 성동, 광진, 동대문, 중랑, 성북, 강북, 도봉, 노원
    서북권 : 은평, 서대문, 마포
    """
    supply_demand_index = pd.read_csv('/Users/jiyongkim/Downloads/supply_demand_index.csv',
                                      encoding='cp949')
    t_supply_demand_df = supply_demand_index.transpose().reset_index()
    new_t_supply_demand_df = t_supply_demand_df.iloc[67:, [0, 10, 11, 12, 14, 15]]
    new_t_supply_demand_df.columns = ['DATE', 'GANGBUK_DOSIM', 'GANGBUK_EAST',
                                      'GANGBUK_WEST', 'GANGNAM_WEST', 'GANGNAM_EAST']
    return new_t_supply_demand_df


def get_house_occupancy_df():
    ho_df = pd.read_csv('/Users/jiyongkim/Downloads/house_occupancy.csv', encoding='UTF8')
    ho_df = ho_df.iloc[:, 0:6]
    region_lst = ho_df['지역'].unique().tolist()
    all_region_lst_data_lst = []
    for region_single in tqdm(region_lst, mininterval=0.01):
        temp = ho_df[ho_df['지역'] == region_single]
        temp = temp.copy()
        region_code = get_region_key_value(region_single)
        temp = temp.transpose().reset_index()
        temp.drop([0], inplace=True)
        temp['REGION_CODE'] = region_code
        temp.columns = ['DATE', 'HOUSE_OCCUPANCY', 'REGION_CODE']
        all_region_lst_data_lst.append(temp)
    all_region_house_occupancy_df = pd.concat(all_region_lst_data_lst)
    return all_region_house_occupancy_df


def get_house_unsold_df():
    house_unsold2022 = pd.read_excel('/Users/jiyongkim/Downloads/house_unsold_2022.xlsx',
                                     sheet_name=2, header=2)
    region_lst = house_unsold2022.loc[1:25, '시.군.구'].tolist()
    lst = list(range(52, 109))
    house_unsold_t = house_unsold2022.iloc[1:26, lst]
    house_unsold_t['REGION'] = region_lst

    date_lst = ['2018-01', '2018-02', '2018-03', '2018-04', '2018-05', '2018-06', '2018-07', '2018-08', '2018-09', '2018-10', '2018-11', '2018-12', '2019-01', '2019-02', '2019-03', '2019-04',
                '2019-05', '2019-06', '2019-07', '2019-08', '2019-09', '2019-10', '2019-11', '2019-12', '2020-01', '2020-02', '2020-03', '2020-04',
                '2020-05', '2020-06', '2020-07', '2020-08', '2020-09', '2020-10', '2020-11', '2020-12', '2021-01', '2021-02', '2021-03', '2021-04',
                '2021-05', '2021-06', '2021-07', '2021-08', '2021-09', '2021-10', '2021-11', '2021-12', '2022-01', '2022-02', '2022-03', '2022-04',
                '2022-05', '2022-06', '2022-07', '2022-08', '2022-09']

    region_lst = house_unsold_t['REGION'].unique().tolist()
    all_region_lst_data_lst = []
    for region_single in tqdm(region_lst, mininterval=0.01):
        temp = house_unsold_t[house_unsold_t['REGION'] == region_single]
        temp = temp.copy()
        region_code = get_region_key_value(region_single)
        temp = temp.transpose().reset_index()
        temp.drop([57], inplace=True)
        temp['REGION_CODE'] = region_code
        temp.columns = ['DATE', 'HOUSE_UNSOLD', 'REGION_CODE']
        temp['HOUSE_UNSOLD'] = temp['HOUSE_UNSOLD'].astype('str')
        temp['HOUSE_UNSOLD'] = temp.HOUSE_UNSOLD.str.replace('             ', '')
        temp['HOUSE_UNSOLD'] = temp.HOUSE_UNSOLD.str.replace('-', '')
        temp['HOUSE_UNSOLD'] = pd.to_numeric(temp['HOUSE_UNSOLD'], errors='coerce')
        temp = temp.fillna(0)
        temp['DATE'] = date_lst
        all_region_lst_data_lst.append(temp)
    all_region_house_unsold_df = pd.concat(all_region_lst_data_lst)
    return all_region_house_unsold_df
