import pandas as pd
from tqdm import tqdm


def get_nine_feature_gen_macro():
    return None


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


def get_mean_pir_df():

    return None
