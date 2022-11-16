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


# 매매병합
def merge_index_cumsum(final_df, concat_index_cumsum_final):
    # concat_index_cumsum_final['DATE_NEW'] = pd.to_datetime(concat_index_cumsum_final['DATE_NEW'])
    # final_df['DATE'] = pd.to_datetime(final_df['DATE'])
    final_df['REGION_CODE'] = final_df['REGION_CODE'].astype('int64')
    concat_index_cumsum_final['REGION_CODE'] = concat_index_cumsum_final['REGION_CODE'].astype('int64')

    final_df_copy = pd.merge(concat_index_cumsum_final, final_df, on=['REGION_CODE', 'DATE_NEW'])

    return final_df_copy


def merge_pir_df(final_df_copy, final_pir_df):
    gangbuk = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114]
    gangnam = [201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211]

    gangbuk_df_lst = []
    for i in gangbuk:
        temp = final_df_copy[final_df_copy['REGION_CODE'] == i]
        temp_copy = pd.merge(temp, final_pir_df[['YEAR', 'FINAL_GANGBUK_PIR']],
                             on=['YEAR'])
        gangbuk_df_lst.append(temp_copy)
    gangbuk_df = pd.concat(gangbuk_df_lst)

    gangnam_df_lst = []
    for i in gangnam:
        temp = final_df_copy[final_df_copy['REGION_CODE'] == i]
        temp_copy = pd.merge(temp, final_pir_df[['YEAR', 'FINAL_GANGNAM_PIR']],
                             on=['YEAR'])
        gangnam_df_lst.append(temp_copy)
    gangnam_df = pd.concat(gangnam_df_lst)

    gangbuk_df.rename(columns={'FINAL_GANGBUK_PIR': 'INCOME_PIR'}, inplace=True)
    gangnam_df.rename(columns={'FINAL_GANGNAM_PIR': 'INCOME_PIR'}, inplace=True)

    pir_all_df = pd.concat([gangbuk_df, gangnam_df])
    return pir_all_df


def merge_consumer_df(pir_all_df, final_consumer_price_df):
    # final_consumer_price_df['DATE_NEW'] = pd.to_datetime(final_consumer_price_df['DATE']).dt.strftime('%Y-%m')
    final_consumer_price_df['REGION_CODE'] = final_consumer_price_df['REGION_CODE'].astype('int64')
    pir_all_df['REGION_CODE'] = pir_all_df['REGION_CODE'].astype('int64')

    consumer_all_df = pd.merge(pir_all_df, final_consumer_price_df,
                               on=['DATE_NEW', 'REGION_CODE'])
    return consumer_all_df


def merge_khai_df(t_khai_df, consumer_all_df):
    gangbuk = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114]
    gangnam = [201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211]

    t_khai_df.rename(columns={'index': 'DATE'}, inplace=True)
    t_khai_df['DATE_NEW'] = pd.to_datetime(t_khai_df['DATE']).dt.strftime('%Y-%m')

    gangbuk_khai_mean = t_khai_df[t_khai_df['YEAR'] == '2022']['GANGBUK_KHAI'].mean()
    gangnam_khai_mean = t_khai_df[t_khai_df['YEAR'] == '2022']['GANGNAM_KHAI'].mean()

    gangbuk_df_lst = []
    for i in gangbuk:
        temp = consumer_all_df[consumer_all_df['REGION_CODE'] == i]
        temp_copy = pd.merge(temp, t_khai_df[['DATE_NEW', 'GANGBUK_KHAI']],
                             on=['DATE_NEW'], how='outer')
        gangbuk_df_lst.append(temp_copy)
    gangbuk_df = pd.concat(gangbuk_df_lst)
    gangbuk_df = gangbuk_df.fillna(gangbuk_khai_mean)

    gangnam_df_lst = []
    for i in gangnam:
        temp = consumer_all_df[consumer_all_df['REGION_CODE'] == i]
        temp_copy = pd.merge(temp, t_khai_df[['DATE_NEW', 'GANGNAM_KHAI']], on=['DATE_NEW'], how='outer')
        gangnam_df_lst.append(temp_copy)
    gangnam_df = pd.concat(gangnam_df_lst)
    gangnam_df = gangnam_df.fillna(gangnam_khai_mean)

    gangbuk_df.rename(columns={'GANGBUK_KHAI': 'FINAL_KHAI'}, inplace=True)
    gangnam_df.rename(columns={'GANGNAM_KHAI': 'FINAL_KHAI'}, inplace=True)

    khai_all_df = pd.concat([gangbuk_df, gangnam_df])
    return khai_all_df


def merge_sale_over_jeonse(khai_all_df, all_sale_over_jeonse_df):
    # all_sale_over_jeonse_df['DATE_NEW'] = pd.to_datetime(all_sale_over_jeonse_df['DATE']).dt.strftime('%Y-%m')
    khai_all_df['REGION_CODE'] = khai_all_df['REGION_CODE'].astype('int64')
    all_sale_over_jeonse_df['REGION_CODE'] = all_sale_over_jeonse_df['REGION_CODE'].astype('int64')
    sale_over_jeonse_df = pd.merge(khai_all_df, all_sale_over_jeonse_df,
                                   on=['DATE_NEW', 'REGION_CODE'])

    return sale_over_jeonse_df


def merge_supply_demand_df(sale_over_jeonse_df, new_t_supply_demand_df):
    doxim_lst = [101, 102, 103]
    dongnam_lst = [208, 209, 210, 211]  # GANGNAM_EAST
    seonam_lst = [201, 202, 203, 204, 205, 206, 207]  # GANGNAM_WEST
    dongbuk_lst = [104, 105, 106, 107, 108, 109, 110, 111]  # GANGBUK_EAST
    seobuk_lst = [112, 113, 114]  # GANGBUK_WEST

    # new_t_supply_demand_df['DATE_NEW'] = pd.to_datetime(new_t_supply_demand_df['DATE']).dt.strftime('%Y-%m')

    new_t_supply_demand_df['YEAR'] = pd.to_datetime(new_t_supply_demand_df['DATE_NEW']).dt.year

    mean_dosim_supply = \
        new_t_supply_demand_df[new_t_supply_demand_df['YEAR'] == 2022]['GANGBUK_DOSIM'].mean()

    mean_bukeast_supply = \
        new_t_supply_demand_df[new_t_supply_demand_df['YEAR'] == 2022]['GANGBUK_EAST'].mean()

    mean_bukwest_supply = \
        new_t_supply_demand_df[new_t_supply_demand_df['YEAR'] == 2022]['GANGBUK_WEST'].mean()

    mean_nameast_supply = \
        new_t_supply_demand_df[new_t_supply_demand_df['YEAR'] == 2022]['GANGNAM_EAST'].mean()

    mean_namwest_supply = \
        new_t_supply_demand_df[new_t_supply_demand_df['YEAR'] == 2022]['GANGNAM_WEST'].mean()

    doxim_df_lst = []
    for i in doxim_lst:
        temp = sale_over_jeonse_df[sale_over_jeonse_df['REGION_CODE'] == i]
        temp_copy = pd.merge(temp, new_t_supply_demand_df[['DATE_NEW', 'GANGBUK_DOSIM']],
                             on=['DATE_NEW'], how='outer')
        doxim_df_lst.append(temp_copy)
    doxim_df = pd.concat(doxim_df_lst)
    doxim_df = doxim_df.fillna(mean_dosim_supply)

    dongnam_df_lst = []
    for i in dongnam_lst:
        temp = sale_over_jeonse_df[sale_over_jeonse_df['REGION_CODE'] == i]
        temp_copy = pd.merge(temp, new_t_supply_demand_df[['DATE_NEW', 'GANGNAM_EAST']],
                             on=['DATE_NEW'], how='outer')
        dongnam_df_lst.append(temp_copy)
    dongnam_df = pd.concat(dongnam_df_lst)
    dongnam_df = dongnam_df.fillna(mean_nameast_supply)

    seonam_df_lst = []
    for i in seonam_lst:
        temp = sale_over_jeonse_df[sale_over_jeonse_df['REGION_CODE'] == i]
        temp_copy = pd.merge(temp, new_t_supply_demand_df[['DATE_NEW', 'GANGNAM_WEST']],
                             on=['DATE_NEW'], how='outer')
        seonam_df_lst.append(temp_copy)
    seonam_df = pd.concat(seonam_df_lst)
    seonam_df = seonam_df.fillna(mean_namwest_supply)

    dongbuk_df_lst = []
    for i in dongbuk_lst:
        temp = sale_over_jeonse_df[sale_over_jeonse_df['REGION_CODE'] == i]
        temp_copy = pd.merge(temp, new_t_supply_demand_df[['DATE_NEW', 'GANGBUK_EAST']],
                             on=['DATE_NEW'], how='outer')
        dongbuk_df_lst.append(temp_copy)
    dongbuk_df = pd.concat(dongbuk_df_lst)
    dongbuk_df = dongbuk_df.fillna(mean_bukeast_supply)

    seobuk_df_lst = []
    for i in seobuk_lst:
        temp = sale_over_jeonse_df[sale_over_jeonse_df['REGION_CODE'] == i]
        temp_copy = pd.merge(temp, new_t_supply_demand_df[['DATE_NEW', 'GANGBUK_WEST']],
                             on=['DATE_NEW'], how='outer')
        seobuk_df_lst.append(temp_copy)
    seobuk_df = pd.concat(seobuk_df_lst)
    seobuk_df = seobuk_df.fillna(mean_bukwest_supply)

    doxim_df.rename(columns={'GANGBUK_DOSIM': 'SUPPLY_DEMAND'}, inplace=True)
    dongnam_df.rename(columns={'GANGNAM_EAST': 'SUPPLY_DEMAND'}, inplace=True)
    seonam_df.rename(columns={'GANGNAM_WEST': 'SUPPLY_DEMAND'}, inplace=True)
    dongbuk_df.rename(columns={'GANGBUK_EAST': 'SUPPLY_DEMAND'}, inplace=True)
    seobuk_df.rename(columns={'GANGBUK_WEST': 'SUPPLY_DEMAND'}, inplace=True)

    all_supply_df = pd.concat([doxim_df, dongnam_df])
    all_supply_df = pd.concat([all_supply_df, seonam_df])
    all_supply_df = pd.concat([all_supply_df, dongbuk_df])
    all_supply_df = pd.concat([all_supply_df, seobuk_df])

    return all_supply_df


def merge_house_occu(all_supply_df, all_region_house_occupancy_df):
    # all_region_house_occupancy_df['DATE_NEW'] = pd.to_datetime(all_region_house_occupancy_df['DATE']).dt.strftime(
    # '%Y-%m')
    all_region_house_occupancy_df['YEAR'] = pd.to_datetime(all_region_house_occupancy_df['DATE_NEW']).dt.year
    mean_h_occu = all_region_house_occupancy_df[all_region_house_occupancy_df['YEAR'] == 2022]['HOUSE_OCCUPANCY'].mean()
    mean_h_occu = int(mean_h_occu)

    all_supply_df['REGION_CODE'] = all_supply_df['REGION_CODE'].astype('int64')
    all_region_house_occupancy_df['REGION_CODE'] = all_region_house_occupancy_df['REGION_CODE'].astype('int64')

    house_occu_all_df = pd.merge(all_supply_df,
                                 all_region_house_occupancy_df[['YEAR', 'HOUSE_OCCUPANCY', 'REGION_CODE']],
                                 on=['YEAR', 'REGION_CODE'], how='outer')
    house_occu_all_df = house_occu_all_df.fillna(mean_h_occu)
    return house_occu_all_df


def merge_house_unsold(house_occu_all_df, house_unsold_df):
    # house_unsold_df['DATE_NEW'] = pd.to_datetime(house_unsold_df['DATE']).dt.strftime('%Y-%m')
    house_unsold_df['YEAR'] = pd.to_datetime(house_unsold_df['DATE_NEW']).dt.year
    mean_h_unsold = house_unsold_df[house_unsold_df['YEAR'] == 2022]['HOUSE_UNSOLD'].mean()
    mean_h_unsold = int(mean_h_unsold)

    house_occu_all_df['REGION_CODE'] = house_occu_all_df['REGION_CODE'].astype('int64')
    house_unsold_df['REGION_CODE'] = house_unsold_df['REGION_CODE'].astype('int64')
    house_unsold_df['HOUSE_UNSOLD'] = house_unsold_df['HOUSE_UNSOLD'].astype('int64')

    unsold_all_df = pd.merge(house_occu_all_df, house_unsold_df[['DATE_NEW', 'REGION_CODE', 'HOUSE_UNSOLD']],
                             on=['DATE_NEW', 'REGION_CODE'], how='outer')
    unsold_all_df = unsold_all_df.fillna(mean_h_unsold)
    return unsold_all_df


def get_kahi_test_df(test_df):
    gangbuk = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114]
    gangnam = [201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211]
    gangbuk_df_lst = []
    for i in gangbuk:
        temp = test_df[test_df['REGION_CODE'] == i]
        temp['FINAL_KHAI'] = 44.36666666666667

        gangbuk_df_lst.append(temp)
    gangbuk_df = pd.concat(gangbuk_df_lst)

    gangnam_df_lst = []
    for i in gangnam:
        temp = test_df[test_df['REGION_CODE'] == i]
        temp['FINAL_KHAI'] = 30.97666666666667

        gangnam_df_lst.append(temp)
    gangnam_df = pd.concat(gangnam_df_lst)

    test_df = pd.concat([gangbuk_df, gangnam_df])
    return test_df


def get_supply_test_df(sale_over_jeonse_df, new_t_supply_demand_df):
    doxim_lst = [101, 102, 103]
    dongnam_lst = [208, 209, 210, 211]  # GANGNAM_EAST
    seonam_lst = [201, 202, 203, 204, 205, 206, 207]  # GANGNAM_WEST
    dongbuk_lst = [104, 105, 106, 107, 108, 109, 110, 111]  # GANGBUK_EAST
    seobuk_lst = [112, 113, 114]  # GANGBUK_WEST

    # new_t_supply_demand_df['DATE_NEW'] = pd.to_datetime(new_t_supply_demand_df['DATE']).dt.strftime('%Y-%m')
    new_t_supply_demand_df['YEAR'] = pd.to_datetime(new_t_supply_demand_df['DATE_NEW']).dt.year

    g_new_t_supply_demand_df = new_t_supply_demand_df.groupby('YEAR').mean().reset_index()
    g_new_t_supply_demand_df = \
        g_new_t_supply_demand_df[g_new_t_supply_demand_df['YEAR'] == 2022]
    g_new_t_supply_demand_df['DATE_NEW'] = pd.to_datetime('2022-10')
    # g_new_t_supply_demand_df['DATE_NEW'] = pd.to_datetime(g_new_t_supply_demand_df['DATE']).dt.strftime('%Y-%m')

    doxim_df_lst = []
    for i in doxim_lst:
        temp = sale_over_jeonse_df[sale_over_jeonse_df['REGION_CODE'] == i]
        temp_copy = pd.merge(temp, g_new_t_supply_demand_df[['DATE_NEW', 'GANGBUK_DOSIM']],
                             on=['DATE_NEW'])
        doxim_df_lst.append(temp_copy)
    doxim_df = pd.concat(doxim_df_lst)

    dongnam_df_lst = []
    for i in dongnam_lst:
        temp = sale_over_jeonse_df[sale_over_jeonse_df['REGION_CODE'] == i]
        temp_copy = pd.merge(temp, g_new_t_supply_demand_df[['DATE_NEW', 'GANGNAM_EAST']],
                             on=['DATE_NEW'])
        dongnam_df_lst.append(temp_copy)
    dongnam_df = pd.concat(dongnam_df_lst)

    seonam_df_lst = []
    for i in seonam_lst:
        temp = sale_over_jeonse_df[sale_over_jeonse_df['REGION_CODE'] == i]
        temp_copy = pd.merge(temp, g_new_t_supply_demand_df[['DATE_NEW', 'GANGNAM_WEST']],
                             on=['DATE_NEW'])
        seonam_df_lst.append(temp_copy)
    seonam_df = pd.concat(seonam_df_lst)

    dongbuk_df_lst = []
    for i in dongbuk_lst:
        temp = sale_over_jeonse_df[sale_over_jeonse_df['REGION_CODE'] == i]
        temp_copy = pd.merge(temp, g_new_t_supply_demand_df[['DATE_NEW', 'GANGBUK_EAST']],
                             on=['DATE_NEW'])
        dongbuk_df_lst.append(temp_copy)
    dongbuk_df = pd.concat(dongbuk_df_lst)

    seobuk_df_lst = []
    for i in seobuk_lst:
        temp = sale_over_jeonse_df[sale_over_jeonse_df['REGION_CODE'] == i]
        temp_copy = pd.merge(temp, g_new_t_supply_demand_df[['DATE_NEW', 'GANGBUK_WEST']],
                             on=['DATE_NEW'])
        seobuk_df_lst.append(temp_copy)
    seobuk_df = pd.concat(seobuk_df_lst)

    doxim_df.rename(columns={'GANGBUK_DOSIM': 'SUPPLY_DEMAND'}, inplace=True)
    dongnam_df.rename(columns={'GANGNAM_EAST': 'SUPPLY_DEMAND'}, inplace=True)
    seonam_df.rename(columns={'GANGNAM_WEST': 'SUPPLY_DEMAND'}, inplace=True)
    dongbuk_df.rename(columns={'GANGBUK_EAST': 'SUPPLY_DEMAND'}, inplace=True)
    seobuk_df.rename(columns={'GANGBUK_WEST': 'SUPPLY_DEMAND'}, inplace=True)

    all_supply_df = pd.concat([doxim_df, dongnam_df])
    all_supply_df = pd.concat([all_supply_df, seonam_df])
    all_supply_df = pd.concat([all_supply_df, dongbuk_df])
    all_supply_df = pd.concat([all_supply_df, seobuk_df])

    return all_supply_df


def get_house_occu_test_df(all_supply_df, all_region_house_occupancy_df):
    all_copy = all_region_house_occupancy_df.copy()
    # all_copy['DATE'] = pd.to_datetime(all_copy['DATE'])
    all_copy['YEAR'] = pd.to_datetime(all_copy['DATE_NEW']).dt.year
    all_copy = all_copy[all_copy['YEAR'] == 2022]
    all_copy['HOUSE_OCCUPANCY'] = all_copy['HOUSE_OCCUPANCY'].astype('int64'
                                                                     )
    all_supply_df['REGION_CODE'] = all_supply_df['REGION_CODE'].astype('int64')
    all_copy['REGION_CODE'] = all_copy['REGION_CODE'].astype('int64')

    house_occu_all_df = pd.merge(all_supply_df, all_copy[['YEAR', 'HOUSE_OCCUPANCY', 'REGION_CODE']],
                                 on=['YEAR', 'REGION_CODE'])
    return house_occu_all_df


def get_house_unsold_test_df(house_occu_all_df, house_unsold_df):
    house_unsold_df_copy = house_unsold_df.copy()
    # house_unsold_df_copy['DATE'] = pd.to_datetime(house_unsold_df_copy['DATE'])
    house_unsold_df_copy['YEAR'] = pd.to_datetime(house_unsold_df_copy['DATE_NEW']).dt.year
    g_unsold_df = house_unsold_df_copy[house_unsold_df_copy['YEAR'] == 2022].groupby('REGION_CODE').mean().reset_index()
    g_unsold_df['HOUSE_UNSOLD'] = g_unsold_df['HOUSE_UNSOLD'].astype('int64')
    g_unsold_df['YEAR'] = g_unsold_df['YEAR'].astype('int64')

    house_occu_all_df['REGION_CODE'] = house_occu_all_df['REGION_CODE'].astype('int64')

    unsold_all_df = pd.merge(house_occu_all_df, g_unsold_df,
                             on=['YEAR', 'REGION_CODE'])
    return unsold_all_df


def preprocess_gu_feature(test_df):
    test_df['DATE_NEW'] = pd.to_datetime(test_df['DATE']).dt.strftime('%Y-%m')
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
        temp.columns = ['DATE_NEW', 'SALE_RATE', 'REGION_CODE']
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
        temp.columns = ['DATE_NEW', 'JEONSE_RATE', 'REGION_CODE']
        temp.drop(0, axis=0, inplace=True)
        temp['CUM_SUM_RATE'] = temp['JEONSE_RATE'].cumsum()
        all_region_lst_data_lst.append(temp)

    all_region_lst_df = pd.concat(all_region_lst_data_lst)
    return all_region_lst_df


def get_final_jeonse_price_index_df():
    sale_df = get_sale_price_cum_sum_index()
    jeonse_index_df = get_jeonse_price_cum_sum_index_df()
    concat_index_cumsum = \
        pd.merge(sale_df, jeonse_index_df, on=['DATE_NEW', 'REGION_CODE'])
    # Undervalued index compared to Jeonse
    # 1. sale_df jeonse_index_df 병합
    # 2. UNDERVALUE_JEONSE 생성 (전세 누적 증감률 - 매매 누적증감률)
    concat_index_cumsum['UNDERVALUE_JEONSE'] = \
        concat_index_cumsum['CUM_SUM_RATE_y'] - concat_index_cumsum['CUM_SUM_RATE_x']
    concat_index_cumsum_final = \
        concat_index_cumsum[['DATE_NEW', 'REGION_CODE', 'SALE_RATE', 'JEONSE_RATE', 'UNDERVALUE_JEONSE']]
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
        temp.columns = ['DATE_NEW', 'SALE_OVER_JEONSE', 'REGION_CODE']
        temp.drop(0, axis=0, inplace=True)
        all_region_lst_data_lst.append(temp)
    all_region_lst_df = pd.concat(all_region_lst_data_lst)
    return all_region_lst_df


def get_consumer_price_index():
    concat_index_cumsum_final = get_final_jeonse_price_index_df()
    consumer_price_index_df = pd.read_csv('/Users/jiyongkim/Downloads/consumer_price_index_df.csv', encoding='UTF8')
    consumer_price_index_df = consumer_price_index_df.dropna()
    consumer_price_index_df['DATE_NEW'] = consumer_price_index_df['DATE']
    final_consumer_price_df = pd.merge(consumer_price_index_df, concat_index_cumsum_final, on=['DATE_NEW'])
    final_consumer_price_df['SALE_CONSUMER_FLAG'] = 0
    cond = final_consumer_price_df['SALE_RATE'] > final_consumer_price_df['CONSUMER_PRICE_RATE']
    index_lst = final_consumer_price_df[cond].index.tolist()
    for i in index_lst:
        final_consumer_price_df.iloc[i, -1] = 1
    final_consumer_price_df = final_consumer_price_df[['DATE_NEW', 'REGION_CODE', 'SALE_CONSUMER_FLAG']]
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
    new_t_supply_demand_df.columns = ['DATE_NEW', 'GANGBUK_DOSIM', 'GANGBUK_EAST',
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
        temp.columns = ['DATE_NEW', 'HOUSE_OCCUPANCY', 'REGION_CODE']
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
        temp.columns = ['DATE_NEW', 'HOUSE_UNSOLD', 'REGION_CODE']
        temp['HOUSE_UNSOLD'] = temp['HOUSE_UNSOLD'].astype('str')
        temp['HOUSE_UNSOLD'] = temp.HOUSE_UNSOLD.str.replace('             ', '')
        temp['HOUSE_UNSOLD'] = temp.HOUSE_UNSOLD.str.replace('-', '')
        temp['HOUSE_UNSOLD'] = pd.to_numeric(temp['HOUSE_UNSOLD'], errors='coerce')
        temp = temp.fillna(0)
        temp['DATE_NEW'] = date_lst
        all_region_lst_data_lst.append(temp)
    all_region_house_unsold_df = pd.concat(all_region_lst_data_lst)
    return all_region_house_unsold_df
