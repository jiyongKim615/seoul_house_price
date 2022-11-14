import pandas as pd

from preprocess.feature_generation import *
from preprocess.preprocess_utils import *


def get_merge_macro_eco_features(final_df):
    us_ir_df, korea_ir_df, temp_korea_filter, temp_eu_filter, temp_china_filter, temp_usa_filter = \
        get_macro_eco_feature_utils()

    temp_korea_filter, temp_eu_filter, temp_china_filter, temp_usa_filter = \
        rename_macro_eco_features(temp_korea_filter, temp_eu_filter, temp_china_filter, temp_usa_filter)

    final_df_merge = pd.merge(final_df, us_ir_df, on=['DATE'])
    final_df_merge = pd.merge(final_df_merge, korea_ir_df, on=['DATE'])
    final_df_merge = pd.merge(final_df_merge, temp_korea_filter, on=['DATE'])
    final_df_merge = pd.merge(final_df_merge, temp_eu_filter, on=['DATE'])
    final_df_merge = pd.merge(final_df_merge, temp_china_filter, on=['DATE'])
    final_df_merge = pd.merge(final_df_merge, temp_usa_filter, on=['DATE'])

    return final_df_merge


def get_train_df():
    # 시계열 특성을 고려하여 학습/테스트 데이터 분리 진행
    train_df, val_df, test_df = get_gen_ml_train_test_df()
    train_df['TRAIN_VAL'] = 0
    val_df['TRAIN_VAL'] = 1

    all_train_df = pd.concat([train_df, val_df])
    ## 학습 데이터에 대한 처리
    all_train_df = preprocess_fe_existing(all_train_df)
    all_train_df['DATE_MONTH'] = all_train_df['DATE'].dt.strftime('%Y-%m')

    all_train_df = preprocess_gu_feature(all_train_df)
    all_train_df.dropna(inplace=True)
    y_test = all_train_df[['AMOUNT']]
    y_test.to_csv('label_df.csv')
    # 필터링된 최종 학습 데이터 및 변수
    all_train_df_filtered = filter_feature_lst(all_train_df)
    all_train_df_filtered.reset_index(drop=True, inplace=True)
    all_train_df_filtered.to_csv('all_train_df.csv')
    test_df.to_csv('test_df.csv')
    return all_train_df_filtered, y_test, test_df


def merge_nine_features_df(final_df, concat_index_cumsum_final, final_pir_df, final_consumer_price_df,
                           t_khai_df, all_sale_over_jeonse_df, new_t_supply_demand_df,
                           all_region_house_occupancy_df, house_unsold_df):
    final_df.dropna(inplace=True)
    final_df_copy = merge_index_cumsum(final_df, concat_index_cumsum_final)
    pir_all_df = merge_pir_df(final_df_copy, final_pir_df)
    consumer_all_df = merge_consumer_df(pir_all_df, final_consumer_price_df)
    khai_all_df = merge_khai_df(t_khai_df, consumer_all_df)
    sale_over_jeonse_df = merge_sale_over_jeonse(khai_all_df, all_sale_over_jeonse_df)
    all_supply_df = merge_supply_demand_df(sale_over_jeonse_df, new_t_supply_demand_df)
    house_occu_all_df = merge_house_occu(all_supply_df, all_region_house_occupancy_df)
    unsold_all_df = merge_house_unsold(house_occu_all_df, house_unsold_df)
    final_df = unsold_all_df.copy()
    return final_df


def merge_nine_features_test_df(final_df, concat_index_cumsum_final, final_pir_df, final_consumer_price_df,
                                t_khai_df, all_sale_over_jeonse_df, new_t_supply_demand_df,
                                all_region_house_occupancy_df, house_unsold_df):
    final_df_copy = merge_index_cumsum(final_df, concat_index_cumsum_final)
    pir_all_df = merge_pir_df(final_df_copy, final_pir_df)
    consumer_all_df = merge_consumer_df(pir_all_df, final_consumer_price_df)
    khai_all_df = get_kahi_test_df(consumer_all_df)
    sale_over_jeonse_df = merge_sale_over_jeonse(khai_all_df, all_sale_over_jeonse_df)
    all_supply_df = get_supply_test_df(sale_over_jeonse_df, new_t_supply_demand_df)
    house_occu_all_df = get_house_occu_test_df(all_supply_df, all_region_house_occupancy_df)
    unsold_all_df = get_house_unsold_test_df(house_occu_all_df, house_unsold_df)
    final_df = unsold_all_df.copy()
    return final_df


def preprocess_fe_existing_test_df(final_df, test_df_copy):
    region_code_lst = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
                       201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211]

    temp_2022 = final_df[final_df['YEAR'] == 2022]
    all_test_df_copy_lst = []
    for region_code in tqdm(region_code_lst, mininterval=0.01):
        temp = temp_2022[temp_2022['REGION_CODE'] == region_code]
        GU_DONG_AMOUNT_MEAN = temp['GU_DONG_AMOUNT_MEAN'].mean()
        GU_DONG_AMOUNT_MEDIAN = temp['GU_DONG_AMOUNT_MEDIAN'].mean()
        GU_DONG_AMOUNT_SKEW = temp['GU_DONG_AMOUNT_SKEW'].mean()
        GU_DONG_AMOUNT_MIN = temp['GU_DONG_AMOUNT_MIN'].mean()
        GU_DONG_AMOUNT_MAX = temp['GU_DONG_AMOUNT_MAX'].mean()
        GU_DONG_AMOUNT_MAD = temp['GU_DONG_AMOUNT_MAD'].mean()
        COMPLEX_NAME_AMOUNT_MEAN = temp['COMPLEX_NAME_AMOUNT_MEAN'].mean()
        COMPLEX_NAME_AMOUNT_MEDIAN = temp['COMPLEX_NAME_AMOUNT_MEDIAN'].mean()
        COMPLEX_NAME_AMOUNT_SKEW = temp['COMPLEX_NAME_AMOUNT_SKEW'].mean()
        COMPLEX_NAME_AMOUNT_MIN = temp['COMPLEX_NAME_AMOUNT_MIN'].mean()
        COMPLEX_NAME_AMOUNT_MAX = temp['COMPLEX_NAME_AMOUNT_MAX'].mean()
        COMPLEX_NAME_AMOUNT_MAD = temp['COMPLEX_NAME_AMOUNT_MAD'].mean()

        test_df = test_df_copy[test_df_copy['REGION_CODE'] == region_code]
        test_df['GU_DONG_AMOUNT_MEAN'] = GU_DONG_AMOUNT_MEAN
        test_df['GU_DONG_AMOUNT_MEDIAN'] = GU_DONG_AMOUNT_MEDIAN
        test_df['GU_DONG_AMOUNT_SKEW'] = GU_DONG_AMOUNT_SKEW
        test_df['GU_DONG_AMOUNT_MIN'] = GU_DONG_AMOUNT_MIN
        test_df['GU_DONG_AMOUNT_MAX'] = GU_DONG_AMOUNT_MAX
        test_df['GU_DONG_AMOUNT_MAD'] = GU_DONG_AMOUNT_MAD
        test_df['COMPLEX_NAME_AMOUNT_MEAN'] = COMPLEX_NAME_AMOUNT_MEAN
        test_df['COMPLEX_NAME_AMOUNT_MEDIAN'] = COMPLEX_NAME_AMOUNT_MEDIAN
        test_df['COMPLEX_NAME_AMOUNT_SKEW'] = COMPLEX_NAME_AMOUNT_SKEW
        test_df['COMPLEX_NAME_AMOUNT_MIN'] = COMPLEX_NAME_AMOUNT_MIN
        test_df['COMPLEX_NAME_AMOUNT_MAX'] = COMPLEX_NAME_AMOUNT_MAX
        test_df['COMPLEX_NAME_AMOUNT_MAD'] = COMPLEX_NAME_AMOUNT_MAD
        all_test_df_copy_lst.append(test_df)

    test_df = pd.concat(all_test_df_copy_lst)
    test_df = get_date_preprocess(test_df)
    test_df['DATE_MONTH'] = test_df['DATE'].dt.strftime('%Y-%m')
    return test_df


def get_gu_test_df(train_df):
    train_df_copy = train_df.copy()
    city_district_df = train_df_copy['CITY_DISTRICT'].str.split(' ', expand=True)
    city_district_df.rename(columns={0: 'CITY'}, inplace=True)
    city_district_df.rename(columns={1: 'GU'}, inplace=True)
    city_district_df.rename(columns={2: 'DONG'}, inplace=True)

    train_df_copy['CITY'] = city_district_df['CITY']
    train_df_copy['GU'] = city_district_df['GU']
    train_df_copy['DONG'] = city_district_df['DONG']
    return train_df_copy


def preprocess_test_df(final_df, test_df):
    test_df = get_amount_int(test_df)
    test_df = get_gu_test_df(test_df)
    test_df = preprocess_gu_feature(test_df)
    test_df = preprocess_fe_existing_test_df(final_df, test_df)
    test_df = filter_feature_lst_test(test_df)
    return test_df


def get_all_feature_gen_macro():
    concat_index_cumsum_final = get_final_jeonse_price_index_df()
    final_pir_df = get_final_pir_index_df()
    final_consumer_price_df = get_consumer_price_index()
    t_khai_df = get_final_khai_df()
    all_sale_over_jeonse_df = get_sale_over_jeonse_df()
    new_t_supply_demand_df = get_supply_demand_index()
    all_region_house_occupancy_df = get_house_occupancy_df()
    house_unsold_df = get_house_unsold_df()

    final_df, test_df = get_first_df()
    # test_df = preprocess_test_df(test_df)

    # 부동산 외 거시경제 지표 생성
    final_df = get_merge_macro_eco_features(final_df)
    test_df = get_merge_macro_eco_features(test_df)

    final_df = merge_nine_features_df(final_df, concat_index_cumsum_final, final_pir_df, final_consumer_price_df,
                                      t_khai_df, all_sale_over_jeonse_df, new_t_supply_demand_df,
                                      all_region_house_occupancy_df, house_unsold_df)

    test_df = merge_nine_features_test_df(test_df, concat_index_cumsum_final, final_pir_df, final_consumer_price_df,
                                          t_khai_df, all_sale_over_jeonse_df, new_t_supply_demand_df,
                                          all_region_house_occupancy_df, house_unsold_df)

    return final_df, test_df


def get_first_df():
    df = pd.read_csv('all_train_df.csv', index_col=0)
    label_y_df = pd.read_csv('label_df.csv', index_col=0)
    test_df = pd.read_csv('test_df.csv', index_col=0)
    final_df = pd.concat([df, label_y_df], axis=1)
    final_df.rename(columns={'DATE': 'DATE2'}, inplace=True)
    final_df.rename(columns={'DATE_MONTH': 'DATE'}, inplace=True)
    final_df['DATE'] = pd.to_datetime(final_df['DATE'])
    final_df['YEAR'] = final_df['DATE'].dt.year

    test_df = preprocess_test_df(final_df, test_df)
    test_df.rename(columns={'DATE': 'DATE2'}, inplace=True)
    test_df.rename(columns={'DATE_MONTH': 'DATE'}, inplace=True)
    test_df['DATE'] = pd.to_datetime(test_df['DATE'])
    test_df['YEAR'] = test_df['DATE'].dt.year
    return final_df, test_df


# 매매병합
def merge_index_cumsum(final_df, concat_index_cumsum_final):
    concat_index_cumsum_final['DATE'] = pd.to_datetime(concat_index_cumsum_final['DATE'])
    # final_df['DATE'] = pd.to_datetime(final_df['DATE'])
    final_df['REGION_CODE'] = final_df['REGION_CODE'].astype('int64')
    concat_index_cumsum_final['REGION_CODE'] = concat_index_cumsum_final['REGION_CODE'].astype('int64')

    final_df_copy = pd.merge(concat_index_cumsum_final, final_df, on=['REGION_CODE', 'DATE'])

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
    final_consumer_price_df['DATE'] = pd.to_datetime(final_consumer_price_df['DATE'])
    final_consumer_price_df['REGION_CODE'] = final_consumer_price_df['REGION_CODE'].astype('int64')
    pir_all_df['REGION_CODE'] = pir_all_df['REGION_CODE'].astype('int64')

    consumer_all_df = pd.merge(pir_all_df, final_consumer_price_df,
                               on=['DATE', 'REGION_CODE'])
    return consumer_all_df


def merge_khai_df(t_khai_df, consumer_all_df):
    gangbuk = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114]
    gangnam = [201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211]

    t_khai_df.rename(columns={'index': 'DATE'}, inplace=True)
    t_khai_df['DATE'] = pd.to_datetime(t_khai_df['DATE'])

    gangbuk_khai_mean = t_khai_df[t_khai_df['YEAR'] == '2022']['GANGBUK_KHAI'].mean()
    gangnam_khai_mean = t_khai_df[t_khai_df['YEAR'] == '2022']['GANGNAM_KHAI'].mean()

    gangbuk_df_lst = []
    for i in gangbuk:
        temp = consumer_all_df[consumer_all_df['REGION_CODE'] == i]
        temp_copy = pd.merge(temp, t_khai_df[['DATE', 'GANGBUK_KHAI']],
                             on=['DATE'], how='outer')
        gangbuk_df_lst.append(temp_copy)
    gangbuk_df = pd.concat(gangbuk_df_lst)
    gangbuk_df = gangbuk_df.fillna(gangbuk_khai_mean)

    gangnam_df_lst = []
    for i in gangnam:
        temp = consumer_all_df[consumer_all_df['REGION_CODE'] == i]
        temp_copy = pd.merge(temp, t_khai_df[['DATE', 'GANGNAM_KHAI']], on=['DATE'], how='outer')
        gangnam_df_lst.append(temp_copy)
    gangnam_df = pd.concat(gangnam_df_lst)
    gangnam_df = gangnam_df.fillna(gangnam_khai_mean)

    gangbuk_df.rename(columns={'GANGBUK_KHAI': 'FINAL_KHAI'}, inplace=True)
    gangnam_df.rename(columns={'GANGNAM_KHAI': 'FINAL_KHAI'}, inplace=True)

    khai_all_df = pd.concat([gangbuk_df, gangnam_df])
    return khai_all_df


def merge_sale_over_jeonse(khai_all_df, all_sale_over_jeonse_df):
    all_sale_over_jeonse_df['DATE'] = pd.to_datetime(all_sale_over_jeonse_df['DATE'])
    khai_all_df['REGION_CODE'] = khai_all_df['REGION_CODE'].astype('int64')
    all_sale_over_jeonse_df['REGION_CODE'] = all_sale_over_jeonse_df['REGION_CODE'].astype('int64')
    sale_over_jeonse_df = pd.merge(khai_all_df, all_sale_over_jeonse_df,
                                   on=['DATE', 'REGION_CODE'])

    return sale_over_jeonse_df


def merge_supply_demand_df(sale_over_jeonse_df, new_t_supply_demand_df):
    doxim_lst = [101, 102, 103]
    dongnam_lst = [208, 209, 210, 211]  # GANGNAM_EAST
    seonam_lst = [201, 202, 203, 204, 205, 206, 207]  # GANGNAM_WEST
    dongbuk_lst = [104, 105, 106, 107, 108, 109, 110, 111]  # GANGBUK_EAST
    seobuk_lst = [112, 113, 114]  # GANGBUK_WEST

    new_t_supply_demand_df['DATE'] = pd.to_datetime(new_t_supply_demand_df['DATE'])
    new_t_supply_demand_df['YEAR'] = new_t_supply_demand_df['DATE'].dt.year

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
        temp_copy = pd.merge(temp, new_t_supply_demand_df[['DATE', 'GANGBUK_DOSIM']],
                             on=['DATE'], how='outer')
        doxim_df_lst.append(temp_copy)
    doxim_df = pd.concat(doxim_df_lst)
    doxim_df = doxim_df.fillna(mean_dosim_supply)

    dongnam_df_lst = []
    for i in dongnam_lst:
        temp = sale_over_jeonse_df[sale_over_jeonse_df['REGION_CODE'] == i]
        temp_copy = pd.merge(temp, new_t_supply_demand_df[['DATE', 'GANGNAM_EAST']],
                             on=['DATE'], how='outer')
        dongnam_df_lst.append(temp_copy)
    dongnam_df = pd.concat(dongnam_df_lst)
    dongnam_df = dongnam_df.fillna(mean_nameast_supply)

    seonam_df_lst = []
    for i in seonam_lst:
        temp = sale_over_jeonse_df[sale_over_jeonse_df['REGION_CODE'] == i]
        temp_copy = pd.merge(temp, new_t_supply_demand_df[['DATE', 'GANGNAM_WEST']],
                             on=['DATE'], how='outer')
        seonam_df_lst.append(temp_copy)
    seonam_df = pd.concat(seonam_df_lst)
    seonam_df = seonam_df.fillna(mean_namwest_supply)

    dongbuk_df_lst = []
    for i in dongbuk_lst:
        temp = sale_over_jeonse_df[sale_over_jeonse_df['REGION_CODE'] == i]
        temp_copy = pd.merge(temp, new_t_supply_demand_df[['DATE', 'GANGBUK_EAST']],
                             on=['DATE'], how='outer')
        dongbuk_df_lst.append(temp_copy)
    dongbuk_df = pd.concat(dongbuk_df_lst)
    dongbuk_df = dongbuk_df.fillna(mean_bukeast_supply)

    seobuk_df_lst = []
    for i in seobuk_lst:
        temp = sale_over_jeonse_df[sale_over_jeonse_df['REGION_CODE'] == i]
        temp_copy = pd.merge(temp, new_t_supply_demand_df[['DATE', 'GANGBUK_WEST']],
                             on=['DATE'], how='outer')
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
    all_region_house_occupancy_df['DATE'] = pd.to_datetime(all_region_house_occupancy_df['DATE'])
    all_region_house_occupancy_df['YEAR'] = all_region_house_occupancy_df['DATE'].dt.year
    mean_h_occu = all_region_house_occupancy_df[all_region_house_occupancy_df['YEAR'] == 2022]['HOUSE_OCCUPANCY'].mean()
    mean_h_occu = int(mean_h_occu)

    all_supply_df['REGION_CODE'] = all_supply_df['REGION_CODE'].astype('int64')
    all_region_house_occupancy_df['REGION_CODE'] = all_region_house_occupancy_df['REGION_CODE'].astype('int64')

    house_occu_all_df = pd.merge(all_supply_df,
                                 all_region_house_occupancy_df[['DATE', 'HOUSE_OCCUPANCY', 'REGION_CODE']],
                                 on=['DATE', 'REGION_CODE'], how='outer')
    house_occu_all_df = house_occu_all_df.fillna(mean_h_occu)
    return house_occu_all_df


def merge_house_unsold(house_occu_all_df, house_unsold_df):
    house_unsold_df['DATE'] = pd.to_datetime(house_unsold_df['DATE'])
    house_unsold_df['YEAR'] = house_unsold_df['DATE'].dt.year
    mean_h_unsold = house_unsold_df[house_unsold_df['YEAR'] == 2022]['HOUSE_UNSOLD'].mean()
    mean_h_unsold = int(mean_h_unsold)

    house_occu_all_df['REGION_CODE'] = house_occu_all_df['REGION_CODE'].astype('int64')
    house_unsold_df['REGION_CODE'] = house_unsold_df['REGION_CODE'].astype('int64')
    house_unsold_df['HOUSE_UNSOLD'] = house_unsold_df['HOUSE_UNSOLD'].astype('int64')

    unsold_all_df = pd.merge(house_occu_all_df, house_unsold_df[['DATE', 'REGION_CODE', 'HOUSE_UNSOLD']],
                             on=['DATE', 'REGION_CODE'], how='outer')
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

    new_t_supply_demand_df['DATE2'] = pd.to_datetime(new_t_supply_demand_df['DATE'])
    new_t_supply_demand_df['YEAR'] = new_t_supply_demand_df['DATE2'].dt.year

    g_new_t_supply_demand_df = new_t_supply_demand_df.groupby('YEAR').mean().reset_index()
    g_new_t_supply_demand_df = \
        g_new_t_supply_demand_df[g_new_t_supply_demand_df['YEAR'] == 2022]
    g_new_t_supply_demand_df['DATE'] = pd.to_datetime('2022-10-01')
    g_new_t_supply_demand_df['DATE'] = pd.to_datetime(g_new_t_supply_demand_df['DATE'])

    doxim_df_lst = []
    for i in doxim_lst:
        temp = sale_over_jeonse_df[sale_over_jeonse_df['REGION_CODE'] == i]
        temp_copy = pd.merge(temp, g_new_t_supply_demand_df[['DATE', 'GANGBUK_DOSIM']],
                             on=['DATE'])
        doxim_df_lst.append(temp_copy)
    doxim_df = pd.concat(doxim_df_lst)

    dongnam_df_lst = []
    for i in dongnam_lst:
        temp = sale_over_jeonse_df[sale_over_jeonse_df['REGION_CODE'] == i]
        temp_copy = pd.merge(temp, g_new_t_supply_demand_df[['DATE', 'GANGNAM_EAST']],
                             on=['DATE'])
        dongnam_df_lst.append(temp_copy)
    dongnam_df = pd.concat(dongnam_df_lst)

    seonam_df_lst = []
    for i in seonam_lst:
        temp = sale_over_jeonse_df[sale_over_jeonse_df['REGION_CODE'] == i]
        temp_copy = pd.merge(temp, g_new_t_supply_demand_df[['DATE', 'GANGNAM_WEST']],
                             on=['DATE'])
        seonam_df_lst.append(temp_copy)
    seonam_df = pd.concat(seonam_df_lst)

    dongbuk_df_lst = []
    for i in dongbuk_lst:
        temp = sale_over_jeonse_df[sale_over_jeonse_df['REGION_CODE'] == i]
        temp_copy = pd.merge(temp, g_new_t_supply_demand_df[['DATE', 'GANGBUK_EAST']],
                             on=['DATE'])
        dongbuk_df_lst.append(temp_copy)
    dongbuk_df = pd.concat(dongbuk_df_lst)

    seobuk_df_lst = []
    for i in seobuk_lst:
        temp = sale_over_jeonse_df[sale_over_jeonse_df['REGION_CODE'] == i]
        temp_copy = pd.merge(temp, g_new_t_supply_demand_df[['DATE', 'GANGBUK_WEST']],
                             on=['DATE'])
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
    all_copy['DATE'] = pd.to_datetime(all_copy['DATE'])
    all_copy['YEAR'] = all_copy['DATE'].dt.year
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
    house_unsold_df_copy['DATE2'] = pd.to_datetime(house_unsold_df_copy['DATE'])
    house_unsold_df_copy['YEAR'] = house_unsold_df_copy['DATE2'].dt.year
    g_unsold_df = house_unsold_df_copy[house_unsold_df_copy['YEAR'] == 2022].groupby('REGION_CODE').mean().reset_index()
    g_unsold_df['HOUSE_UNSOLD'] = g_unsold_df['HOUSE_UNSOLD'].astype('int64')
    g_unsold_df['YEAR'] = g_unsold_df['YEAR'].astype('int64')

    house_occu_all_df['REGION_CODE'] = house_occu_all_df['REGION_CODE'].astype('int64')

    unsold_all_df = pd.merge(house_occu_all_df, g_unsold_df,
                             on=['YEAR', 'REGION_CODE'])
    return unsold_all_df
