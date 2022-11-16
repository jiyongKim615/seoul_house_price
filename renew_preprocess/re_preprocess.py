from preprocess.target_encoding import TargetEncode
from renew_preprocess.re_preprocess_utils import *
from renew_preprocess.re_train_utils import *
from renew_preprocess.re_house_eco_fe_utils import *


def get_raw_preprocess_df():
    # RAW 데이터 불러오기
    list_data = get_raw_house_data()
    # 변수명 바꾸기
    df = get_rename(list_data)
    # AMOUNT(target) 전처리
    df_copy = get_amount_int(df)
    # 시/구/동 변수 생성 --> 다 서울시로 한정했으므로 CITY는 제외할 것
    df_copy = get_gu_dong_fe(df_copy)
    # complex_name에 지역이 다른데 중복된 이름이 될 수 있음
    # --> 지역과 결합해서 진행 --> 동 + COMPLEX_NAME
    df_copy = get_dong_cpx_nme_fe(df_copy)
    # ROAD_NAME 첫 번째 가져오기
    df_copy = get_road_name_first_only(df_copy)
    # 시간 관련 변수 생성 (YEAR, MONTH, MONTH_SIN, MONTH_COS)
    df_copy = create_feature_time(df_copy, col='DATE')
    # target log transform
    df_copy = get_log_transform(df_copy, 'AMOUNT')
    # BUILD_1ST_NUM열의 결측값 제거
    df_copy.dropna(subset=['BUILD_1ST_NUM'], inplace=True)
    return df_copy


### categories를 파라미터로 설정할 것
def get_train_test_preprocess_df(df_copy, categories=['DONG_CPX_NME', 'GU', 'DONG', 'ROAD_NAME'],
                                 subway_dist_fe=True,
                                 groupby_fee_add=True, macro_eco_fe=True, house_eco=True, park_fe=True, fe_norm=True,
                                 method='minmax', time_split=False):
    """
    :param feature_select: 'raw', 'group_fe', 'macro_economics'
    """
    # standard, minmax, robust
    # 학습 데이터와 테스트 데이터 나누기 (테스트 데이터는 처음보는 데이터 개념 --> 가장 최근 데이터)
    if macro_eco_fe:
        df_copy = get_merge_macro_eco_features(df_copy)

    if subway_dist_fe:
        df_copy = get_subway_dist(df_copy)

    if house_eco:
        df_copy = preprocess_gu_feature(df_copy)
        df_copy = get_merge_house_eco_fe(df_copy)

    if park_fe:
        df_copy = get_park_df_utils(df_copy)

    train_df, test_df = get_train_test_df(df_copy)

    if groupby_fee_add:
        train_df, test_df = get_groupby_fe(train_df, test_df)

    # 타겟 인코딩(Target Encoding)
    # categories = ['DONG_CPX_NME', 'GU', 'DONG', 'ROAD_NAME']
    te = TargetEncode(categories=categories)
    te.fit(train_df, train_df['AMOUNT'])
    final_train = te.transform(train_df)
    final_test = te.transform(test_df)

    '''
    if feature_select == 'raw':
        final_feature_x = feature_x1
    elif feature_select == 'group_fe':
        final_feature_x = feature_x2
    else:
        final_feature_x = feature_x3
    '''
    final_feature_x = feature_x1
    if fe_norm:
        scaler = get_scaling_method(method)

        y = final_train['AMOUNT'].copy()
        # change data into numpy type
        X_numpy = final_train[final_feature_x].to_numpy()
        X_numpy = scaler.fit_transform(X_numpy)

        X_test = final_test[final_feature_x].copy().to_numpy()
        X_test = scaler.transform(X_test)
        y_numpy = y.copy().to_numpy().reshape(-1, 1)
        y_test = final_test['AMOUNT'].values
    else:
        # change data into numpy type
        X_numpy = final_train[final_feature_x].to_numpy()
        y = final_train['AMOUNT'].copy()
        y_numpy = y.copy().to_numpy().reshape(-1, 1)
        X_test = final_test[final_feature_x].copy().values
        y_test = final_test['AMOUNT'].values

    return X_numpy, y_numpy, X_test, y_test, final_test
