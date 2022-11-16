from preprocess.target_encoding import TargetEncode
from renew_preprocess.re_preprocess_utils import *
from renew_preprocess.re_train_utils import feature_x1


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


def get_groupby_fe():
    return None


def get_macro_fe():
    return None


def get_park_fe():
    return None


def get_subway_time_fe():
    return None


### categories를 파라미터로 설정할 것
def get_train_test_preprocess_df(df_copy, categories):
    # 학습 데이터와 테스트 데이터 나누기 (테스트 데이터는 처음보는 데이터 개념 --> 가장 최근 데이터)
    train_df, test_df = get_train_test_df(df_copy)
    # 타겟 인코딩(Target Encoding)
    categories = ['DONG_CPX_NME', 'GU', 'DONG', 'ROAD_NAME']
    te = TargetEncode(categories=categories)
    te.fit(train_df, train_df['AMOUNT'])
    final_train = te.transform(train_df)
    final_test = te.transform(test_df)
    X = final_train[feature_x1].copy()
    X_test = final_test[feature_x1].copy()
    y = final_train['AMOUNT'].copy()
    # change data into numpy type
    X_numpy = final_train[feature_x1].to_numpy()
    y_numpy = y.copy().to_numpy().reshape(-1, 1)

    return X_numpy, y_numpy, final_test
