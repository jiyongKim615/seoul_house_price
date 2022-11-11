import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, skew  # for some statistics
from scipy import stats

from preprocess.preprocess_all_in_one import *
from sklearn.preprocessing import LabelEncoder

all_common_feature = ['DATE',
                      'SALE_RATE',
                      'JEONSE_RATE',
                      'UNDERVALUE_JEONSE',
                      'AREA',
                      'FLOOR',
                      'GU_DONG_AMOUNT_MEAN',
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
                      'COMPLEX_NAME',
                      'REGION_CODE',
                      'INCOME_PIR',
                      'SALE_CONSUMER_FLAG',
                      'FINAL_KHAI',
                      'SALE_OVER_JEONSE',
                      'SUPPLY_DEMAND',
                      'HOUSE_OCCUPANCY',
                      'HOUSE_UNSOLD',
                      'KOR_VALUE',
                      'EU_VALUE',
                      'CN_VALUE',
                      'USA_VALUE',
                      'INTEREST_RATE',
                      'KOREA_IR',
                      'AMOUNT']

feature_x = ['SALE_RATE',
             'JEONSE_RATE',
             'UNDERVALUE_JEONSE',
             'AREA',
             'FLOOR',
             'GU_DONG_AMOUNT_MEAN',
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
             'COMPLEX_NAME',
             'REGION_CODE',
             'INCOME_PIR',
             'SALE_CONSUMER_FLAG',
             'FINAL_KHAI',
             'SALE_OVER_JEONSE',
             'SUPPLY_DEMAND',
             'HOUSE_OCCUPANCY',
             'HOUSE_UNSOLD',
             'KOR_VALUE',
             'EU_VALUE',
             'CN_VALUE',
             'USA_VALUE',
             'INTEREST_RATE',
             'KOREA_IR']

target = 'AMOUNT'


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


def redefine_train_df_test_df(num=70):
    final_df, test_df = get_all_feature_gen_macro()
    final_df = final_df[all_common_feature]
    test_df = test_df[all_common_feature]
    all_df = pd.concat([final_df, test_df])
    all_df = all_df.reset_index(drop=True)
    # 아파트 가격 로그변환

    complx_lst = all_df.COMPLEX_NAME.unique().tolist()

    train_df, test_df = utils_redefine_train_test(all_df, complx_lst, num)

    complx_train_lst = train_df.COMPLEX_NAME.unique().tolist()
    all_test_lst = []
    for complex_name in tqdm(complx_train_lst, mininterval=0.01):
        temp = test_df[test_df['COMPLEX_NAME'] == complex_name]
        all_test_lst.append(temp)

    test_df = pd.concat(all_test_lst)

    return train_df, test_df


def utils_redefine_train_test(all_df, complx_lst, num):
    temp_train_lst = []
    temp_test_lst = []

    for complex_name in tqdm(complx_lst, mininterval=0.01):
        temp = all_df[all_df['COMPLEX_NAME'] == complex_name]
        temp.sort_values('DATE', inplace=True)
        num = len(temp) * num / 100
        train_len = int(num)
        temp_train_df = temp.iloc[:train_len, :]
        temp_test_df = temp.iloc[train_len:, :]

        temp_train_lst.append(temp_train_df)
        temp_test_lst.append(temp_test_df)

    train_df = pd.concat(temp_train_lst)
    test_df = pd.concat(temp_test_lst)

    return train_df, test_df


def label_encoding(train_df_new, test_df_new):
    encoder = LabelEncoder()
    encoder.fit(train_df_new['COMPLEX_NAME'])

    train_df_new['COMPLEX_NAME'] = encoder.transform(train_df_new['COMPLEX_NAME'])
    test_df_new['COMPLEX_NAME'] = encoder.transform(test_df_new['COMPLEX_NAME'])
    return train_df_new, test_df_new
