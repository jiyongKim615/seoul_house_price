import pandas as pd
import re
import os


def concat_file(type_name, num, folder_PATH):
    os.chdir(folder_PATH)
    budongsan_01 = pd.read_csv('{0}_20{1}01.csv'.format(type_name, num))  # 부동산_201801.csv
    budongsan_02 = pd.read_csv('{0}_20{1}02.csv'.format(type_name, num))
    budongsan_03 = pd.read_csv('{0}_20{1}03.csv'.format(type_name, num))
    budongsan_04 = pd.read_csv('{0}_20{1}04.csv'.format(type_name, num))
    budongsan_05 = pd.read_csv('{0}_20{1}05.csv'.format(type_name, num))
    budongsan_06 = pd.read_csv('{0}_20{1}06.csv'.format(type_name, num))
    budongsan_07 = pd.read_csv('{0}_20{1}07.csv'.format(type_name, num))
    budongsan_08 = pd.read_csv('{0}_20{1}08.csv'.format(type_name, num))
    budongsan_09 = pd.read_csv('{0}_20{1}09.csv'.format(type_name, num))
    budongsan_10 = pd.read_csv('{0}_20{1}10.csv'.format(type_name, num))
    budongsan_11 = pd.read_csv('{0}_20{1}11.csv'.format(type_name, num))
    budongsan_12 = pd.read_csv('{0}_20{1}12.csv'.format(type_name, num))

    budongsan_01['date'] = '201801'
    budongsan_02['date'] = '201802'
    budongsan_03['date'] = '201803'
    budongsan_04['date'] = '201804'
    budongsan_05['date'] = '201805'
    budongsan_06['date'] = '201806'
    budongsan_07['date'] = '201807'
    budongsan_08['date'] = '201808'
    budongsan_09['date'] = '201809'
    budongsan_10['date'] = '201810'
    budongsan_11['date'] = '201811'
    budongsan_12['date'] = '201812'

    budong_san_2018 = pd.concat([budongsan_01, budongsan_02])
    budong_san_2018 = pd.concat([budong_san_2018, budongsan_03])
    budong_san_2018 = pd.concat([budong_san_2018, budongsan_04])
    budong_san_2018 = pd.concat([budong_san_2018, budongsan_05])
    budong_san_2018 = pd.concat([budong_san_2018, budongsan_06])
    budong_san_2018 = pd.concat([budong_san_2018, budongsan_07])
    budong_san_2018 = pd.concat([budong_san_2018, budongsan_08])
    budong_san_2018 = pd.concat([budong_san_2018, budongsan_09])
    budong_san_2018 = pd.concat([budong_san_2018, budongsan_10])
    budong_san_2018 = pd.concat([budong_san_2018, budongsan_11])
    budong_san_2018 = pd.concat([budong_san_2018, budongsan_12])

    return budong_san_2018


def clean_df(df_test):
    all_lst = []
    for text in df_test:
        text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…《\》]', '', text)
        all_lst.append(text.replace('\t', '').replace('\n', '').replace('\\', ''))
    final_df = pd.DataFrame()
    final_df['title'] = all_lst
    return final_df


# 윗 과정 모듈화
def get_concat_df(num):
    df = concat_file(num)
    df_test = df[['title']]
    final_df = clean_df(df_test)
    return final_df
