from distance import *
import pandas as pd
from os.path import join

# 住家竊盜資料
DATA_ROOT = '轉經緯度後資料'
data_name_list = ['home_theft_coord', 'bicycle_coord',
                  'car_theft_coord', 'motor_theft_coord']

for data_name in data_name_list:
    print(data_name)
    df = pd.read_csv(join(DATA_ROOT, data_name + '.csv'),
                        engine='python', encoding='big5', sep='\t')
    df = df[['Response_Address', 'Response_X', 'Response_Y']].dropna()
    df = df.rename(
        columns={'Response_Address': 'address', 'Response_X': 'lng', 'Response_Y': 'lat'})

    # 假資料
    random_df = pd.read_csv('random_address.csv')

    # check 假資料 y 是否能為 0
    random_df['check'] = random_df.apply(
        lambda x: dist_check(df, x['lng'], x['lat']), axis=1)

    random_df = random_df[random_df['check']].drop(columns='check')
    random_df.to_csv('zero_data/non_' + data_name + '.csv', index=False)
