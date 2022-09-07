import pandas as pd
from distance import dist_min

col_list = ['school', 'park', 'mrt', 'rob']
data_name_list = ['home', 'bicycle','car', 'non_motor',
                  'non_home', 'non_bicycle','non_car', 'non_motor']


for data_name in data_name_list:
    district_df = pd.read_csv('merge/' + data_name + '_district.csv')

    for col_name in col_list:
        print(data_name, col_name)

        df = pd.read_csv('clean_data/' + col_name + '.csv')

        district_df[col_name] = district_df.apply(
            lambda x: dist_min(df, x['lng'], x['lat']), axis=1)

    district_df.to_csv('merge/' + data_name + '.csv', index=False)
