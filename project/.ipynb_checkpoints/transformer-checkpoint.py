import pandas as pd
import numpy as np

from os.path import join
from distance import dist_min
from sklearn.preprocessing import StandardScaler

NON_DISTRICT_COL = ['school', 'park', 'mrt', 'rob']
DATA_ROOT = 'data'
PROPERTY_ROOT = 'property'

# this function may not be called, since we use the pretrained model
def read_data(type):
    raw_df = pd.read_csv(join(DATA_ROOT, f'{type}.csv'))
    return raw_df

def merge_district(spot_df, district_name='district.xlsx', PROPERTY_ROOT=PROPERTY_ROOT):

    spot_df['district'] = spot_df['address'].str.split('市').str[1].str.split('區').str[0] + '區'
    district_df = pd.read_excel(join(PROPERTY_ROOT, district_name), sheet_name='工作表1')
    if 'district' not in district_df.columns:
        district_df = district_df.rename(columns={'市轄區': 'district'})
    merge_df = pd.merge(spot_df, district_df, how='left', on='district')

    return merge_df

def merge_non_district(df, PROPERTY_ROOT=PROPERTY_ROOT, columns=NON_DISTRICT_COL):
    for col_name in columns:
        property = pd.read_csv(join(PROPERTY_ROOT, col_name + '.csv'))
        df[col_name] = df.apply(lambda x: dist_min(property, x['lat'], x['lng']), axis=1)
    return df

# if spot contains address or not could be discussed 
def generate_features(spots, theft_type):  # spots should be list or dict
    df = pd.DataFrame(spots, columns=['address', 'lat', 'lng'])

    # district
    df = merge_district(df)
    # non-district
    df = merge_non_district(df)


    known_df = read_data(theft_type)
    # full feature
    for col in NON_DISTRICT_COL:
        if len(col) > 3:
            df[f'DistanceType1_{col.capitalize()}'] = df[col] < known_df[col].mean() + known_df[col].std()
            df[f'DistanceType2_{col.capitalize()}'] = df[col] < known_df[col].mean()
        else:
            df[f'DistanceType1_{col.upper()}'] = df[col] < known_df[col].mean() + known_df[col].std()
            df[f'DistanceType2_{col.upper()}'] = df[col] < known_df[col].mean()

    df['MonitorType1'] = df['monitor'] > known_df['monitor'].mean() + known_df['monitor'].std()
    df['MonitorType2'] = df['monitor'] > known_df['monitor'].mean()

    df['Feature1'] = df['school'] * df['park'] * df['mrt']
    df['Feature2'] = (df[[c for c in df.columns if c.startswith('DistanceType')]]).sum(axis=1)
    left = df[[c for c in df.columns if (c.startswith('DistanceType1') and not c.endswith('ROB'))]]
    right = df[[c for c in df.columns if (c.startswith('DistanceType2') and not c.endswith('ROB'))]]
    df['Feature3'] = (left.to_numpy() * right.to_numpy()).sum(axis=1)
    df['Feature4'] = (df['school'] + df['park'] + df['mrt']) / (df['robbery'] + 1)
    df['Feature5'] = df['robbery'] / df['monitor']
    df['Feature6'] = df['publand'] / df['monitor']
    df['Feature7'] = np.log(df['publand'] * (df['urban_proj'] - df['idle']))
    df['Feature8'] = ((df[[c for c in df.columns if c.startswith('DistanceType1')]]).sum(axis=1) ==\
                      (df[[c for c in df.columns if c.startswith('DistanceType2')]]).sum(axis=1))

    return df



def generate_input(spots, theft_type, columns, scaling_cols=None):
    df = generate_features(spots, theft_type)
    df = df[columns]
    if scaling_cols:
        known_df = read_data(theft_type)
        for col in scaling_cols:
            scaler = StandardScaler()
            scaler.fit(known_df[col].to_numpy().reshape(-1, 1))
            df[col] = scaler.transform(df[col].to_numpy().reshape(-1, 1))

    return df






