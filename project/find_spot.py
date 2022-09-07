import numpy as np
import pandas as pd
import math

def HaversineDistance(_lat1, _lon1, _lat2, _lon2):
    '''
    Euclidean Distance works for the flat surface like a Cartesian plain however,
        Earth is not flat. So we have to use a special type of formula known as Haversine Distance.

    Haversine Distance can be defined as the angular distance between two locations on the Earth’s surface.
    '''
    # radius of the Earth
    R = 6373.0

    # coordinates
    lat1 = math.radians(_lat1)
    lon1 = math.radians(_lon1)
    lat2 = math.radians(_lat2)
    lon2 = math.radians(_lon2)

    #change in coordinates
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * \
        math.cos(lat2) * math.sin(dlon / 2)**2
    # Haversine formula

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c

    return distance


def ManhattanDistance(_lat1, _lon1, _lat2, _lon2):
    pick = [_lat1, _lon2]
    distance1 = HaversineDistance(_lat1, _lon1, pick[0], pick[1])
    distance2 = HaversineDistance(_lat2, _lon2, pick[0], pick[1])
    distance = distance1 + distance2

    return distance

def read_raw_data(data_path, data_name):
    df = pd.read_csv(data_path + data_name + 'Merge.csv')
    return df

def near_point(df, new_lat, new_lng, notsave='1', num=10):
    '''
    df: 要比對的原始資料，有 0/1
    notsave: 1 回傳最近的 n 個危險點，0 回傳最近的 n 個 save 點
    return 最近的 10 筆資料
    '''
    df = df[df['output'] == notsave]
    df['min'] = df.apply(lambda x: ManhattanDistance(x['lat'], x['lng'], new_lat, new_lng), axis=1)

    return df.sort_values(by='min').iloc[0:10, :3]