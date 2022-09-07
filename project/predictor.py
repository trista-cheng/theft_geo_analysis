import joblib

from os.path import join
from transformer import generate_input

MODEL_ROOT = 'pretrained_model'

continuous_col = ['school', 'park', 'mrt', 'rob', 'publand', 'robbery', 'urban_proj', 'monitor', 'idle']
binary_col = ['DistanceType1_School', 'DistanceType1_Park',
            'DistanceType1_MRT', 'DistanceType1_ROB', 
            'DistanceType2_School', 'DistanceType2_Park', 
            'DistanceType2_MRT', 'DistanceType2_ROB',
            'MonitorType1', 'MonitorType2',]
mix_col = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 
               'Feature5', 'Feature6', 'Feature7', 'Feature8']
all_col = continuous_col + binary_col + mix_col

target_col = ['output']

def predict(spots, model_file, theft_type, selected_cols, scaling_cols, output_type='prob') -> None:
    model = joblib.load(join(MODEL_ROOT, model_file))
    x_input = generate_input(spots, theft_type, selected_cols, scaling_cols)
    if output_type == 'prob':
        y_pred = model.predict_proba(x_input)
    else:
        y_pred = model.predict(x_input)
    return y_pred


predict([['臺北市文山區萬美里24鄰萬寧街129號',25.003557,121.569674], ['臺北市信義區富臺里6鄰忠孝東路五段295巷6弄6號',25.041779,121.572792]], 
        'home_svm.pkl', 'home', all_col, all_col, output_type='prob')