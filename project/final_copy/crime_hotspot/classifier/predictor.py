import joblib

from os.path import join
from transformer import generate_input

MODEL_ROOT = 'pretrained_model'

def predict(spots, model_file, theft_type, selected_cols, scaling_cols, ) -> None:
    model = joblib.load(join(MODEL_ROOT, model_file))
    x_input = generate_input(spots, theft_type, selected_cols, scaling_cols)
    y_pred = model.predict(x_input)
    return y_pred
