import pandas as pd
from app_data import base_features
import joblib

pipeline = joblib.load("logistic_model.pkl")

feature_order = list(base_features.keys())
features_df = pd.DataFrame([], columns=feature_order)

#heatmap
def predict_cream_quality(params):
    df = pd.DataFrame([params])[feature_order]
    pred = pipeline.predict(df)[0]
    return int(pred)

def generate_heatmap_matrix(x_param, y_param, fixed_params=None):
    if fixed_params is None:
        fixed_params = {}

    ranges = {
        "mixing_time": [5, 10, 15, 20, 25],
        "temperature": [50, 60, 70, 80, 90],
        "stirring_speed": [100, 200, 300, 400, 500],
        "fat_content": [5, 10, 15, 20],
        "water_content": [60, 70, 80, 90],
        "ph_value": [5.5, 6, 6.5, 7, 7.5]
    }

    x_values = ranges.get(x_param)
    y_values = ranges.get(y_param)

    defaults = {feature: base_features[feature] for feature in feature_order}
    base_params = {**defaults}

    # Wandle fixed_params in floats um
    for key, val in fixed_params.items():
        try:
            base_params[key] = float(val)
        except:
            base_params[key] = defaults.get(key, 0.0)

    matrix = []

    for y in reversed(y_values):
        row = []
        for x in x_values:
            params = base_params.copy()
            params[x_param] = float(x)
            params[y_param] = float(y)

            # DEBUG
            print("DEBUG Params:", params)

            score = predict_cream_quality(params)
            row.append(score)
        matrix.append(row)

    return matrix
