import joblib
from xgboost import XGBClassifier
import numpy as np
import re

replacement_rules_feature = {
    "M": 1,
    "F": 0,
    "yes": 1,
    "no":0
}

def perform_inference(model_path:str, input:np.array) -> np.array:

    match = re.search(r'\[([\d,\s]+)\]', model_path)
    indices_str = match.group(1)
    indices = [int(idx.strip()) -1 for idx in indices_str.split(',')]
    input = input[:, indices]
    means = np.array([2.8280, 5.8495, 7.1613, 2.4280, 0.4194]).reshape(1, -1)[:, indices]
    stds = np.array([2.0196, 3.2234, 3.0335, 1.0566, 0.4961]).reshape(1, -1)[:, indices]
    input = (input - means) / stds

    if model_path.endswith('joblib'):
        model = joblib.load(model_path)
    elif model_path.endswith('.json'):
        model = XGBClassifier()
        model.load_model(model_path)
    
    prediction = model.predict(input)
    return prediction