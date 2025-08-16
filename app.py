from flask import Flask, render_template, request, url_for, redirect, session, jsonify
import joblib
from xgboost import XGBClassifier
import json
import numpy as np
import re

app = Flask(__name__)
app.secret_key = "110105103105104103104101103110"

replacement_rules_feature = {
    "Male": 1,
    "Female": 0,
    "yes": 1,
    "no": 0
}

@app.route('/')
def displayInformation():
    return "<h1><center>The bulk of this website is for the API access of the ACL Injury Website</center></h1>"

@app.route("/healthz", methods=["POST"])
def inference():
    features = request.get_json(force=True)
    
    SelectedModel = features.get("selected_model")
    CTS = features.get("CoronalTibialSlope")
    MTS = features.get("MedialTibialSlope")
    LTS = features.get("LateralTibialSlope")
    MTD = features.get("MedialTibialDepth")
    Sex = features.get("selected-sex")

    input_list = [CTS, MTS, LTS, MTD, Sex]

    inference = {"Prediction: ":perform_inference(SelectedModel, input_list)}
    
    return jsonify(inference)

def perform_inference(model_path:str, array:np.array) -> np.array:
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
    
    prediction = model.predict(array)
    return int(prediction[-1])
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
