from flask import Flask, render_template, request, url_for, redirect, session, jsonify
from flask_cors import CORS
import joblib
from xgboost import XGBClassifier
import numpy as np
import re

app = Flask(__name__)
CORS(app)
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

@app.route("/healthz", methods=["GET", "POST"])
def inference():
    if (request.method != "GET"):
        data = request.get_json(force=True)
    
        selected_model = data.get("selected-model")
        
        if (data.get("CoronalTibialSlope") != "null"):
            cts = float(data.get("CoronalTibialSlope"))
        else:
            cts = -1
            
        if (data.get("MedialTibialSlope") != "null"):
            mts = float(data.get("MedialTibialSlope"))
        else:
            mts = -1
            
        if (data.get("LateralTibialSlope") != "null"):
            lts = float(data.get("LateralTibialSlope"))
        else:
            lts = -1
            
        if (data.get("MedialTibialDepth") != "null"):
            mtd = float(data.get("MedialTibialDepth"))
        else:
            mtd = -1
            
        if (data.get("selected-sex") != "null"):
            sex = int(replacement_rules_feature.get(data.get("selected-sex")))
        else:
            sex = -1

        input_list = np.array([cts, mts, lts, mtd, sex]).reshape(1, -1)
    
        return jsonify({'Prediction':perform_inference(selected_model, input_list)})
    else:
        return "<h1><center>This API is currently not in use.</center></h1>"

def perform_inference(model_path:str, input:np.array):
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
    inference = prediction.tolist()
    
    return int(inference[-1])
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
