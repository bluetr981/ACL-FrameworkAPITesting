from flask import Flask, render_template, request, url_for, redirect, session, jsonify
from flask_cors import CORS
import numpy as np
import re

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.core.metrics import make_scorer
from sklearn.metrics import fbeta_score
from autogluon.tabular import TabularPredictor
import shap
import torch
import matplotlib.pyplot as plt
        
app = Flask(__name__)
CORS(app)
app.secret_key = "110105103105104103104101103110"

replacement_rules_feature = {
    "Male": 1,
    "Female": 0,
    "yes": 1,
    "no": 0
}

class AutogluonWrapper:
    def __init__(self, predictor, feature_names, model_name):
        self.ag_model = predictor
        self.feature_names = feature_names
        self.model_name = model_name

    def predict_binary_prob(self, X):
        if isinstance(X, pd.Series):
            X = X.values.reshape(1, -1)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        return self.ag_model.predict_proba(X, model=self.model_name, as_multiclass=False)
        
@app.route('/')
def displayInformation():
    return "<h1><center>The bulk of this website is for the API access of the ACL Injury Website</center></h1>"

@app.route("/healthz", methods=["GET", "POST"])
def inference():
    if (request.method != "GET"):
            return jsonify({'Output':'This is a POST Request.'})
    return jsonify({'Output':'This is a GET REQUEST!'})
        
        '''data = request.get_json(force=True)
    
        selected_model = data.get("selected-model")
        
        if (data.get("CoronalTibialSlope") != "-1"):
            cts = float(data.get("CoronalTibialSlope"))
        else:
            cts = -1
            
        if (data.get("MedialTibialSlope") != "-1"):
            mts = float(data.get("MedialTibialSlope"))
        else:
            mts = -1
            
        if (data.get("LateralTibialSlope") != "-1"):
            lts = float(data.get("LateralTibialSlope"))
        else:
            lts = -1
            
        if (data.get("MedialTibialDepth") != "-1"):
            mtd = float(data.get("MedialTibialDepth"))
        else:
            mtd = -1
            
        if (data.get("selected-sex") != "-1"):
            sex = int(replacement_rules_feature.get(data.get("selected-sex")))
        else:
            sex = -1

        input_list = np.array([cts, mts, lts, mtd, sex]).reshape(1, -1)
    
        return jsonify({'Outputs':perform_inference(selected_model, input_list)})
    else:
        return "<h1><center>This API is currently not in use.</center></h1>'''

def perform_inference(predictor_path:str, input:np.array):
        return predictor_path
        
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
