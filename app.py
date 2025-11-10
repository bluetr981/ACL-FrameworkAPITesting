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

UNLIKELY_RESULT = 0
LIKELY_RESULT = 1

predictorOne = TabularPredictor.load('models/AutoGluon_balanced_accuracy_42/', require_py_version_match = False)
predictorTwo = TabularPredictor.load('models/AutoGluon_balanced_accuracy_Full/', require_py_version_match = False)
predictorThree = TabularPredictor.load('models/AutoGluon_f1_42/', require_py_version_match = False)
predictorFour = TabularPredictor.load('models/AutoGluon_f1_Full/', require_py_version_match = False)
predictorFive = TabularPredictor.load('models/AutoGluon_f2_42/', require_py_version_match = False)
predictorSix = TabularPredictor.load('models/AutoGluon_f2_Full/', require_py_version_match = False)

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
        if request.method != "GET":
                specific_model_name = 'WeightedEnsemble_L2'
                data = request.get_json(force=True)
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
                
                input_list = np.array([cts, mts, lts, mtd, sex])
                test_df = pd.DataFrame([input_list], columns=['CTS', 'MTS', 'LTS', 'MTD', 'Sex'])

                match selected_model:
                        case "models/AutoGluon_balanced_accuracy_42/":
                                predictions = predictorOne.predict(test_df, model=specific_model_name).reset_index(drop=True)
                                predicted_probs = predictorOne.predict_proba(test_df, model=specific_model_name).reset_index(drop=True)
                        case "models/AutoGluon_balanced_accuracy_Full/":
                                predictions = predictorTwo.predict(test_df, model=specific_model_name).reset_index(drop=True)
                                predicted_probs = predictorTwo.predict_proba(test_df, model=specific_model_name).reset_index(drop=True)
                        case "models/AutoGluon_f1_42/":
                                predictions = predictorThree.predict(test_df, model=specific_model_name).reset_index(drop=True)
                                predicted_probs = predictorThree.predict_proba(test_df, model=specific_model_name).reset_index(drop=True)
                        case "models/AutoGluon_f1_Full/":
                                predictions = predictorFour.predict(test_df, model=specific_model_name).reset_index(drop=True)
                                predicted_probs = predictorFour.predict_proba(test_df, model=specific_model_name).reset_index(drop=True)
                        case "models/AutoGluon_f2_42/":
                                predictions = predictorFive.predict(test_df, model=specific_model_name).reset_index(drop=True)
                                predicted_probs = predictorFive.predict_proba(test_df, model=specific_model_name).reset_index(drop=True)
                        case "models/AutoGluon_f2_Full/":
                                predictions = predictorSix.predict(test_df, model=specific_model_name).reset_index(drop=True)
                                predicted_probs = predictorSix.predict_proba(test_df, model=specific_model_name).reset_index(drop=True)

                verdict = predictions.iloc[0]
                confidence = predicted_probs.iloc[0, verdict]

                outputs = {'Verdict':verdict, 
                           'Confidence':confidence}
        
                return jsonify(outputs)
        else:
                return "<h1><center>This API is currently not in use.</center></h1>"
        
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, require_py_version_match=False)
