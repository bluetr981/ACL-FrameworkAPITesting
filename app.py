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

@app.route('/healthz')
def displayInformation():
    return "<h1>The bulk of this website is for the API access of the ACL Injury Website</h1>"
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
