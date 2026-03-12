# wann und wo 
# X_new_scaled = scaler.transform(X_new)
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import pandas as pd
import numpy as np
import joblib
import json
import plotly
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
from openai import OpenAI
from dotenv import load_dotenv
import os
print("Flask starts")

load_dotenv()

app = Flask(__name__)
app.secret_key = "supersecretkey"
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

#loading model and scaler
pipeline = joblib.load("logistic_model.pkl")

le = joblib.load("label_encoder.pkl")
#print("Classes:", le.classes_)
for index, label in enumerate(le.classes_):
    print(index, "=", label)

base_features = {
    "mixing_time": 15,
    "temperature": 70,
    "stirring_speed": 300,
    "fat_content": 10,
    "water_content": 80,
    "ph_value": 7.0
}

feature_order = list(base_features.keys())

#starting page
@app.route("/", methods=["GET", "POST"])
def home():
    print("Route called")
    message = ""
    form_values = base_features.copy()
    quality_pred = None
    prediction_label = None

    if request.method == "POST":
        try:
            for key in feature_order:
                form_values[key] = float(request.form.get(key, 0))

            features_df = pd.DataFrame([form_values], columns=feature_order)

            quality_pred = pipeline.predict(features_df)[0]
            prediction_label = le.inverse_transform([quality_pred])[0]



            message = f"Prediction of cream quality: {quality_pred:.2f}"

        except Exception as e:
            message = f"Error: {e}"

    return render_template("home.html", message=message, form_values=form_values, quality_pred=prediction_label, prediction_label=prediction_label)

    
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


"""@app.route("/heatmap", methods=["GET","POST"])
def heatmap():
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form

    x_param = data.get("x_param", "temperature")
    y_param = data.get("y_param", "ph_value")

    # fixed_params richtig parsen, falls es ein String aus Formular ist
    fixed_params = data.get("fixed_params", {})
    if isinstance(fixed_params, str):
        # z.B. "{}" oder "fat_content=10"
        try:
            fixed_params = json.loads(fixed_params)
        except:
            fixed_params = {}

    matrix = generate_heatmap_matrix(x_param, y_param, fixed_params)

    return jsonify({"matrix": matrix})"""

@app.route("/heatmap", methods=["GET", "POST"])
def heatmap():

    # Standardwerte nur für GET
    x_param = "temperature"
    y_param = "fat_content"

    if request.method == "POST":
        x_param = request.form.get("x_param", x_param) or x_param
        y_param = request.form.get("y_param", y_param) or y_param

    # Matrix erzeugen
    matrix = generate_heatmap_matrix(x_param, y_param)

    # Labels aus Ranges
    ranges = {
        "mixing_time": [5,10,15,20,25],
        "temperature": [50,60,70,80,90],
        "stirring_speed": [100,200,300,400,500],
        "fat_content": [5,10,15,20],
        "water_content": [60,70,80,90],
        "ph_value": [5.5,6,6.5,7,7.5]
    }

    x_labels = ranges[x_param]
    y_labels = ranges[y_param]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=x_labels,
            y=y_labels,
            colorscale="Viridis"
        )
    )

    fig.update_layout(
        title="Cream Quality Heatmap",
        xaxis_title=x_param,
        yaxis_title=y_param
    )

    graphJson = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template(
        "heatmap.html",
        graphJson=graphJson,
        x_param=x_param,
        y_param=y_param
    )

@app.route("/further", methods=["GET", "POST"])
def index():
    response_text = None

    if request.method == "POST":
        prompt = request.form["prompt"]

        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        response_text = response.choices[0].message.content
    print("DEBUG Params:", params)
    print("Prediction:", score)
    return render_template("index.html", response=response_text)


if __name__== "__main__":
    app.run(debug=True)