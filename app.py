# wann und wo 
# X_new_scaled = scaler.transform(X_new)
from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
import joblib
import json
import plotly
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go

print("Flask starts")

app = Flask(__name__)
app.secret_key = "supersecretkey"

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
@app.route("/heatmap", methods=["GET", "POST"])
def heatmap():
    print("Heatmap Route Called")
    base_features = {
        "mixing_time": 15,
        "temperature": 70,
        "stirring_speed": 300,
        "fat_content": 10,
        "water_content": 80,
        "ph_value": 7.0
    }

    x_param = request.form.get("x_param", "temperature")
    y_param = request.form.get("y_param", "fat_content")

    x_values = np.linspace(50, 90, 20)
    y_values = np.linspace(5, 20, 20)
    
    
    heatmap_matrix = np.zeros((len(y_values), len(x_values)))
    keys = list(base_features.keys())

    for i, y_val in enumerate(y_values):
        for j, x_val in enumerate(x_values):

            temp_features = base_features.copy()
            temp_features[x_param] = x_val
            temp_features[y_param] = y_val

            features_df = pd.DataFrame([temp_features], columns=feature_order)

            heatmap_matrix[i, j] = pipeline.predict_proba(features_df)[0][1]


    fig = go.Figure(
        data = go.Heatmap(
            z=heatmap_matrix,
            x=x_values,
            y=y_values,
            colorscale='Viridis'
            )
        )

    graphJson = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    #fig = go.Figure(data=go.Heatmap(z=heatmap_matrix,x=x_values,y=y_values, colorscale='Viridis'))

    graphJson = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    print("Heatmap min:", np.min(heatmap_matrix))
    print("Heatmap max:", np.max(heatmap_matrix))
 

    return render_template(
    "heatmap.html",
    x_param=x_param,
    y_param=y_param,
    x_values=x_values,
    y_values=y_values,
    heatmap_matrix=heatmap_matrix,
    graphJson=graphJson
)

if __name__== "__main__":
    app.run(debug=True)