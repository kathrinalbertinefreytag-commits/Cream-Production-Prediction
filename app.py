# wann und wo 
# X_new_scaled = scaler.transform(X_new)
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import json
import plotly
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
from openai import OpenAI
from dotenv import load_dotenv
import os
import seaborn as sns
import io
import base64
from flask import Flask, request, jsonify, render_template
from rag_cosmetic.rag_chain import ask_rag  
from app_data import base_features, PARAMETER_MAP
from app_prompt import detect_parameter, build_focus_schema, Full_Cream_Schema, build_cream_prompt
from app_predict import generate_heatmap_matrix

print("Flask starts")

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")
api_key = os.getenv("OPENAI_API_KEY")


client = OpenAI(api_key=api_key)


#loading model and scaler
pipeline = joblib.load("logistic_model.pkl")

le = joblib.load("label_encoder.pkl")
#print("Classes:", le.classes_)
for index, label in enumerate(le.classes_):
    print(index, "=", label)


feature_order = list(base_features.keys())
features_df = pd.DataFrame([], columns=feature_order)




#starting page
@app.route("/", methods=["GET", "POST"])
def home():
    global features_df
    print("Route called")
    message = ""
    form_values = base_features.copy()
    quality_pred = None
    prediction_label = None

    if request.method == "POST":
        try:
            for key in feature_order:
                value = request.form.get(key)

                if value == "" or value is None:
                    form_values[key] = base_features[key]
                else:
                    form_values[key] = float(value)

            features_df = pd.DataFrame([form_values], columns=feature_order)
            print(features_df)
            quality_pred = pipeline.predict(features_df)[0]
            prediction_label = le.inverse_transform([quality_pred])[0]



            message = f"Prediction of cream quality: Your creme has the quality-score {quality_pred:.2f} what means it is:"

        except Exception as e:
            message = f"Error: {e}"

    return render_template("home.html", message=message, form_values=form_values, quality_pred=prediction_label, prediction_label=prediction_label)

    



@app.route("/heatmap", methods=["GET", "POST"])
def heatmap():

    # Standardwerte nur für GET
    x_param = "temperature"
    y_param = "fat_content"

    print(x_param, y_param)

    if request.method == "POST":
        x_param = request.form.get("x_param", x_param) or x_param
        y_param = request.form.get("y_param", y_param) or y_param

        print(f"hallo!",x_param, y_param)

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
    data = None
    references = []
    user_prompt = None
    
    if request.method == "POST":
        user_prompt = request.form["prompt"]
        detected = detect_parameter(user_prompt)        
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[ {"role": "system", "content": build_cream_prompt(features_df, detected)},
                {"role": "user", "content": user_prompt}
            ], 
            response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "cream_analysis",
            "schema": Full_Cream_Schema }})
        

        data = response.choices[0].message.content

        import json
        data = json.loads(data)
        references = data.get("references", [])

    return render_template(
        "index.html",
        data=data,
        references=references,
        user_prompt=user_prompt
     )
        

@app.route("/further_rag", methods=["GET", "POST"])
def further_rag():
    response = None
    print("hallo!")
    if request.method == "POST":
        user_query = request.form.get("prompt")

        if user_query:
            response = ask_rag(user_query)

    return render_template("index.html", response=response)



if __name__== "__main__":
    app.run(debug=True)