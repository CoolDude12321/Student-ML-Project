# src/app.py
# =================================================
# This is the main application file for the Flask web application.
# It defines the routes and logic for handling user requests, rendering templates, and making predictions using the machine learning model.
# =================================================


# =============================================
# Importing the Necessary Libraries
# ---------------------------------------------
# Pickle for object serialization
# Flask for creating the web application and handling requests
# Numpy for numerical operations
# Pandas for data manipulation and analysis
# OS for interacting with the operating system
# StandardScaler for feature scaling
# src.Pipeline.predict_pipeline for the CustomData class and PredictPipeline class to handle data input and prediction logic
# =============================================
import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from src.Pipeline.predict_pipeline import CustomData, PredictPipeline 


# =============================================
# Flask Application Setup
# ---------------------------------------------
# This section sets up the Flask application and defines the routes for the home page and prediction endpoint.
# The home page route renders the index.html template, while the prediction route handles both GET and POST requests to render the home.html template and display the prediction results.
# =============================================
application = Flask(__name__)
app = application

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender = request.form.get('gender'),
            race_ethnicity = request.form.get('race_ethnicity'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch = request.form.get('lunch'),
            test_preparation_course = request.form.get('test_preparation_course'),
            writing_score = float(request.form.get('writing_score')),
            reading_score = float(request.form.get('reading_score'))
      )

        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
