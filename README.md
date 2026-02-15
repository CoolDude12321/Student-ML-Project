# 🎓 Student-Marks-Predictor

A production-ready machine learning web application that predicts student math scores using demographic and academic features, deployed on AWS with a fully automated CI/CD pipeline.

---

## 📌 Project Overview

This project demonstrates how machine learning models can be deployed into production using **Flask, AWS Elastic Beanstalk, and automated CI/CD pipelines**. The system processes structured student data, applies preprocessing transformations, and predicts math scores using a trained regression model. The application is lightweight, scalable, and built with a clean modular ML architecture suitable for portfolio and real-world deployment scenarios.

---

## ✨ Features

### 🎯 Regression-Based Score Prediction

Predicts **Math Score** based on:

- Gender  
- Race/Ethnicity  
- Parental Level of Education  
- Lunch Type  
- Test Preparation Course  
- Writing Score  
- Reading Score  

---

### 🧹 Data Processing & Feature Engineering

- Handles missing values  
- Applies:
  - One-Hot Encoding for categorical variables  
  - Standard Scaling for numerical variables  
- Uses a reusable preprocessing pipeline  
- Saves preprocessor object for inference reuse  

---

### 🧠 Model Training & Evaluation

- Evaluated multiple regression models  
- Selected best model based on **R² Score**  
- Saved trained model using `pickle`  
- Separated training and inference pipelines  

---

### 💾 Model & Artifact Serialization

- Stores trained model (`model.pkl`)  
- Stores preprocessing pipeline (`preprocessor.pkl`)  
- Eliminates retraining during deployment  
- Optimized for inference-only production usage  

---

### 🌐 Web Application (Flask)

- Clean and minimal UI  
- User-friendly form input  
- Displays predicted math score  
- Hosted on AWS Elastic Beanstalk  
- Powered by Gunicorn WSGI server  

---

### ☁ Fully Automated CI/CD Pipeline

- GitHub → AWS CodePipeline → CodeBuild → Elastic Beanstalk  
- Automatic deployment on every push  
- No manual zip uploads required  
- Versioned deployments via Elastic Beanstalk  

---

### ⚡ Lightweight & Cloud-Optimized

- Optimized for AWS t3.micro instance  
- Removed heavy experimental libraries  
- Training separated from production runtime  
- Fast startup and low memory usage  

---

### 🧩 Modular & Scalable Codebase

The project is built in clearly separated stages:

1. Data Ingestion  
2. Data Transformation  
3. Model Training  
4. Prediction Pipeline  
5. Web App Deployment  
6. CI/CD Automation  

---

## 📁 Project Structure

- .ebextensions/
  - python.config
- Notebook/
  - Dataset/
    - Student.csv
  - catboost_info/
    - learn
    - catboost_training.json
    - learn_error.tsv
    - time_left.tsv
  - EDA Student Performance.ipynb
  - Model Training.ipynb
- artifacts/
  - data.csv
  - model.pkl
  - preprocessor.pkl
  - test.csv
  - train.csv
- src/
  - components/
    - __init__.py
    - data_ingestion.py
    - data_transformation.py
    - model_trainer.py
  - Pipeline/
    - __init__.py
    - predict_pipeline.py
    - train_pipeline.py
  - __init__.py
  - exception.py
  - logger.py
  - utils.py
- templates/
  - home.html
  - index.html
- .gitignore
- Procfile
- README.md
- app.py
- application.py
- buildspec.yml
- requirements.txt
- setup.py

---

## 📊 Model Performance

- Evaluation Metric: **R² Score**  
- Final Selected Model: Linear Regression  
- Designed for structured tabular data prediction  

---

## 🛠️ Tools & Technologies Used

- **Python 3.11**  
- **Flask**  
- **Scikit-learn**  
- **Pandas & NumPy**  
- **Gunicorn**  
- **AWS Elastic Beanstalk**  
- **AWS CodePipeline**  
- **AWS CodeBuild**  
- **GitHub**  

---

## ▶️ How to Run the Project Locally

pip install -r requirements.txt
python app.py

---

## ▶️ How to Run the Project through URL

http://student-marks-predictor.us-east-1.elasticbeanstalk.com

---

## 👤 Author
- Prakhar Srivastava
- Aspiring Data Scientist & Business Analyst | Machine Learning, Deep Learning & Generative AI Enthusiast
