## AI Job Risk Predictor 2030

# Overview

This project is an end-to-end machine learning application that predicts the risk of job automation in the future. It is designed to help users understand how vulnerable a particular job role is to automation based on factors such as creativity, analytical complexity, AI dependency, and more.

Unlike basic prediction systems, this project goes a step further by providing explanations for the predicted risk and suggesting safer alternative career options. The goal is to make the output meaningful and actionable rather than just numerical.

# Key Features

Predicts automation risk for different job roles
Provides explanations for why a job is at risk
Suggests safer alternative careers based on input
Interactive user interface built with Streamlit
Uses a structured job profile dataset for realistic predictions
Fully integrated machine learning pipeline from preprocessing to deployment


# Project Structure

ai-job-risk-predictor/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── cleaned_data.csv
│       ├── featured_data.csv
│       └── job_profiles.csv
│
├── models/
│   ├── model.pkl
│   └── columns.pkl
│
├── src/
│   ├── data/
│   │   └── preprocess.ipynb
│   ├── features/
│   │   └── build_features.ipynb
│   └── models/
│       └── train_model.ipynb
│
├── requirements.txt
└── README.md
How It Works

# The system follows a structured pipeline:

Raw data is cleaned and preprocessed
Features are engineered to capture job characteristics
A machine learning model is trained to predict automation risk
The Streamlit application takes user input and generates predictions
The system explains the result and suggests safer job alternatives

# The prediction is based on multiple factors such as:

Task repetition level
AI dependency
Creativity requirement
Social interaction level
Salary and job demand
Technologies Used
Python
Pandas and NumPy for data processing
XGBoost for machine learning
Streamlit for building the web application
Plotly for visualization
Joblib for model serialization
Installation and Setup

# Clone the repository:

git clone https://github.com/your-username/your-repo.git
cd ai-job-risk-predictor

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app/app.py
Deployment

This project is deployed using Streamlit Community Cloud.
You can access the live application here:

https://your-app-name.streamlit.app
Use Case

# This tool can be useful for:

Students planning their careers
Professionals evaluating job stability
Researchers studying automation trends
Career counselors and educators
Future Improvements
Add real-world datasets for improved accuracy
Integrate explainability tools like SHAP
Include personalized career roadmaps
Add user login and history tracking
Expand dataset with more job roles and industries
Author

Gaurav Chahal
B.Tech Computer Science Engineering

License

This project is open-source and available for educational and personal use.