# AI Job Risk Predictor 2030

## Overview

This project is an end-to-end machine learning application that predicts the risk of job automation in the future.

It helps users understand how vulnerable a job role is based on factors such as:
- Creativity
- Analytical complexity
- AI dependency
- Social interaction
- Task repetition

Unlike basic prediction tools, this system also explains *why* a job is at risk and suggests safer alternative career paths.

---

## Key Features

- Predicts automation risk for different job roles  
- Explains why a job is at risk (Explainable AI approach)  
- Suggests safer alternative careers  
- Interactive web interface built with Streamlit  
- End-to-end machine learning pipeline  
- Realistic job profile dataset integration  

---

## Project Structure
```
ai-job-risk-predictor/
│
├── app/
│ └── streamlit_app.py
│
├── data/
│ ├── raw/
│ └── processed/
│ ├── cleaned_data.csv
│ ├── featured_data.csv
│ └── job_profiles.csv
│
├── models/
│ ├── model.pkl
│ └── columns.pkl
│
├── src/
│ └── data/
│ └── preprocess.ipynb
│
├── features/
│ └── build_features.ipynb
│
├── models/
│ └── train_model.ipynb
│
├── utils/
│ └── helpers.py
│
├── requirements.txt
└── README.md

```
---

## How It Works

The system follows a structured pipeline:

1. Raw data is cleaned and preprocessed  
2. Features are engineered to capture job characteristics  
3. A machine learning model is trained using XGBoost  
4. The Streamlit app collects user input  
5. The model predicts automation risk  
6. The system explains the result and suggests safer roles  

---

## Technologies Used

- Python  
- Pandas, NumPy  
- XGBoost  
- Streamlit  
- Plotly  
- Joblib  

---

## Installation and Setup

Clone the repository:


git clone https://github.com/your-username/ai-job-risk-predictor-2030.git

cd ai-job-risk-predictor-2030

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app/streamlit_app.py
Deployment

This project is deployed using Streamlit Community Cloud.

Live App:
https://your-app-name.streamlit.app

Use Cases
Students planning their careers
Professionals analyzing job stability
Career counselors
Researchers studying automation trends
Future Improvements
Integration of real-world datasets
SHAP-based explainability
Personalized career roadmap
User login and history tracking
Author

Gaurav Chahal
B.Tech Computer Science Engineering

License

This project is open-source and intended for educational use.
