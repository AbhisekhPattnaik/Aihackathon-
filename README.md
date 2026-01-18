Jet Engine Predictive Maintenance
Project Overview

The Jet Engine Predictive Maintenance project aims to predict the Remaining Useful Life (RUL) of jet engines using historical sensor data. This helps in preventing unexpected failures, optimizing maintenance schedules, and reducing downtime in aviation operations.

We use machine learning models to analyze engine sensor readings and flight parameters to estimate when maintenance will be required.

Features

Predicts the Remaining Useful Life (RUL) of jet engines.

Uses sensor and operational data for analysis.

Provides an interactive interface for testing engine data in Streamlit.

Visualizes predictions and maintenance status.

Project Structure
jet_engine_predictive_maintenance/
│
├── app.py                  # Main Streamlit app
├── model/
│   └── jet_engine_model.pkl  # Trained ML model
├── data/
│   ├── train.csv           # Training dataset
│   ├── test.csv            # Test dataset
│   └── sample_input.csv    # Sample input for prediction
├── requirements.txt        # Required Python packages
├── utils.py                # Helper functions (data preprocessing, prediction)
└── README.md               # Project documentation

Prerequisites

Before running the project, ensure you have the following installed:

Python 3.10+

pip (Python package manager)

Virtual environment (optional but recommended)

Installation

Clone the repository

git clone <your-repo-link>
cd jet_engine_predictive_maintenance


Create a virtual environment (optional but recommended)

python -m venv venv


Activate the virtual environment

Windows

venv\Scripts\activate


Linux / Mac

source venv/bin/activate


Install dependencies

pip install -r requirements.txt

Running the Project

The project uses Streamlit to provide an interactive web interface.

Run the Streamlit app

streamlit run app.py


Open the browser
Streamlit will automatically open your default browser. If not, go to the link shown in the terminal (usually http://localhost:8501).

Using the app

Upload engine data in CSV format (e.g., sample_input.csv).

Click Predict to see the Remaining Useful Life of the engines.

Visualizations and prediction outputs will be displayed interactively.

File Details

app.py: Entry point for the Streamlit web application.

model/jet_engine_model.pkl: Pre-trained machine learning model for prediction.

data/: Contains sample and training datasets.

utils.py: Contains functions for preprocessing input data, scaling, and generating predictions.

requirements.txt: Lists all Python libraries required for running the project.

Dependencies

Key Python libraries used in this project:

streamlit – For interactive UI

pandas – For data handling

numpy – For numerical computations

scikit-learn – For ML modeling and scaling

matplotlib / seaborn – For visualizations

joblib – For loading/saving ML models

Install all dependencies using:

pip install -r requirements.txt

How It Works

Load engine sensor and operational data.

Preprocess the data (scaling, cleaning missing values, feature selection).

Pass the data to the pre-trained ML model.

Predict the Remaining Useful Life (RUL) for each engine.

Visualize results with Streamlit.

Sample Usage
import pandas as pd
from utils import preprocess_data, predict_rul
import joblib

# Load sample input
data = pd.read_csv('data/sample_input.csv')

# Preprocess
processed_data = preprocess_data(data)

# Load model
model = joblib.load('model/jet_engine_model.pkl')

# Predict RUL
rul_predictions = predict_rul(model, processed_data)
print(rul_predictions)

Future Enhancements

Integrate real-time engine sensor data streaming.

Improve model accuracy using deep learning techniques.

Add maintenance scheduling recommendations based on predictions.

Deploy on cloud platforms like AWS or GCP for broader accessibility.
