# Jet Engine Predictive Maintenance

## Overview
This project focuses on **predictive maintenance of jet engines** using machine learning. The goal is to **monitor engine health**, predict potential failures, and reduce unexpected downtime by providing actionable insights for maintenance planning.

The system analyzes sensor data from engines, processes it, and predicts the **Remaining Useful Life (RUL)** of critical components, helping improve safety, reduce costs, and optimize operational efficiency.

---

## Features
- **Predictive Modeling:** Machine learning models forecast engine health and potential failures.  
- **Data Preprocessing:** Handles sensor data scaling, cleaning, and feature engineering.  
- **Visualization:** Generates intuitive plots to track engine health over time.  
- **Streamlit Dashboard:** Interactive dashboard to visualize predictions in real-time.  
- **Alerts & Notifications:** Highlights engines requiring immediate attention.  

---

## Dataset
The project uses engine sensor data, containing features like:

- Sensor readings (temperature, pressure, vibration, etc.)  
- Operating conditions  
- Remaining Useful Life (RUL) labels  

**Publicly available dataset for practice:**  
- [NASA C-MAPSS Dataset](https://ti.arc.nasa.gov/tech/dash/pcoe/prognostic-data-repository/)

---

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/jet-engine-predictive-maintenance.git
cd jet-engine-predictive-maintenance
Create and activate a virtual environment

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```
2.Install dependencies

```bash
pip install -r requirements.txt
#How to Run
Run the Streamlit dashboard

python streamlit run ini.py
```
Predict Engine Health

Upload engine sensor data via the dashboard

View RUL predictions and visualizations

Project Structure
jet-engine-predictive-maintenance/
│
├─ ini.py                # Streamlit dashboard
├─ ini.cpython-313.py    
├─ train_model.py        #dataset
├─train_FD001.txt 
├─ scaler.pkl               # Saved ML models (e.g., .pkl files)
├─ rul_model.pkl                 # Sample datasets
├─ requirements.txt      # Python dependencie
Technologies Used
Python

Pandas & NumPy (Data processing)

Matplotlib & Seaborn (Visualization)

Streamlit (Dashboard)

