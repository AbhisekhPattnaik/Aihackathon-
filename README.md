# ✈️ Jet Engine Predictive Maintenance Dashboard

A machine learning-powered predictive maintenance system that monitors jet engine health and predicts remaining useful life (RUL) to enable preventive maintenance decisions.

## 📋 Overview

This project uses **Real-time sensor data** from jet engines to build a predictive model that forecasts when an engine is likely to fail. By predicting failures before they occur, maintenance teams can schedule repairs proactively, reducing downtime and operational costs.

## 🎯 Key Features

- **RUL Prediction**: Predicts Remaining Useful Life in cycles for any jet engine
- **Health Monitoring**: Real-time engine health percentage (0-100%)
- **Status Alerts**: Three-tier status system (Healthy → Warning → Critical)
- **Interactive Dashboard**: Built with Streamlit for easy visualization
- **Sensor Analytics**: Visualize degradation patterns of critical sensors
- **Feature Importance**: Understand which sensors matter most for RUL prediction
- **Multi-Engine Support**: Monitor and compare multiple engines from a dataset

## 🛠️ Technologies Used

- **Python 3.x**
- **Streamlit**: Interactive web dashboard
- **Scikit-learn**: Machine learning (Random Forest Regressor)
- **Pandas**: Data processing and manipulation
- **NumPy**: Numerical computing
- **Plotly**: Interactive data visualizations
- **Joblib**: Model serialization

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup Instructions

1. **Clone or navigate to the project directory**:
   ```bash
   cd "c:\Users\nitro\Desktop\iitkgp aihack"
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**:
   - **Windows (PowerShell)**:
     ```bash
     .\.venv\Scripts\Activate.ps1
     ```
   - **Windows (CMD)**:
     ```bash
     .venv\Scripts\activate.bat
     ```
   - **macOS/Linux**:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirement.txt
   ```

## 🚀 Usage

### Running the Dashboard

```bash
streamlit run ini.py
```

The dashboard will open in your default browser at `http://localhost:8501`.

### Using the Dashboard

1. **Select an Engine**: Choose an engine ID from the dropdown to monitor its data
2. **View Sensor Trends**: Select individual sensors to see their degradation over operational cycles
3. **Check Health Status**: 
   - **Healthy** (> 70% health): Engine operating normally
   - **Warning** (40-70% health): Schedule maintenance soon
   - **Critical** (< 40% health): Immediate maintenance required
4. **Analyze Key Sensors**: View degradation patterns for critical sensors (2, 4, 7)
5. **Explore Model Insights**: Check feature importance to understand which sensors influence RUL predictions

## 📊 Dataset

**File**: `train_FD001.txt`

The dataset contains operational data from jet engines with:
- **Engine ID**: Unique identifier for each engine
- **Cycle**: Operational cycle number
- **Operating Settings**: 3 operational parameters (settings 1-3)
- **Sensor Readings**: 21 different sensor measurements
- **Target Variable**: Remaining Useful Life (RUL) calculated from the data

The RUL is computed as the difference between the maximum cycle for each engine and its current cycle, capped at 125 cycles for model stability.

## 🤖 Machine Learning Model

### Model Architecture
- **Algorithm**: Random Forest Regressor
- **Number of Trees**: 200
- **Max Depth**: 15
- **Min Samples per Leaf**: 5
- **Features**: Selected sensor readings (filtered to remove near-constant sensors)

### Model Training
The model is trained on sensor readings to predict RUL. The training process includes:
1. Data loading and feature engineering
2. Feature scaling using MinMaxScaler (0-1 normalization)
3. Train-test split (80-20 split)
4. Model fitting and evaluation
5. Model and scaler persistence (saved to `.pkl` files)

### Model Files
- `rul_model.pkl`: Trained Random Forest model
- `scaler.pkl`: MinMaxScaler for feature normalization

## 📁 Project Structure

```
iitkgp aihack/
├── ini.py                    # Main Streamlit application
├── requirement.txt           # Python dependencies
├── train_FD001.txt          # Dataset (jet engine sensor data)
├── rul_model.pkl            # Trained ML model (auto-generated)
├── scaler.pkl               # Feature scaler (auto-generated)
└── README.md                # This file
```

## 🔍 How It Works

1. **Data Loading**: The system loads the training dataset and computes RUL for each engine record
2. **Model Training**: On first run, a Random Forest model is trained and saved to disk (subsequent runs load the pre-trained model)
3. **Engine Selection**: User selects an engine to monitor
4. **Prediction**: The latest sensor readings are scaled and fed to the model to predict RUL
5. **Health Calculation**: Health percentage is derived from RUL (`health = (RUL / 125) * 100`)
6. **Visualization**: Multiple interactive charts show sensor trends, health degradation, and feature importance

## 📈 Predicted Outputs

- **Predicted RUL**: Remaining useful life in operational cycles
- **Health Percentage**: Current engine health (0-100%)
- **Engine Status**: Categorical status (Healthy/Warning/Critical)
- **Sensor Trends**: Historical and degradation patterns
- **Feature Importance**: Top 15 sensors influencing RUL predictions

## ⚙️ Configuration

Key parameters can be adjusted in `ini.py`:
- `MAX_RUL = 125`: Maximum RUL value for health calculation
- `RandomForestRegressor` parameters: Adjust model complexity and performance
- Sensor selection: Modify `important_sensors` list for different analysis

## 🐛 Troubleshooting

### Model Not Loading
If you see "Model or scaler not found" on first run, the system will automatically train a new model. This may take a few moments.

### Missing Dependencies
Ensure all requirements are installed:
```bash
pip install --upgrade -r requirement.txt
```

### Port Already in Use
If port 8501 is already in use, run:
```bash
streamlit run ini.py --server.port 8502
```

## 📝 Notes

- The model uses only sensor data (not operational settings) for RUL prediction
- The system filters out near-constant sensors (std < 0.01) to improve model performance
- RUL is capped at 125 cycles to prevent extreme outliers during training
- The dashboard updates dynamically based on the selected engine and sensor configuration

## 🎓 Use Case

This system demonstrates **predictive maintenance in aviation**, where:
- **Cost Savings**: Prevent unscheduled downtime and costly emergency repairs
- **Safety**: Detect potential failures before they become critical
- **Efficiency**: Optimize maintenance scheduling and resource allocation
- **Reliability**: Improve overall fleet reliability and availability

## 📄 License

This project was created for the IIT KGP AI Hackathon.

## 👤 Contact & Support

For questions or issues, please review the code comments in `ini.py` or modify the configuration parameters as needed.

---

**Happy Monitoring! Keep those engines running smoothly!** ✈️

#working video
link:https://docs.google.com/videos/d/1DOoFNQDBcr0OijoE0cUCP-Bs9xl2KSYA7puTXH2Avyw/edit?usp=sharing
