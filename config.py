from pathlib import Path

MODEL_DIR = Path('models')
PLOTS_DIR = Path('plots')
MLFLOW_DIR = Path('mlruns')
RANDOM_STATE = 42
TEST_SIZE = 0.2

# MLflow
MLFLOW_TRACKING_URI = 'http://localhost:5000'
MLFLOW_EXPERIMENT_NAME = 'credit-risk-scoring'

# Model
TARGET_COLUMN = 'class'
POSITIVE_CLASS = 'bad'  # 'bad' credit = default risk = what we predict
# scikit-learn's fetch_openml returns 'good'/'bad' as strings.
# We encode 'bad' as 1 (the positive class — the risk we care about)
# and 'good' as 0.

# Business cost of errors — standard credit risk assumption
# A bad loan approved (false negative) costs 5x a good loan rejected (false positive)
CLASS_WEIGHT = {0: 1, 1: 5}
# passed to sklearn models as class_weight parameter.
# XGBoost uses scale_pos_weight = 5 instead.

# Risk bands — map probability of default to business label
RISK_BANDS = {
    (0.0, 0.2):  {'band': 'Low',       'label': 'Approve',           'colour': '#27AE60'},
    (0.2, 0.4):  {'band': 'Medium',    'label': 'Review',            'colour': '#F39C12'},
    (0.4, 0.6):  {'band': 'High',      'label': 'Additional checks', 'colour': '#E67E22'},
    (0.6, 1.01): {'band': 'Very High', 'label': 'Decline',           'colour': '#C0392B'},
}

# Risk score: map probability to 0-100 scale (higher = more risk)
# score = int(probability * 100)

# Azure
AZURE_APP_NAME = 'credit-risk-scorer'
AZURE_RESOURCE_GROUP = 'credit-risk-rg'
AZURE_LOCATION = 'westeurope'
