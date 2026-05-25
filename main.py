"""
FastAPI credit risk scoring API.

Loads a trained sklearn pipeline (preprocessor + classifier) and serves
credit risk predictions with SHAP-based plain-English explanations.

Run locally: uvicorn main:app --reload
Docker:      docker-compose up
"""

import json
import pickle
import numpy as np
import pandas as pd
import shap

from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel, Field
from typing import Optional

from config import MODEL_DIR, RISK_BANDS, CLASS_WEIGHT
from risk_explainer import shap_values_to_risk_factors

# ── Global state ──────────────────────────────────────────────────────────────

# Module-level globals so the model and SHAP explainer are loaded once at
# startup and reused across all requests — not reloaded per request.
MODEL_LOADED = False
pipeline = None
model_name = 'unknown'
model_metrics = {}
shap_feature_importance = []
shap_explainer = None
feature_names: list[str] = []

# The full feature list expected by the preprocessor, in the same order
# used during training. The API only exposes a subset to callers; the rest
# are filled with neutral defaults in score_application().
numeric_features = [
    'duration', 'credit_amount', 'installment_commitment',
    'age', 'existing_credits', 'residence_since', 'num_dependents',
]
categorical_features = [
    'checking_status', 'credit_history', 'purpose', 'savings_status',
    'employment', 'personal_status', 'other_parties',
    'property_magnitude', 'other_payment_plans', 'housing', 'job',
    'own_telephone', 'foreign_worker',
]


# lifespan replaces the deprecated @app.on_event("startup") pattern.
# Code before `yield` runs at startup; code after runs at shutdown.
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL_LOADED, pipeline, model_name, model_metrics
    global shap_feature_importance, shap_explainer, feature_names

    try:
        with open(MODEL_DIR / 'best_model.pkl', 'rb') as f:
            pipeline = pickle.load(f)
        model_name = (MODEL_DIR / 'best_model_name.txt').read_text().strip()
        with open(MODEL_DIR / 'model_metrics.json') as f:
            model_metrics = json.load(f)
        with open(MODEL_DIR / 'shap_feature_importance.json') as f:
            shap_feature_importance = json.load(f)

        # Recover the post-encoding feature names from the fitted OHE —
        # these are the column names SHAP sees (e.g. 'checking_status_no checking').
        ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat']
        cat_feature_names = ohe.get_feature_names_out(categorical_features).tolist()
        feature_names = numeric_features + cat_feature_names

        classifier = pipeline.named_steps['classifier']
        # SHAP explainers need a background dataset to estimate feature contributions.
        # We pass one representative sample so the explainer is built once at startup
        # rather than rebuilt per request (which would be very slow).
        dummy_input = pipeline.named_steps['preprocessor'].transform(
            pd.DataFrame([{
                'duration': 12, 'credit_amount': 2000,
                'installment_commitment': 2, 'age': 35,
                'existing_credits': 1, 'residence_since': 3,
                'num_dependents': 1, 'checking_status': '>=200',
                'credit_history': 'existing paid', 'purpose': 'radio/tv',
                'savings_status': '>=1000', 'employment': '>=7',
                'personal_status': 'male single', 'other_parties': 'none',
                'property_magnitude': 'real estate',
                'other_payment_plans': 'none', 'housing': 'own',
                'job': 'skilled', 'own_telephone': 'yes',
                'foreign_worker': 'yes',
            }])
        )

        # Use the exact same explainer logic as train.py so SHAP values
        # are computed with the appropriate algorithm for each model type.
        if model_name == 'LogisticRegression':
            shap_explainer = shap.LinearExplainer(classifier, dummy_input)
        else:
            shap_explainer = shap.Explainer(classifier, dummy_input)

        MODEL_LOADED = True
        print(f'Model loaded: {model_name}')
        print(f'Docs:         http://127.0.0.1:8000/docs')
        print(f'Demo UI:      http://127.0.0.1:8000/demo')
    except Exception as e:
        # API still starts with MODEL_LOADED=False — health endpoint stays up
        # so load balancers can detect the degraded state without a crash.
        print(f'Model loading failed: {e}')
        MODEL_LOADED = False

    yield


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title='Credit Risk Scoring API',
    description=(
        'Predicts probability of loan default using Logistic Regression trained on the '
        'German Credit dataset. Risk scores 0-100, risk bands '
        'Low/Medium/High/Very High.\n\n'
        '**[Try the demo UI →](/demo)**'
    ),
    version='1.0.0',
    docs_url='/docs',
    redoc_url='/redoc',
    lifespan=lifespan,
)

# Permissive CORS so the demo frontend (served at /demo) can call /score
# from a browser, and so third-party clients can explore the API from /docs.
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

templates = Jinja2Templates(directory='templates')


# ── Pydantic models ───────────────────────────────────────────────────────────

class CreditApplication(BaseModel):
    """Input features for credit risk scoring."""
    duration: int = Field(..., description='Loan duration in months', ge=1, le=120)
    credit_amount: float = Field(..., description='Loan amount in DM', ge=0)
    installment_commitment: int = Field(..., description='Installment rate as % of income', ge=1, le=4)
    age: int = Field(..., description='Applicant age in years', ge=18, le=100)
    existing_credits: int = Field(..., description='Number of existing credits at this bank', ge=0, le=4)
    checking_status: str = Field(..., description="Checking account status: 'no checking', '<0', '0<=X<200', '>=200'")
    credit_history: str = Field(..., description="Credit history: 'no credits/all paid', 'all paid', 'existing paid', 'delayed previously', 'critical/other existing credit'")
    purpose: str = Field(..., description="Loan purpose: 'new car', 'used car', 'furniture/equipment', 'radio/tv', 'domestic appliance', 'repairs', 'education', 'retraining', 'business', 'other'")
    savings_status: str = Field(..., description="Savings account status: 'no known savings', '<100', '100<=X<500', '500<=X<1000', '>=1000'")
    employment: str = Field(..., description="Employment duration: 'unemployed', '<1', '1<=X<4', '4<=X<7', '>=7'")


class RiskScore(BaseModel):
    """Credit risk assessment output."""
    probability_of_default: float
    risk_score: int
    risk_band: str
    decision: str
    risk_factors: list[str]
    protective_factors: list[str]
    model_used: str
    shap_values_raw: dict


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_risk_band(probability: float) -> dict:
    for (low, high), info in RISK_BANDS.items():
        if low <= probability < high:
            return info
    return {'band': 'Very High', 'label': 'Decline', 'colour': '#C0392B'}


def score_application(application: CreditApplication) -> RiskScore:
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail='Model not loaded')

    # The API exposes 10 fields. The preprocessor trained on 20 features,
    # so the remaining 10 are filled with neutral/modal values from the
    # training set. These fields have low SHAP importance and don't
    # meaningfully affect the score, but the pipeline requires them.
    input_data = {
        'duration': application.duration,
        'credit_amount': application.credit_amount,
        'installment_commitment': application.installment_commitment,
        'age': application.age,
        'existing_credits': application.existing_credits,
        'checking_status': application.checking_status,
        'credit_history': application.credit_history,
        'purpose': application.purpose,
        'savings_status': application.savings_status,
        'employment': application.employment,
        # Neutral defaults for fields not collected from the caller:
        'residence_since': 3,
        'num_dependents': 1,
        'personal_status': 'male single',
        'other_parties': 'none',
        'property_magnitude': 'real estate',
        'other_payment_plans': 'none',
        'housing': 'own',
        'job': 'skilled',
        'own_telephone': 'yes',
        'foreign_worker': 'yes',
    }

    df_input = pd.DataFrame([input_data])
    X_transformed = pipeline.named_steps['preprocessor'].transform(df_input)

    # predict_proba returns [[p_good, p_bad]] — index 1 is probability of the
    # positive class (bad credit = default), which is what we report.
    probability = float(pipeline.predict_proba(df_input)[0][1])
    risk_score = int(probability * 100)
    band_info = get_risk_band(probability)

    shap_vals_obj = shap_explainer(X_transformed)
    # Normalise output: Explanation object vs raw array (see train.py note).
    shap_vals = shap_vals_obj.values[0] if hasattr(shap_vals_obj, 'values') else shap_vals_obj[0]

    risk_factors, protective_factors = shap_values_to_risk_factors(
        shap_vals.tolist(), feature_names, top_n=3,
    )

    # Include the raw SHAP dict in the response so API consumers can build
    # their own explanations or audit the model's reasoning per prediction.
    shap_raw = {name: round(float(val), 4)
                for name, val in zip(feature_names, shap_vals)}

    return RiskScore(
        probability_of_default=round(probability, 4),
        risk_score=risk_score,
        risk_band=band_info['band'],
        decision=band_info['label'],
        risk_factors=risk_factors,
        protective_factors=protective_factors,
        model_used=model_name,
        shap_values_raw=shap_raw,
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get('/')
def root():
    return {
        'message': 'Credit Risk Scoring API',
        'docs': '/docs',
        'health': '/health',
    }


@app.get('/health', summary='Health check', tags=['System'])
def health():
    """
    Check whether the API is running and the model is loaded.

    Returns `model_loaded: true` when the scoring model is ready to accept
    requests. If `model_loaded` is false, run `python train.py` to generate
    the model files, then restart the server.
    """
    return {
        'status': 'ok',
        'model_loaded': MODEL_LOADED,
        'model_name': model_name,
    }


@app.get('/demo', response_class=HTMLResponse, summary='Interactive demo UI', tags=['System'])
def demo(request: Request):
    """
    Open a simple web form where you can fill in a loan application and see
    the risk score instantly — no coding required. Good for a quick demo or
    to get a feel for how the model behaves across different applicant profiles.
    """
    return templates.TemplateResponse('index.html', {'request': request})


@app.post('/score', response_model=RiskScore, summary='Score a single application', tags=['Scoring'])
def score(application: CreditApplication):
    """
    Submit one loan application and get back an instant risk assessment.

    The response tells you:
    - **probability_of_default** — the model's estimated chance the applicant won't repay (0 to 1)
    - **risk_score** — that probability mapped to 0–100 (higher = riskier)
    - **risk_band** — Low / Medium / High / Very High
    - **decision** — Approve / Review / Additional checks / Decline
    - **risk_factors** — the top reasons the model flagged this application as risky
    - **protective_factors** — the top reasons working in the applicant's favour
    - **shap_values_raw** — the raw numbers behind the explanation (for technical users)

    Example — a high-risk applicant:
    ```json
    {
      "duration": 48,
      "credit_amount": 15000,
      "installment_commitment": 4,
      "age": 22,
      "existing_credits": 3,
      "checking_status": "no checking",
      "credit_history": "delayed previously",
      "purpose": "new car",
      "savings_status": "no known savings",
      "employment": "<1"
    }
    ```
    """
    return score_application(application)


@app.post('/score/batch', response_model=list[RiskScore], summary='Score up to 100 applications at once', tags=['Scoring'])
def score_batch(applications: list[CreditApplication]):
    """
    Submit a list of loan applications (up to 100) and get back a risk
    assessment for each one in a single API call.

    The response is a list in the same order as the input — the first result
    corresponds to the first application, and so on. Useful when a bank needs
    to process a queue of applications in bulk rather than one at a time.
    """
    if len(applications) > 100:
        raise HTTPException(status_code=400, detail='Maximum 100 applications per batch')
    return [score_application(app) for app in applications]


@app.get('/api/model-info', summary='Model training results', tags=['Model'])
def model_info():
    """
    Returns the performance metrics for all three models trained during the
    last run of `train.py` (Logistic Regression, Random Forest, XGBoost).

    The primary selection metric is `cost_weighted_score`, not accuracy —
    because in credit lending, approving a bad loan costs roughly 5× more
    than rejecting a good one. The model with the highest cost-weighted score
    is the one currently serving predictions.
    """
    return {
        'primary_metric': 'cost_weighted_score',
        'models': model_metrics,
    }


@app.get('/api/feature-importance', summary='Top 10 factors driving predictions', tags=['Model'])
def feature_importance():
    """
    Returns the 10 features that have the biggest influence on the model's
    predictions, ranked by their average SHAP impact across all test applicants.

    SHAP (SHapley Additive exPlanations) measures how much each feature
    pushes a prediction up or down compared to the average. A high value here
    means the feature frequently shifts the risk score by a large amount —
    regardless of whether it increases or reduces risk.
    """
    return shap_feature_importance


@app.get('/api/risk-bands', summary='Risk band thresholds and decisions', tags=['Model'])
def risk_bands():
    """
    Returns the four risk bands and the business decision associated with each.

    The risk score (0–100) is divided into bands based on the probability of
    default. Each band maps to a recommended action:
    - **Low (0–20)** → Approve
    - **Medium (20–40)** → Review
    - **High (40–60)** → Additional checks
    - **Very High (60–100)** → Decline

    These thresholds reflect standard credit risk practice and can be adjusted
    to suit a lender's risk appetite.
    """
    return [
        {
            'min': low,
            'max': high,
            'band': info['band'],
            'label': info['label'],
            'colour': info['colour'],
        }
        for (low, high), info in RISK_BANDS.items()
    ]
