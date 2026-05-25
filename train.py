"""
Training script for credit risk scoring models.

Trains Logistic Regression, Random Forest, and XGBoost on the German Credit
dataset. Logs all runs to MLflow, selects best model by cost-weighted score,
and saves artefacts for the FastAPI scoring API.

Run: python train.py
"""

import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — no display required when saving to file
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import mlflow
import mlflow.sklearn
import pickle

from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from xgboost import XGBClassifier

from config import (
    MODEL_DIR, PLOTS_DIR, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
    TARGET_COLUMN, POSITIVE_CLASS, CLASS_WEIGHT, RANDOM_STATE, TEST_SIZE,
)

MODEL_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


# ── Data loading ──────────────────────────────────────────────────────────────

print('Loading German Credit dataset...')
# parser='pandas' uses pandas to parse the ARFF file instead of the default
# liac-arff parser — significantly faster on sklearn 1.3+.
# The dataset is cached in ~/scikit_learn_data/ after the first download,
# so subsequent runs are instant regardless of network speed.
data = fetch_openml('credit-g', version=1, as_frame=True, parser='pandas')
df = data.frame

print(f'Dataset shape: {df.shape}')
print(f'\nTarget column: {TARGET_COLUMN}')
class_counts = df[TARGET_COLUMN].value_counts()
print(class_counts)
imbalance_ratio = class_counts['bad'] / class_counts['good']
print(f'Imbalance ratio (bad/good): {imbalance_ratio:.2f}')

# Encode target: 'bad' → 1, 'good' → 0. Binary int is required by most
# sklearn metrics and lets us use .sum() to count positives.
df['target'] = (df[TARGET_COLUMN] == POSITIVE_CLASS).astype(int)
df = df.drop(columns=[TARGET_COLUMN])

# Detect feature types from pandas dtypes — avoids hardcoding column names
# that could change across dataset versions.
numeric_features = df.select_dtypes(include=['number']).columns.tolist()
numeric_features = [f for f in numeric_features if f != 'target']  # exclude label
categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

print(f'\nNumeric features ({len(numeric_features)}): {numeric_features}')
print(f'Categorical features ({len(categorical_features)}): {categorical_features}')


# ── EDA plots ─────────────────────────────────────────────────────────────────

print('\nGenerating EDA plots...')

# 01 — class distribution
fig, ax = plt.subplots(figsize=(7, 5))
labels = ['Good (0)', 'Bad (1)']
values = [class_counts['good'], class_counts['bad']]
colours = ['#27AE60', '#C0392B']
bars = ax.bar(labels, values, color=colours, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, values):
    pct = val / sum(values) * 100
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
            f'{val} ({pct:.0f}%)', ha='center', va='bottom', fontweight='bold')
ax.set_title('Class Distribution — German Credit Dataset', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Count')
ax.set_ylim(0, max(values) * 1.15)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(PLOTS_DIR / '01_class_distribution.png', dpi=150)
plt.close()

# 02–04 — KDE plots for key numeric features
for i, (feature, title) in enumerate([
    ('credit_amount', 'Credit Amount by Risk Class'),
    ('duration', 'Loan Duration by Risk Class'),
    ('age', 'Applicant Age by Risk Class'),
], start=2):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, colour, val in [('Good (low risk)', '#27AE60', 0), ('Bad (high risk)', '#C0392B', 1)]:
        subset = df[df['target'] == val][feature]
        subset.plot.kde(ax=ax, label=label, color=colour, linewidth=2)
        ax.axvline(subset.median(), color=colour, linestyle='--', alpha=0.6,
                   label=f'Median {label.split()[0]}: {subset.median():.0f}')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel(feature.replace('_', ' ').title())
    ax.set_ylabel('Density')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'0{i}_{feature}_by_risk.png', dpi=150)
    plt.close()

# 05 — correlation heatmap
fig, ax = plt.subplots(figsize=(9, 7))
corr = df[numeric_features + ['target']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))  # hide upper triangle — avoids duplicate pairs
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn_r',
            center=0, square=True, linewidths=0.5, ax=ax,
            cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(PLOTS_DIR / '05_feature_correlation.png', dpi=150)
plt.close()

# 06 — categorical features default rate
cat_features_to_plot = [
    'checking_status', 'credit_history', 'purpose',
    'savings_status', 'employment', 'personal_status',
]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, feature in zip(axes.flatten(), cat_features_to_plot):
    default_rate = df.groupby(feature)['target'].mean().sort_values(ascending=False)
    colours = ['#C0392B' if r > 0.4 else '#F39C12' if r > 0.25 else '#27AE60'
               for r in default_rate.values]
    default_rate.plot.bar(ax=ax, color=colours, edgecolor='white')
    ax.set_title(feature.replace('_', ' ').title(), fontweight='bold')
    ax.set_ylabel('Default Rate')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right', fontsize=8)
    ax.set_ylim(0, 1)
    ax.axhline(0.3, color='grey', linestyle='--', alpha=0.5, linewidth=1)
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Default Rate by Categorical Feature', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(PLOTS_DIR / '06_categorical_features.png', dpi=150, bbox_inches='tight')
plt.close()

print('EDA plots saved to plots/')


# ── Preprocessing ─────────────────────────────────────────────────────────────

# StandardScaler normalises numeric features — required for Logistic Regression
# (gradient descent converges faster) and has no negative effect on tree models.
# OneHotEncoder drop='first' removes one column per feature to avoid the dummy
# variable trap (perfect multicollinearity with an intercept term).
# sparse_output=False returns a dense array compatible with SHAP.
# handle_unknown='ignore' silently zero-encodes categories not seen in training
# rather than raising an error — important for API input that may differ from
# the training set.
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
     categorical_features),
])

X = df.drop(columns=['target'])
y = df['target']

# stratify=y preserves the 70/30 class ratio in both splits — without this,
# random chance could give the test set a very different imbalance than training.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
)

print(f'\nTrain set: {X_train.shape[0]} samples')
print(f'Train class distribution: {y_train.value_counts().to_dict()}')
print(f'Test set: {X_test.shape[0]} samples')
print(f'Test class distribution: {y_test.value_counts().to_dict()}')


# ── Model training ────────────────────────────────────────────────────────────

def cost_weighted_score_metric(y_true, y_pred):
    """
    Business metric: weighted error where FN costs 5x FP.

    A model that maximises ROC-AUC may not minimise business cost.
    The cost-weighted score directly optimises for the asymmetric error
    costs in credit lending.
    """
    cm = confusion_matrix(y_true, y_pred)
    # confusion_matrix layout: cm[actual][predicted]
    # cm[1][0] = actual bad, predicted good → false negative (approved bad loan)
    # cm[0][1] = actual good, predicted bad → false positive (rejected good loan)
    fn = cm[1][0]
    fp = cm[0][1]
    total_bad = y_true.sum()
    total_good = len(y_true) - total_bad
    # Formula: 1 - (weighted error) / (worst-case weighted error)
    # Score of 1.0 = perfect; score of 0.0 = every bad loan approved.
    return 1 - (5 * fn + fp) / (5 * total_bad + total_good)


def evaluate_model(pipeline, X_test, y_test, model_name):
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'cost_weighted_score': cost_weighted_score_metric(y_test, y_pred),
    }
    print(f'\n{model_name} results:')
    for k, v in metrics.items():
        print(f'  {k}: {v:.4f}')
    return metrics, cm


models_config = [
    ('LogisticRegression', LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,         # default 100 often fails to converge on this dataset
        class_weight=CLASS_WEIGHT)),
    ('RandomForest', RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, class_weight=CLASS_WEIGHT)),
    ('XGBoost', XGBClassifier(
        n_estimators=100, random_state=RANDOM_STATE,
        scale_pos_weight=5,    # XGBoost equivalent of class_weight={0:1, 1:5}
        eval_metric='logloss', # suppresses a deprecation warning in XGBoost 2.x
        verbosity=0)),         # silences XGBoost's per-tree training output
]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

all_metrics = {}
all_pipelines = {}
mlflow_run_ids = {}

print('\nTraining models with MLflow tracking...')

for model_name, model in models_config:
    print(f'\n--- {model_name} ---')
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model),
    ])

    with mlflow.start_run(run_name=model_name) as run:
        mlflow_run_ids[model_name] = run.info.run_id

        params = {'model': model_name}
        if hasattr(model, 'get_params'):
            # Filter out dict-valued and None params — MLflow log_params
            # only accepts string-serialisable scalar values.
            params.update({k: v for k, v in model.get_params().items()
                           if not isinstance(v, dict) and v is not None})
        mlflow.log_params(params)

        pipeline.fit(X_train, y_train)
        metrics, cm = evaluate_model(pipeline, X_test, y_test, model_name)

        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        mlflow.sklearn.log_model(pipeline, 'model')

        for plot_file in PLOTS_DIR.glob('*.png'):
            mlflow.log_artifact(str(plot_file))

    all_metrics[model_name] = metrics
    all_pipelines[model_name] = pipeline


# ── Model selection ───────────────────────────────────────────────────────────

print('\n\n=== Model Comparison ===')
header = f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>6} {'ROC-AUC':>9} {'Cost-Weighted':>14}"
print(header)
print('-' * len(header))
for name, m in all_metrics.items():
    print(f"{name:<22} {m['accuracy']:>9.4f} {m['precision']:>10.4f} "
          f"{m['recall']:>8.4f} {m['f1']:>6.4f} {m['roc_auc']:>9.4f} "
          f"{m['cost_weighted_score']:>14.4f}")

best_model_name = max(all_metrics, key=lambda n: all_metrics[n]['cost_weighted_score'])
best_pipeline = all_pipelines[best_model_name]
best_metrics = all_metrics[best_model_name]

print(f'\nWinner: {best_model_name}')
print(f'  Cost-weighted score: {best_metrics["cost_weighted_score"]:.4f}')
print(f'  Selected by cost_weighted_score — the primary business metric.')
print(f'  (A bad loan approved costs 5x a good loan rejected.)')


# ── Save best model ───────────────────────────────────────────────────────────

with open(MODEL_DIR / 'best_model.pkl', 'wb') as f:
    pickle.dump(best_pipeline, f)

(MODEL_DIR / 'best_model_name.txt').write_text(best_model_name)

model_metrics_payload = {
    name: metrics for name, metrics in all_metrics.items()
}
with open(MODEL_DIR / 'model_metrics.json', 'w') as f:
    json.dump(model_metrics_payload, f, indent=2)

mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
with mlflow.start_run(run_name=f'{best_model_name}_best_registered'):
    mlflow.sklearn.log_model(
        best_pipeline,
        'best_model',
        registered_model_name='credit-risk-best',
    )

print(f'\nModel saved to {MODEL_DIR}/best_model.pkl')


# ── SHAP analysis ─────────────────────────────────────────────────────────────

print('\nRunning SHAP analysis...')

# Transform X_test manually — we need the transformed array to pass to SHAP,
# which expects the preprocessed feature space, not raw strings.
X_test_transformed = best_pipeline.named_steps['preprocessor'].transform(X_test)
classifier = best_pipeline.named_steps['classifier']

# Reconstruct the full feature name list after one-hot encoding.
# get_feature_names_out() returns names like 'checking_status_no checking'
# that match the keys in FEATURE_LABELS and DIRECTION_TEMPLATES.
ohe = best_pipeline.named_steps['preprocessor'].named_transformers_['cat']
cat_feature_names = ohe.get_feature_names_out(categorical_features).tolist()
all_feature_names = numeric_features + cat_feature_names

# LinearExplainer is faster and exact for linear models (Logistic Regression).
# shap.Explainer auto-selects TreeExplainer for XGBoost and Random Forest,
# which is also exact. Using the wrong explainer type raises a TypeError.
if best_model_name == 'LogisticRegression':
    explainer = shap.LinearExplainer(classifier, X_test_transformed)
else:
    explainer = shap.Explainer(classifier, X_test_transformed)

shap_values = explainer(X_test_transformed)

# shap.Explainer returns an Explanation object (shap_values.values) while
# shap.LinearExplainer may return a raw array — handle both.
shap_vals_array = shap_values.values if hasattr(shap_values, 'values') else shap_values

# 07 — SHAP summary beeswarm
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_vals_array, X_test_transformed,
                  feature_names=all_feature_names, show=False, max_display=15)
plt.title(f'SHAP Feature Importance — {best_model_name}', fontsize=14,
          fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(PLOTS_DIR / '07_shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()

# Identify highest and lowest risk applicants
y_prob_test = best_pipeline.predict_proba(X_test)[:, 1]
highest_risk_idx = np.argmax(y_prob_test)
lowest_risk_idx = np.argmin(y_prob_test)

# 08 — waterfall for highest-risk applicant
plt.figure(figsize=(10, 6))
shap.waterfall_plot(shap_values[highest_risk_idx], max_display=10, show=False)
plt.title(f'SHAP Waterfall — Highest Risk Applicant (p={y_prob_test[highest_risk_idx]:.2f})',
          fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / '08_shap_waterfall_highrisk.png', dpi=150, bbox_inches='tight')
plt.close()

# 09 — waterfall for lowest-risk applicant
plt.figure(figsize=(10, 6))
shap.waterfall_plot(shap_values[lowest_risk_idx], max_display=10, show=False)
plt.title(f'SHAP Waterfall — Lowest Risk Applicant (p={y_prob_test[lowest_risk_idx]:.2f})',
          fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / '09_shap_waterfall_lowrisk.png', dpi=150, bbox_inches='tight')
plt.close()

# 10 — model comparison bar chart
metric_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'cost_weighted_score']
x = np.arange(len(metric_names))
width = 0.25
fig, ax = plt.subplots(figsize=(14, 6))
colours = ['#3498DB', '#E67E22', '#2ECC71']
for i, (name, m) in enumerate(all_metrics.items()):
    values = [m[k] for k in metric_names]
    bars = ax.bar(x + i * width, values, width, label=name, color=colours[i],
                  edgecolor='white', linewidth=0.8)
ax.set_xticks(x + width)
ax.set_xticklabels([m.replace('_', '\n') for m in metric_names], fontsize=9)
ax.set_ylim(0, 1.15)
ax.set_ylabel('Score')
ax.set_title('Model Comparison — All Metrics', fontsize=14, fontweight='bold', pad=15)
ax.legend()
ax.spines[['top', 'right']].set_visible(False)
ax.axhline(1.0, color='grey', linestyle='--', alpha=0.3, linewidth=1)
plt.tight_layout()
plt.savefig(PLOTS_DIR / '10_model_comparison.png', dpi=150)
plt.close()

# SHAP feature importance — top 10.
# Mean absolute SHAP value across all test samples is the standard global
# importance metric: it tells us which features move predictions the most
# on average, regardless of direction.
mean_abs_shap = np.abs(shap_vals_array).mean(axis=0)
feature_importance = sorted(
    zip(all_feature_names, mean_abs_shap.tolist()),
    key=lambda x: x[1], reverse=True,
)[:10]

shap_importance_payload = [
    {'feature': name, 'importance': round(val, 4)} for name, val in feature_importance
]
with open(MODEL_DIR / 'shap_feature_importance.json', 'w') as f:
    json.dump(shap_importance_payload, f, indent=2)

print('\nTop 5 SHAP features:')
for feat, imp in feature_importance[:5]:
    print(f'  {feat}: {imp:.4f}')


# ── Final summary ─────────────────────────────────────────────────────────────

print('\n\n=== Final Summary ===')
print(f'Best model: {best_model_name}')
print(f'Cost-weighted score: {best_metrics["cost_weighted_score"]:.4f}')
print(f'ROC-AUC: {best_metrics["roc_auc"]:.4f}')
print('\nMLflow run IDs:')
for name, run_id in mlflow_run_ids.items():
    print(f'  {name}: {run_id}')
print('\nModel files:')
print(f'  {MODEL_DIR}/best_model.pkl')
print(f'  {MODEL_DIR}/best_model_name.txt')
print(f'  {MODEL_DIR}/model_metrics.json')
print(f'  {MODEL_DIR}/shap_feature_importance.json')
print(f'\nPlots saved to {PLOTS_DIR}/')
print('\nDone.')
