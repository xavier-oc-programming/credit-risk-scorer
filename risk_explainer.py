"""
Translates SHAP values into plain-English credit risk factors.

This module is the primary interface between raw model explainability output
and human-readable risk factors returned by the FastAPI scoring endpoint.
"""

FEATURE_LABELS = {
    'duration': 'loan duration',
    'credit_amount': 'loan amount',
    'installment_commitment': 'installment rate as % of income',
    'age': 'applicant age',
    'existing_credits': 'number of existing credits',
    'checking_status_no checking': 'no checking account',
    'checking_status_0<=X<200': 'low checking account balance',
    'checking_status_>=200': 'high checking account balance',
    'checking_status_<0': 'overdrawn checking account',
    'credit_history_no credits/all paid': 'no prior credits or all paid',
    'credit_history_all paid': 'all credits paid on time',
    'credit_history_critical/other existing credit': 'critical credit history',
    'credit_history_delayed previously': 'previous payment delays',
    'savings_status_no known savings': 'no known savings',
    'savings_status_<100': 'savings below 100 DM',
    'savings_status_100<=X<500': 'savings between 100-500 DM',
    'savings_status_500<=X<1000': 'savings between 500-1000 DM',
    'savings_status_>=1000': 'savings above 1000 DM',
    'purpose_new car': 'new car purchase',
    'purpose_used car': 'used car purchase',
    'purpose_furniture/equipment': 'furniture/equipment purchase',
    'purpose_radio/tv': 'radio/TV purchase',
    'purpose_domestic appliance': 'domestic appliance purchase',
    'purpose_repairs': 'home repairs',
    'purpose_education': 'education loan',
    'purpose_retraining': 'retraining loan',
    'purpose_business': 'business loan',
    'purpose_other': 'other purpose',
    'employment_unemployed': 'unemployed applicant',
    'employment_<1': 'employed less than 1 year',
    'employment_1<=X<4': 'employed 1-4 years',
    'employment_4<=X<7': 'employed 4-7 years',
    'employment_>=7': 'long-term employment (7+ years)',
    'personal_status_female div/dep/mar': 'female divorced/dependent/married',
    'personal_status_male single': 'male single',
    'personal_status_male mar/wid': 'male married/widowed',
    'personal_status_female single': 'female single',
    'other_parties_guarantor': 'has guarantor',
    'other_parties_co applicant': 'has co-applicant',
    'property_magnitude_real estate': 'owns real estate',
    'property_magnitude_life insurance': 'has life insurance',
    'property_magnitude_car': 'owns a car',
    'property_magnitude_no known property': 'no known property',
    'other_payment_plans_bank': 'other bank payment plans',
    'other_payment_plans_stores': 'store payment plans',
    'housing_free': 'free housing',
    'housing_own': 'owns home',
    'job_unskilled resident': 'unskilled resident worker',
    'job_skilled': 'skilled worker',
    'job_high qualif/self emp/mgmt': 'highly qualified/self-employed/management',
    'own_telephone_yes': 'has own telephone',
    'foreign_worker_yes': 'foreign worker',
    'residence_since': 'years at current residence',
    'num_dependents': 'number of dependents',
}

DIRECTION_TEMPLATES = {
    'duration': (
        'longer loan duration increases default risk',
        'shorter loan duration reduces default risk',
    ),
    'credit_amount': (
        'higher loan amount increases default risk',
        'lower loan amount reduces default risk',
    ),
    'installment_commitment': (
        'high installment rate relative to income increases default risk',
        'low installment rate relative to income reduces default risk',
    ),
    'age': (
        'younger applicant age increases default risk',
        'older applicant age reduces default risk',
    ),
    'existing_credits': (
        'multiple existing credits increase default risk',
        'fewer existing credits reduce default risk',
    ),
    'checking_status_no checking': (
        'no checking account increases default risk',
        'having a checking account reduces default risk',
    ),
    'checking_status_0<=X<200': (
        'low checking account balance increases default risk',
        'checking account balance is not a risk factor here',
    ),
    'checking_status_>=200': (
        'checking account status increases default risk',
        'high checking account balance reduces default risk',
    ),
    'checking_status_<0': (
        'overdrawn checking account significantly increases default risk',
        'checking account is not overdrawn, reducing risk',
    ),
    'credit_history_no credits/all paid': (
        'no prior credit history increases default risk',
        'clean credit record reduces default risk',
    ),
    'credit_history_all paid': (
        'credit history increases default risk',
        'all credits paid on time reduces default risk',
    ),
    'credit_history_critical/other existing credit': (
        'critical credit history significantly increases default risk',
        'credit history is not a risk factor here',
    ),
    'credit_history_delayed previously': (
        'previous payment delays significantly increase default risk',
        'no previous payment delays reduce default risk',
    ),
    'savings_status_no known savings': (
        'no known savings significantly increase default risk',
        'having savings reduces default risk',
    ),
    'savings_status_<100': (
        'very low savings increase default risk',
        'savings level reduces default risk',
    ),
    'savings_status_100<=X<500': (
        'moderate savings are a slight risk factor',
        'moderate savings reduce default risk',
    ),
    'savings_status_500<=X<1000': (
        'savings level increases default risk',
        'substantial savings reduce default risk',
    ),
    'savings_status_>=1000': (
        'savings level increases default risk',
        'high savings significantly reduce default risk',
    ),
    'purpose_new car': (
        'new car purchase increases default risk',
        'loan purpose reduces default risk',
    ),
    'purpose_used car': (
        'loan purpose increases default risk',
        'used car purchase reduces default risk',
    ),
    'purpose_furniture/equipment': (
        'furniture/equipment purchase increases default risk',
        'loan purpose reduces default risk',
    ),
    'purpose_radio/tv': (
        'loan purpose increases default risk',
        'radio/TV purchase reduces default risk',
    ),
    'purpose_domestic appliance': (
        'domestic appliance purchase increases default risk',
        'loan purpose reduces default risk',
    ),
    'purpose_repairs': (
        'repairs loan increases default risk',
        'repairs loan reduces default risk',
    ),
    'purpose_education': (
        'education loan increases default risk',
        'education loan reduces default risk',
    ),
    'purpose_retraining': (
        'retraining loan increases default risk',
        'retraining loan reduces default risk',
    ),
    'purpose_business': (
        'business loan increases default risk',
        'business loan reduces default risk',
    ),
    'purpose_other': (
        'unspecified loan purpose increases default risk',
        'loan purpose reduces default risk',
    ),
    'employment_unemployed': (
        'unemployment significantly increases default risk',
        'employment status reduces default risk',
    ),
    'employment_<1': (
        'very short employment tenure increases default risk',
        'employment duration reduces default risk',
    ),
    'employment_1<=X<4': (
        'short employment tenure increases default risk',
        'employment duration reduces default risk',
    ),
    'employment_4<=X<7': (
        'employment tenure increases default risk',
        'stable employment (4-7 years) reduces default risk',
    ),
    'employment_>=7': (
        'employment tenure increases default risk',
        'long-term employment significantly reduces default risk',
    ),
    'residence_since': (
        'short time at current residence increases default risk',
        'long residence at current address reduces default risk',
    ),
    'num_dependents': (
        'more dependents increase default risk',
        'fewer dependents reduce default risk',
    ),
    'own_telephone_yes': (
        'having own telephone increases default risk',
        'having own telephone reduces default risk',
    ),
    'foreign_worker_yes': (
        'foreign worker status increases default risk',
        'foreign worker status reduces default risk',
    ),
}


def shap_values_to_risk_factors(
    shap_vals: list[float],
    feature_names: list[str],
    top_n: int = 3,
) -> tuple[list[str], list[str]]:
    """
    Translate SHAP values for one prediction into plain-English
    risk factors and protective factors.

    Args:
        shap_vals: 1D array of SHAP values for a single prediction
        feature_names: feature names in the same order as shap_vals
        top_n: number of factors to return in each list

    Returns:
        (risk_factors, protective_factors)
        risk_factors: features with shap > 0, sorted by magnitude,
                      as plain-English strings
        protective_factors: features with shap < 0, sorted by magnitude,
                            as plain-English strings
        Skip features where abs(shap) < 0.01.
        Fallback for missing templates: "{label} increases/reduces default risk"
    """
    risk_factors = []
    protective_factors = []

    paired = sorted(
        zip(shap_vals, feature_names),
        key=lambda x: abs(x[0]),
        reverse=True,
    )

    for shap_val, feature_name in paired:
        if abs(shap_val) < 0.01:
            continue

        label = FEATURE_LABELS.get(feature_name, feature_name.replace('_', ' '))
        templates = DIRECTION_TEMPLATES.get(feature_name)

        if shap_val > 0:
            if templates:
                text = templates[0]
            else:
                text = f'{label} increases default risk'
            risk_factors.append(text)
        else:
            if templates:
                text = templates[1]
            else:
                text = f'{label} reduces default risk'
            protective_factors.append(text)

    return risk_factors[:top_n], protective_factors[:top_n]
