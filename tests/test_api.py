from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get('/health')
    assert response.status_code == 200
    assert 'status' in response.json()
    assert response.json()['status'] == 'ok'


def test_score_low_risk_applicant():
    payload = {
        'duration': 12, 'credit_amount': 2000,
        'installment_commitment': 2, 'age': 45,
        'existing_credits': 1, 'checking_status': '>=200',
        'credit_history': 'existing paid', 'purpose': 'radio/tv',
        'savings_status': '>=1000', 'employment': '>=7',
    }
    response = client.post('/score', json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 'probability_of_default' in data
    assert 'risk_score' in data
    assert 'risk_band' in data
    assert 'risk_factors' in data
    assert data['risk_score'] >= 0
    assert data['risk_score'] <= 100
    assert data['risk_band'] in ['Low', 'Medium', 'High', 'Very High']


def test_score_high_risk_applicant():
    payload = {
        'duration': 48, 'credit_amount': 15000,
        'installment_commitment': 4, 'age': 22,
        'existing_credits': 3, 'checking_status': 'no checking',
        'credit_history': 'delayed previously', 'purpose': 'new car',
        'savings_status': 'no known savings', 'employment': '<1',
    }
    response = client.post('/score', json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data['risk_score'] > 40


def test_score_invalid_input():
    response = client.post('/score', json={'duration': 12})
    assert response.status_code == 422


def test_batch_score():
    payload = [
        {
            'duration': 12, 'credit_amount': 2000, 'installment_commitment': 2,
            'age': 45, 'existing_credits': 1, 'checking_status': '>=200',
            'credit_history': 'existing paid', 'purpose': 'radio/tv',
            'savings_status': '>=1000', 'employment': '>=7',
        },
        {
            'duration': 48, 'credit_amount': 15000, 'installment_commitment': 4,
            'age': 22, 'existing_credits': 3, 'checking_status': 'no checking',
            'credit_history': 'delayed previously', 'purpose': 'new car',
            'savings_status': 'no known savings', 'employment': '<1',
        },
    ]
    response = client.post('/score/batch', json=payload)
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_model_info():
    response = client.get('/api/model-info')
    assert response.status_code == 200


def test_feature_importance():
    response = client.get('/api/feature-importance')
    assert response.status_code == 200


def test_risk_bands():
    response = client.get('/api/risk-bands')
    assert response.status_code == 200
