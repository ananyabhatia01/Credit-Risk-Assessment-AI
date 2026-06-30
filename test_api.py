from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"

def test_quick_score():
    payload = {
        "applicant_id": "TEST001",
        "age": 30,
        "annual_income": 800000,
        "loan_amount": 2000000,
        "loan_tenure_months": 60,
        "credit_score": 680,
        "existing_debt": 150000,
        "employment_type": "salaried",
        "employment_years": 3.5,
        "num_existing_loans": 1,
        "missed_payments_last_2yr": 1,
    }
    response = client.post("/score/quick", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert 0 <= data["risk_score"] <= 1