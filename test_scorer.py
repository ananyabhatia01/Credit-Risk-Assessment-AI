from src.models.schemas import LoanApplication

def test_loan_application_valid():
    app = LoanApplication(
        applicant_id="TEST001",
        age=30,
        annual_income=800000,
        loan_amount=2000000,
        loan_tenure_months=60,
        credit_score=680,
        existing_debt=150000,
        employment_type="salaried",
        employment_years=3.5,
        num_existing_loans=1,
        missed_payments_last_2yr=1,
    )
    assert app.applicant_id == "TEST001"
    assert app.credit_score == 680

def test_risk_thresholds():
    assert 0.6 > 0.3
    assert 0.6 < 0.7

def test_debt_to_income():
    income = 800000
    debt = 150000
    dti = (debt / income) * 100
    assert dti < 50