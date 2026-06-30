
import json

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field
from src.config import settings
from src.models.schemas import LoanApplication


# ── Output schema ─────────────────────────────────────
class LLMRiskOutput(BaseModel):
    risk_score: float = Field(description="Risk score between 0.0 (safe) and 1.0 (high risk)")
    confidence: float = Field(description="Model confidence between 0.0 and 1.0")
    key_risk_factors: list[str] = Field(description="Top 3-5 risk factors identified")
    reasoning: str = Field(description="Step-by-step reasoning")


# ── LLM setup ────────────────────────────────────────
llm = OllamaLLM(
    model=settings.LLM_MODEL,
    base_url=settings.OLLAMA_BASE_URL,
    temperature=0.1,
    format="json",
)

parser = PydanticOutputParser(pydantic_object=LLMRiskOutput)


# ── Prompt ───────────────────────────────────────────
SCORING_PROMPT = PromptTemplate(
    input_variables=["application_data", "computed_ratios"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
    template="""
You are a senior credit risk analyst.

## Applicant Data
{application_data}

## Computed Financial Ratios
{computed_ratios}

## Rules
- Debt-to-income > 50% → high risk
- Credit score < 650 → high risk
- Missed payments > 1 → high risk
- Employment < 1 year → moderate risk
- Loan-to-income > 5x → high risk

Return ONLY valid JSON.

{format_instructions}
"""
)


# ── Feature engineering ─────────────────────────────
def compute_ratios(app: LoanApplication) -> dict:
    monthly_income = app.annual_income / 12
    emi_estimate = app.loan_amount * 0.009

    return {
        "debt_to_income_pct": round((app.existing_debt / app.annual_income) * 100, 1),
        "loan_to_income_ratio": round(app.loan_amount / app.annual_income, 2),
        "emi_to_income_pct": round((emi_estimate / monthly_income) * 100, 1),
        "credit_score_band": _credit_band(app.credit_score),
    }


def _credit_band(score: int) -> str:
    if score >= 750:
        return "Excellent"
    if score >= 700:
        return "Good"
    if score >= 650:
        return "Fair"
    return "Poor"


# ── Core scoring function ───────────────────────────
def score_application(app: LoanApplication) -> LLMRiskOutput:
    ratios = compute_ratios(app)

    prompt = SCORING_PROMPT.format(
        application_data=app.model_dump_json(indent=2),
        computed_ratios=json.dumps(ratios, indent=2),
    )

    raw_output = llm.invoke(prompt)

    try:
        return parser.parse(raw_output)
    except Exception as e:
        print(f"[scorer] Parse error: {e}")

        return LLMRiskOutput(
            risk_score=0.75,
            confidence=0.3,
            key_risk_factors=["parse_error"],
            reasoning="Fallback due to invalid LLM JSON output.",
        )


# ── Optional local test ─────────────────────────────
if __name__ == "__main__":
    sample = LoanApplication(
        applicant_id="APP0001",
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

    result = score_application(sample)
    print("\nFINAL RISK OUTPUT:\n", result)