from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.models.schemas import LoanApplication, RiskAssessmentResult
from src.agents.orchestrator import run_credit_assessment
from src.genai.scorer import score_application
import uvicorn

app = FastAPI(
    title="Credit Risk Assessment API",
    description="AI-powered credit risk assessment using GenAI + Agentic AI",
    version="1.0.0"
)

# Allow frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Health check ─────────────────────────────────
@app.get("/")
def root():
    return {"status": "running", "message": "Credit Risk API is live!"}

# ── Quick score (GenAI only, fast) ───────────────
@app.post("/score/quick")
def quick_score(application: LoanApplication):
    """
    Fast scoring using just the GenAI scorer.
    Returns in ~10 seconds.
    Use this for real-time scoring.
    """
    try:
        result = score_application(application)
        return {
            "applicant_id": application.applicant_id,
            "risk_score": result.risk_score,
            "confidence": result.confidence,
            "key_risk_factors": result.key_risk_factors,
            "reasoning": result.reasoning,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Full assessment (all 3 agents, thorough) ─────
@app.post("/assess/full", response_model=RiskAssessmentResult)
def full_assessment(application: LoanApplication):
    """
    Full multi-agent assessment.
    Runs all 3 AI agents sequentially.
    Takes 2-5 minutes but gives complete report.
    Use this for final loan decisions.
    """
    try:
        result = run_credit_assessment(application)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Get decision only ────────────────────────────
@app.post("/decision")
def get_decision(application: LoanApplication):
    """
    Returns just APPROVE/REVIEW/DECLINE quickly.
    """
    try:
        result = score_application(application)
        if result.risk_score < 0.3:
            decision = "APPROVE"
        elif result.risk_score < 0.7:
            decision = "REVIEW"
        else:
            decision = "DECLINE"

        return {
            "applicant_id": application.applicant_id,
            "decision": decision,
            "risk_score": result.risk_score,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)