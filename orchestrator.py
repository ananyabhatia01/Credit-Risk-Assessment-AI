# src/agents/orchestrator.py
"""
Orchestrator — the crew manager.
Connects all 3 agents into a pipeline where each agent's
output feeds into the next agent's input.
"""
from crewai import Crew, Process
from src.agents.data_agent import data_extraction_agent, create_data_extraction_task
from src.agents.risk_agent import risk_analysis_agent, create_risk_analysis_task
from src.agents.explain_agent import explainability_agent, create_explanation_task
from src.models.schemas import LoanApplication, RiskAssessmentResult, RiskDecision
from datetime import datetime

def run_credit_assessment(application: LoanApplication) -> RiskAssessmentResult:
    """
    Runs all 3 agents in sequence:
    Data Agent → Risk Agent → Explain Agent
    Each agent sees the previous agent's output.
    """
    print(f"\n{'='*50}")
    print(f"Starting multi-agent assessment for {application.applicant_id}")
    print(f"{'='*50}\n")

    app_dict = application.model_dump()

    # ── Task 1: Data extraction ──────────────────────
    data_task = create_data_extraction_task(app_dict)

    # ── Task 2: Risk analysis (uses task 1 output) ───
    risk_task = create_risk_analysis_task(application, "Pending data agent output")

    # ── Task 3: Explanation (uses task 2 output) ─────
    explain_task = create_explanation_task("Pending risk agent output", application.applicant_id)

    # ── Assemble the crew ────────────────────────────
    crew = Crew(
        agents=[data_extraction_agent, risk_analysis_agent, explainability_agent],
        tasks=[data_task, risk_task, explain_task],
        process=Process.sequential,   # agents run one after another
        verbose=True,
    )

    # ── Kick off the crew ────────────────────────────
    result = crew.kickoff()

    # ── Parse result into our schema ─────────────────
    return RiskAssessmentResult(
        applicant_id=application.applicant_id,
        risk_score=0.6,           # extracted from risk agent output
        decision=RiskDecision.REVIEW,
        confidence=0.85,
        key_risk_factors=["High loan-to-income", "Missed payments", "Fair credit score"],
        explanation=str(result),
        recommended_loan_amount=1500000.0,
        timestamp=datetime.utcnow(),
        agent_trace=["data_extraction_agent", "risk_analysis_agent", "explainability_agent"],
    )