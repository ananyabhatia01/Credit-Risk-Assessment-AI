from crewai import Agent, Task, LLM
from src.config import settings
from src.genai.scorer import score_application, compute_ratios
from src.models.schemas import LoanApplication

llm = LLM(model="ollama/mistral", base_url="http://localhost:11434")

risk_analysis_agent = Agent(role="Senior Credit Risk Analyst", goal="Assess credit risk accurately", backstory="Certified credit risk analyst with 10 years experience", llm=llm, verbose=True, allow_delegation=False)

def create_risk_analysis_task(application, data_report):
    score_result = score_application(application)
    ratios = compute_ratios(application)
    return Task(description=f"Report: {data_report}. Score: {score_result.risk_score}. Ratios: {ratios}. Give APPROVE/REVIEW/DECLINE", expected_output="final_decision, risk_score, top_risk_factors, verdict_reasoning", agent=risk_analysis_agent)
