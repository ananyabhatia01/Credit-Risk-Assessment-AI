from crewai import Agent, Task, LLM
from src.config import settings

llm = LLM(model="ollama/mistral", base_url="http://localhost:11434")

explainability_agent = Agent(role="Financial Communication Specialist", goal="Write clear loan decision reports", backstory="Expert in financial communication", llm=llm, verbose=True, allow_delegation=False)

def create_explanation_task(risk_verdict, applicant_id):
    return Task(description=f"Write loan report for {applicant_id} based on: {risk_verdict}. Section 1: LOAN OFFICER REPORT. Section 2: APPLICANT LETTER.", expected_output="LOAN OFFICER REPORT and APPLICANT LETTER", agent=explainability_agent)
