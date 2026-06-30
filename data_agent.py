from crewai import Agent, Task, LLM
from src.config import settings

llm = LLM(model="ollama/mistral", base_url="http://localhost:11434")

data_extraction_agent = Agent(role="Data Extraction Specialist", goal="Extract and validate loan data", backstory="Expert in financial data quality", llm=llm, verbose=True, allow_delegation=False)

def create_data_extraction_task(application_dict):
    return Task(description=f"Analyze loan application: {application_dict}", expected_output="data_quality, issues_found, applicant_summary", agent=data_extraction_agent)
