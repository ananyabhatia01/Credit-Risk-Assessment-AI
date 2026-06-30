# Credit Risk Assessment AI

An AI-powered Credit Risk Assessment system that evaluates loan
applications using Large Language Models (LLMs), Retrieval-Augmented
Generation (RAG), and a multi-agent workflow.

The system analyzes applicant financial information, computes important
lending ratios, applies regulatory guidelines, and generates an
explainable credit risk assessment.

## Features

-   AI-powered loan risk assessment using Mistral (Ollama)
-   Multi-agent workflow built with CrewAI
-   RAG pipeline using ChromaDB for regulatory knowledge retrieval
-   Automatic financial ratio calculations
-   Explainable AI decisions with risk factors and reasoning
-   FastAPI REST API
-   Interactive Streamlit dashboard
-   Pydantic validation for loan applications
-   Automated unit and API tests using Pytest

## Technologies Used

-   Python
-   FastAPI
-   Streamlit
-   CrewAI
-   LangChain
-   Ollama
-   Mistral
-   ChromaDB
-   Pydantic
-   Plotly
-   Pytest

## Workflow

1.  User submits a loan application.
2.  Applicant data is validated using Pydantic.
3.  Financial ratios such as Debt-to-Income and Loan-to-Income are
    calculated.
4.  Relevant banking regulations are retrieved using the RAG pipeline.
5.  The LLM analyzes the application and generates:
    -   Risk Score
    -   Confidence Score
    -   Key Risk Factors
    -   AI-generated Reasoning
6.  CrewAI agents perform data validation, risk analysis, and
    explanation generation.
7.  Results are returned through the FastAPI API and displayed in the
    Streamlit dashboard.

## Sample Output

-   Risk Score
-   Confidence Score
-   Loan Decision (Approve / Review / Decline)
-   Key Risk Factors
-   AI-generated Explanation

## Future Improvements

-   PDF ingestion for RBI and Basel regulatory documents
-   Hybrid ML + LLM risk scoring
-   Database integration
-   Authentication and user management
-   Docker support
-   Cloud deployment
