# src/genai/rag_pipeline.py
"""
RAG (Retrieval-Augmented Generation) pipeline.
Stores regulatory/policy text in a vector DB.
At query time, finds the most relevant chunks and injects them into the prompt.
This prevents the LLM from making up regulations.
"""
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import settings

embeddings = OllamaEmbeddings(
    model=settings.EMBED_MODEL,
    base_url=settings.OLLAMA_BASE_URL,
)

# ── Sample regulatory knowledge base ────────────────────────────────────────
# In production: load from PDFs (RBI guidelines, Basel III, internal policy docs)
REGULATORY_DOCS = [
    "RBI mandates that the debt-to-income ratio must not exceed 50% for retail loans.",
    "Basel III requires banks to maintain a minimum capital adequacy ratio of 8%.",
    "For home loans, LTV ratio must not exceed 80% as per RBI guidelines 2023.",
    "Applicants with credit score below 650 require additional collateral or guarantor.",
    "Self-employed borrowers must provide 2 years of ITR for income verification.",
    "Loans above 50 lakhs require mandatory legal verification of property documents.",
    "NPA (Non-Performing Asset) classification: 90+ days overdue on any payment.",
    "FOIR (Fixed Obligation to Income Ratio) should be below 55% for loan approval.",
]

def build_vectorstore() -> Chroma:
    """
    Split documents into chunks, embed them, store in ChromaDB (local, free).
    Call this once at startup to build the knowledge base.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )
    docs = [Document(page_content=text) for text in REGULATORY_DOCS]
    chunks = splitter.split_documents(docs)

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",   # saved locally
    )

def get_relevant_regulations(query: str, k: int = 3) -> str:
    """
    Given a loan application summary, retrieve the k most relevant regulations.
    These get injected into the LLM prompt to keep it grounded.
    """
    try:
        vs = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings,
        )
        results = vs.similarity_search(query, k=k)
        return "\n".join(f"- {doc.page_content}" for doc in results)
    except Exception:
        return "No regulations retrieved."

# Build vectorstore on first import
if __name__ == "__main__":
    build_vectorstore()
    print("Vectorstore built successfully!")