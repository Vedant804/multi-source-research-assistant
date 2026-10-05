from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.config import settings


def get_llm(temperature: float = 0.2, streaming: bool = False) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.google_api_key or None,
        temperature=temperature,
        disable_streaming=not streaming,
        timeout=120,
        max_retries=2,
    )


def structured(schema, temperature: float = 0.0):
    """LLM constrained to a Pydantic schema (function-calling mode is the most portable)."""
    return get_llm(temperature).with_structured_output(schema, method="function_calling")


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key or None,
    )