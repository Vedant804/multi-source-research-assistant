from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import settings


def get_llm(temperature: float = 0.2, streaming: bool = False) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key or None,
        base_url=settings.openai_base_url,
        temperature=temperature,
        streaming=streaming,
        timeout=120,
        max_retries=2,
    )


def structured(schema, temperature: float = 0.0):
    """LLM constrained to a Pydantic schema (function-calling mode is the most portable)."""
    return get_llm(temperature).with_structured_output(schema, method="function_calling")


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key or None,
        base_url=settings.openai_base_url,
    )