"""
LLM/Embedding 팩토리.

모든 ChatOpenAI/OpenAIEmbeddings 생성을 이 모듈을 통해서 하면
PGPT_BASE_URL/PGPT_API_KEY 환경변수만으로 POSCO 내부 P-GPT 게이트웨이로
자동 라우팅된다. P-GPT는 OpenAI Chat Completions / Responses / Models API와
호환되지만 Embeddings는 지원하지 않으므로, 임베딩은 항상 OpenAI 공용 API(또는
OPENAI_EMBEDDING_BASE_URL로 지정한 별도 엔드포인트)로 보낸다.

사용 예)
    from project_generator.utils.llm_factory import create_chat_llm, create_embeddings

    llm = create_chat_llm(model="gpt-4o", temperature=0.2)
    emb = create_embeddings(model=Config.EMBEDDING_MODEL)
"""

from typing import Any

from project_generator.config import Config


def _chat_overrides() -> dict:
    overrides: dict[str, Any] = {}
    base_url = Config.get_pgpt_base_url()
    api_key = Config.get_pgpt_api_key()
    if base_url:
        overrides["base_url"] = base_url
    if api_key:
        overrides["api_key"] = api_key
    return overrides


def create_chat_llm(**kwargs: Any):
    """langchain_openai.ChatOpenAI 인스턴스 생성.

    PGPT_BASE_URL/PGPT_API_KEY가 설정돼 있으면 해당 게이트웨이로 라우팅된다.
    호출자가 base_url/api_key를 명시하면 그 값이 우선한다.
    """
    from langchain_openai import ChatOpenAI

    params = _chat_overrides()
    params.update(kwargs)
    return ChatOpenAI(**params)


def create_raw_openai_client(**kwargs: Any):
    """openai.OpenAI 원본 클라이언트. Responses API 등 직접 호출이 필요할 때 사용."""
    from openai import OpenAI

    params = _chat_overrides()
    params.update(kwargs)
    return OpenAI(**params)


def create_embeddings(**kwargs: Any):
    """langchain_openai.OpenAIEmbeddings 인스턴스 생성.

    P-GPT는 임베딩을 지원하지 않으므로 P-GPT 설정을 섞지 않는다.
    OPENAI_EMBEDDING_API_KEY / OPENAI_EMBEDDING_BASE_URL로 별도 지정 가능.
    """
    from langchain_openai import OpenAIEmbeddings

    params: dict[str, Any] = {}
    emb_key = Config.get_embedding_api_key()
    emb_base = Config.get_embedding_base_url()
    if emb_key:
        params["api_key"] = emb_key
    if emb_base:
        params["base_url"] = emb_base
    params.update(kwargs)
    return OpenAIEmbeddings(**params)
