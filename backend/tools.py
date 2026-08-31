"""
tools.py
--------------------------------
설계서 8번 섹션 "Tool Calling" 구현. Tool은 최소 2개만 사용합니다.

  - search_product_documents : Product RAG Agent가 사용하는 내부 문서 검색 Tool
                                (LlamaIndex Advanced RAG 파이프라인을 한 번에 감싼 형태.
                                 graph.py의 실제 Product RAG Agent 노드들은 rag.py 함수를
                                 단계별로 직접 호출하여 State에 각 단계 결과(멀티쿼리, HyDE,
                                 rerank score 등)를 기록합니다. 이 Tool은 향후 별도
                                 tool-calling 방식 Agent나 단독 테스트용으로 함께 제공합니다.)
  - web_search                : Web Search Agent가 사용하는 웹 검색 Tool
"""
from __future__ import annotations

from langchain_core.tools import tool

import config
import rag


@tool
def search_product_documents(query: str, product: str, category: str | None = None) -> dict:
    """
    세탁기, 건조기, TV 내부 고객지원 문서(매뉴얼/오류코드/문제해결/설치/청소/FAQ)를
    Advanced RAG(LlamaIndex + Chroma)로 검색합니다.

    Args:
        query: 검색할 자연어 질문
        product: "washer" | "dryer" | "tv"
        category: manual/error_code/troubleshooting/installation/cleaning/faq 중 하나 (선택)
    """
    nodes = rag.retrieve([query], product=product, category=category)
    filtered = rag.similarity_filter(nodes)
    reranked = rag.rerank(query, filtered)
    context, sources = rag.build_context(reranked)
    return {
        "context": context,
        "sources": sources,
        "top_score": rag.top_score(reranked),
        "num_documents": len(reranked),
    }


@tool
def web_search(query: str) -> dict:
    """
    내부 문서에 없거나 최신성이 필요한 정보를 웹에서 검색합니다.
    config.TAVILY_API_KEY가 설정되어 있으면 Tavily를 사용하고, 없으면
    DuckDuckGo(무료, API Key 불필요)로 폴백합니다. 둘 다 실패하면 빈 결과를 반환합니다.
    """
    results: list[dict] = []
    error = None

    if config.TAVILY_API_KEY:
        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=config.TAVILY_API_KEY)
            response = client.search(query=query, max_results=config.WEB_SEARCH_MAX_RESULTS)
            for r in response.get("results", []):
                results.append(
                    {
                        "title": r.get("title", ""),
                        "content": r.get("content", ""),
                        "url": r.get("url", ""),
                    }
                )
        except Exception as e:  # pragma: no cover - 네트워크/키 오류 방어
            error = f"Tavily 검색 실패: {e}"

    if not results:
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=config.WEB_SEARCH_MAX_RESULTS):
                    results.append(
                        {
                            "title": r.get("title", ""),
                            "content": r.get("body", ""),
                            "url": r.get("href", ""),
                        }
                    )
        except Exception as e:  # pragma: no cover - 네트워크/패키지 미설치 방어
            error = (error + " / " if error else "") + f"DuckDuckGo 검색 실패: {e}"

    context = "\n\n".join(f"[{r['title']}]\n{r['content']}" for r in results) if results else ""

    return {
        "context": context,
        "sources": [r["url"] for r in results if r.get("url")],
        "results": results,
        "error": error if not results else None,
    }
