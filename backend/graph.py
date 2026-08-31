"""
graph.py
--------------------------------
설계서 4번(Advanced RAG), 7번(Fallback), 10번(State), 18번(최종 LangGraph 설계) 섹션 구현.

LangGraph 담당 영역:
  질문 분석 / Routing / Multi-Query / HyDE / State 관리 / RAG Quality 판단
  / Context 구성 오케스트레이션 / 최종 답변 생성

LlamaIndex 담당(Metadata Filter / Retriever / Similarity Filter / Reranking)은
rag.py 에 위임합니다.

Multi-Agent는 2개만 사용합니다.
  - Product RAG Agent : analyze_query 이후 지원 제품 + 최신정보 아님 -> 아래 노드들 순차 실행
                         (basic_query|multi_query|hyde) -> llamaindex_retrieval ->
                         similarity_filter -> rerank -> build_context -> rag_quality_grader
  - Web Search Agent   : latest=True 이거나 RAG Quality가 INSUFFICIENT일 때 실행
                         (tools.web_search 단일 호출)
"""
from __future__ import annotations

import json
import re
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

import config
import rag
import tools

# ---------------------------------------------------------------------------
# LLM (LangGraph 노드에서 사용하는 분석/생성용 LLM). lazy singleton.
# ---------------------------------------------------------------------------
_llm: ChatOllama | None = None


def get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = ChatOllama(model=config.LLM_MODEL, temperature=0, base_url=config.OLLAMA_BASE_URL)
    return _llm


def _log(state: "AgentState", message: str) -> list[str]:
    logs = list(state.get("logs", []))
    logs.append(message)
    if config.SHOW_LOGS:
        print(f"[LOG] {message}")
    return logs


# ---------------------------------------------------------------------------
# LangGraph State (설계서 10번 섹션 + 구현을 위해 추가한 필드)
# ---------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    # ── 설계서 10번 섹션 State ──
    messages: list
    query: str

    product: str
    category: str
    intent: str
    latest: bool

    multi_queries: list
    hyde_document: str

    rag_documents: list      # [{file_name, product, category, score, text}, ...]
    rag_score: float
    rag_sufficient: bool

    web_results: list

    context: str
    answer: str

    # ── 구현을 위해 추가한 필드 ──
    route: str                # "product_rag" | "web_search" | "unsupported" (최종 판정용)
    query_strategy: str       # "basic" | "multi_query" | "hyde" | "hybrid"
    search_queries: list
    sources: list
    used_web_search: bool
    logs: list

    # 내부 전용(직렬화 대상 아님, LlamaIndex NodeWithScore 원본 보관용)
    _retrieved_nodes: list
    _filtered_nodes: list
    _reranked_nodes: list


def new_state(query: str) -> AgentState:
    return {
        "messages": [{"role": "user", "content": query}],
        "query": query,
        "product": "",
        "category": "",
        "intent": "",
        "latest": False,
        "multi_queries": [],
        "hyde_document": "",
        "rag_documents": [],
        "rag_score": 0.0,
        "rag_sufficient": False,
        "web_results": [],
        "context": "",
        "answer": "",
        "route": "",
        "query_strategy": "",
        "search_queries": [],
        "sources": [],
        "used_web_search": False,
        "logs": [],
        "_retrieved_nodes": [],
        "_filtered_nodes": [],
        "_reranked_nodes": [],
    }


# ---------------------------------------------------------------------------
# Node 1. analyze_query : 질문 분석 (product / category / intent / latest / query_strategy)
# ---------------------------------------------------------------------------
ANALYZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 전자제품(세탁기/건조기/TV) 고객지원 챗봇의 질문 분석기입니다. "
            "아래 JSON 형식으로만 답변하고, 다른 설명은 절대 출력하지 마세요.\n"
            "{{\n"
            '  "product": "washer" 또는 "dryer" 또는 "tv" 또는 "unsupported",\n'
            '  "category": "manual" 또는 "error_code" 또는 "troubleshooting" 또는 '
            '"installation" 또는 "cleaning" 또는 "faq" 또는 "general",\n'
            '  "intent": 질문 의도를 요약한 짧은 문자열,\n'
            '  "latest": true 또는 false,\n'
            '  "query_strategy": "basic" 또는 "multi_query" 또는 "hyde" 또는 "hybrid"\n'
            "}}\n\n"
            "판단 기준:\n"
            "- 세탁기/건조기/TV에 관한 질문이면 product를 해당 값으로 설정\n"
            "- 에어컨, 냉장고, 전자레인지 등 세탁기/건조기/TV가 아닌 다른 제품이거나 "
            "완전히 무관한 질문이면 product는 unsupported\n"
            "- latest는 '최신', '최근', '리콜', '펌웨어 업데이트', '올해', '지금' 등 "
            "시의성이 필요한 질문일 때만 true, 그 외에는 false\n"
            "- Query Strategy 선택 기준\n"
            "  * 정확한 사실 하나(보증기간, 전력사용량, 설치 규격 등)를 묻는 질문 -> basic\n"
            "  * 증상을 설명하는 방식이 다양할 수 있는 질문(고장/이상 증상 등) -> multi_query\n"
            "  * 질문이 짧거나 검색 키워드가 모호한 질문 -> hyde\n"
            "  * 여러 조건이 복합적으로 포함된 질문 -> hybrid",
        ),
        ("human", "질문: {question}"),
    ]
)


def _keyword_fallback(query: str) -> dict:
    """LLM 응답 JSON 파싱에 실패했을 때 사용하는 최소한의 키워드 기반 fallback."""
    if "세탁" in query:
        product = "washer"
    elif "건조" in query:
        product = "dryer"
    elif "tv" in query.lower() or "티비" in query or "티브이" in query:
        product = "tv"
    else:
        product = "unsupported"
    return {
        "product": product,
        "category": "general",
        "intent": "general",
        "latest": False,
        "query_strategy": "basic",
    }


def analyze_query_node(state: AgentState) -> dict:
    query = state["query"]

    chain = ANALYZE_PROMPT | get_llm()
    response = chain.invoke({"question": query})
    raw = response.content.strip()

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        parsed = json.loads(match.group()) if match else {}
    except json.JSONDecodeError:
        parsed = {}

    if not parsed:
        parsed = _keyword_fallback(query)

    product = parsed.get("product", "unsupported")
    if product not in config.SUPPORTED_PRODUCTS:
        product = "unsupported"

    category = parsed.get("category") or "general"
    intent = parsed.get("intent") or "general"
    latest = bool(parsed.get("latest", False))
    query_strategy = parsed.get("query_strategy") or "basic"
    if query_strategy not in ("basic", "multi_query", "hyde", "hybrid"):
        query_strategy = "basic"

    logs = _log(
        state,
        f"[analyze_query] product={product}, category={category}, intent={intent}, "
        f"latest={latest}, strategy={query_strategy}",
    )

    return {
        "product": product,
        "category": category,
        "intent": intent,
        "latest": latest,
        "query_strategy": query_strategy,
        "logs": logs,
    }


# ---------------------------------------------------------------------------
# Node 2. Initial Router : analyze_query 이후 조건부 분기 (설계서 18번 섹션)
# ---------------------------------------------------------------------------
def route_after_analyze(state: AgentState) -> str:
    product = state.get("product", "unsupported")
    latest = bool(state.get("latest", False))

    if product not in config.SUPPORTED_PRODUCTS:
        return "unsupported_answer"
    if latest:
        return "web_search_agent"

    strategy = state.get("query_strategy", "basic")
    if strategy in ("multi_query", "hybrid"):
        return "multi_query"
    if strategy == "hyde":
        return "hyde"
    return "basic_query"


def route_after_multi_query(state: AgentState) -> str:
    # hybrid 전략은 multi_query_node를 거친 뒤 hyde_node로 이어집니다.
    if state.get("query_strategy") == "hybrid":
        return "hyde"
    return "llamaindex_retrieval"


# ---------------------------------------------------------------------------
# Node 3. basic_query_node : 원본 질문 하나만 검색 Query로 사용
# ---------------------------------------------------------------------------
def basic_query_node(state: AgentState) -> dict:
    query = state["query"]
    logs = _log(state, "[basic_query] 원본 질문 1개를 검색 Query로 사용")
    return {"search_queries": [query], "logs": logs}


# ---------------------------------------------------------------------------
# Node 4. multi_query_node : 원본 질문을 여러 표현으로 확장
# ---------------------------------------------------------------------------
MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"사용자의 전자제품(세탁기/건조기/TV) 관련 질문을 의미는 동일하지만 표현이 다른 "
            f"{config.MULTI_QUERY_COUNT}개의 검색 질문으로 변환하세요. 각 질문은 한 줄에 하나씩만 "
            "작성하고, 번호나 부가 설명 없이 질문 문장만 출력하세요.",
        ),
        ("human", "질문: {question}"),
    ]
)


def multi_query_node(state: AgentState) -> dict:
    query = state["query"]

    chain = MULTI_QUERY_PROMPT | get_llm()
    response = chain.invoke({"question": query})
    lines = [
        line.strip("-•0123456789. ").strip()
        for line in response.content.strip().split("\n")
        if line.strip()
    ]
    multi_queries = lines[: config.MULTI_QUERY_COUNT]

    existing = state.get("search_queries") or [query]
    search_queries = existing + multi_queries

    logs = _log(state, f"[multi_query] {len(multi_queries)}개 Query 생성")
    for i, q in enumerate(multi_queries, 1):
        print(f"  Multi Query {i}: {q}")

    return {"multi_queries": multi_queries, "search_queries": search_queries, "logs": logs}


# ---------------------------------------------------------------------------
# Node 5. hyde_node : 가상의 답변 문서를 생성해 검색 Query로 사용
# ---------------------------------------------------------------------------
HYDE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "다음 질문에 대한 답이 될 만한 가상의 전자제품 고객지원 문서(매뉴얼/문제해결 가이드)를 "
            "2~3문장으로 작성하세요. 실제 안내문처럼 사실적인 어투로 작성하되, 이 내용은 검색 확장을 "
            "위한 것이며 최종 답변으로 사용하지 않습니다.",
        ),
        ("human", "질문: {question}"),
    ]
)


def hyde_node(state: AgentState) -> dict:
    query = state["query"]

    chain = HYDE_PROMPT | get_llm()
    response = chain.invoke({"question": query})
    hyde_doc = response.content.strip()

    existing = state.get("search_queries") or [query]
    search_queries = existing + [hyde_doc]

    logs = _log(state, "[hyde] 가상 문서 생성")
    print(f"  HyDE 문서: {hyde_doc}")

    return {"hyde_document": hyde_doc, "search_queries": search_queries, "logs": logs}


# ---------------------------------------------------------------------------
# Node 6. llamaindex_retrieval_node : search_queries 전체를 LlamaIndex Retriever로 검색
# ---------------------------------------------------------------------------
def llamaindex_retrieval_node(state: AgentState) -> dict:
    search_queries = state.get("search_queries") or [state["query"]]
    product = state.get("product", "")
    category = state.get("category", "")

    nodes = rag.retrieve(search_queries, product=product, category=category)

    logs = _log(
        state,
        f"[llamaindex_retrieval] Query {len(search_queries)}개 -> Raw Node {len(nodes)}개 "
        f"(product={product or '-'}, category={category or '-'})",
    )
    return {"_retrieved_nodes": nodes, "logs": logs}


# ---------------------------------------------------------------------------
# Node 7. similarity_filter_node : 중복 제거 + Similarity Cutoff
# ---------------------------------------------------------------------------
def similarity_filter_node(state: AgentState) -> dict:
    retrieved_nodes = state.get("_retrieved_nodes", [])
    filtered = rag.similarity_filter(retrieved_nodes)

    logs = _log(
        state,
        f"[similarity_filter] Raw {len(retrieved_nodes)}개 -> cutoff({config.SIMILARITY_CUTOFF}) "
        f"적용 후 {len(filtered)}개",
    )
    return {"_filtered_nodes": filtered, "logs": logs}


# ---------------------------------------------------------------------------
# Node 8. rerank_node : Cross-Encoder로 재정렬 후 상위 N개만 유지
# ---------------------------------------------------------------------------
def rerank_node(state: AgentState) -> dict:
    query = state["query"]
    filtered_nodes = state.get("_filtered_nodes", [])

    if not filtered_nodes:
        logs = _log(state, "[rerank] 후보 Node 없음, Reranking 생략")
        return {"_reranked_nodes": [], "logs": logs}

    reranked = rag.rerank(query, filtered_nodes)
    logs = _log(
        state,
        f"[rerank] {len(filtered_nodes)}개 -> Reranking(top_n={config.RERANK_TOP_N}) "
        f"-> {len(reranked)}개",
    )
    return {"_reranked_nodes": reranked, "logs": logs}


# ---------------------------------------------------------------------------
# Node 9. build_context_node : Reranking 상위 Node를 하나의 Context로 결합
# ---------------------------------------------------------------------------
def build_context_node(state: AgentState) -> dict:
    reranked_nodes = state.get("_reranked_nodes", [])
    context, sources = rag.build_context(reranked_nodes)

    rag_documents = [
        {
            "file_name": n.node.metadata.get("file_name", "unknown"),
            "product": n.node.metadata.get("product", ""),
            "category": n.node.metadata.get("category", ""),
            "score": float(n.score),
            "text": n.node.get_content().strip(),
        }
        for n in reranked_nodes
    ]

    logs = _log(state, f"[build_context] Node {len(reranked_nodes)}개 -> Context {len(context)}자")
    return {"context": context, "sources": sources, "rag_documents": rag_documents, "logs": logs}


# ---------------------------------------------------------------------------
# Node 10. rag_quality_grader_node : RAG 근거 충분성 판단 (설계서 7번 섹션)
# ---------------------------------------------------------------------------
EVIDENCE_GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 RAG 시스템의 근거 평가자입니다. 아래 [참고 문서]가 [사용자 질문]에 답하는 데 "
            "실질적으로 도움이 되는 내용을 포함하고 있는지 판단하세요.\n"
            "판단 기준:\n"
            "- 질문의 모든 세부사항을 다루지 않아도, 질문의 핵심 증상/주제에 관련된 설명이나 "
            "해결 방법이 문서에 있다면 SUFFICIENT로 판단하세요.\n"
            "- 문서가 질문과 사실상 관련이 없거나 완전히 다른 주제를 다룰 때만 INSUFFICIENT로 "
            "판단하세요.\n"
            "SUFFICIENT 또는 INSUFFICIENT 중 한 단어만 출력하고 다른 설명은 하지 마세요.",
        ),
        ("human", "[사용자 질문]\n{question}\n\n[참고 문서]\n{context}"),
    ]
)


def rag_quality_grader_node(state: AgentState) -> dict:
    rag_documents = state.get("rag_documents", [])
    context = state.get("context", "")
    query = state["query"]

    if not rag_documents:
        logs = _log(state, "[rag_quality_grader] 검색된 문서 없음 -> INSUFFICIENT")
        return {"rag_sufficient": False, "rag_score": 0.0, "logs": logs}

    top = rag_documents[0]["score"]

    # Reranking 점수가 이미 충분히 높으면(config.RAG_AUTO_SUFFICIENT_SCORE) LLM 판정을
    # 생략하고 SUFFICIENT로 처리합니다. 로컬 소형 모델이 명백히 관련성 높은 문서도 이따금
    # INSUFFICIENT로 잘못 판정해 불필요하게 Web Search로 빠지는 문제를 줄이기 위함입니다.
    if top >= config.RAG_AUTO_SUFFICIENT_SCORE:
        logs = _log(
            state,
            f"[rag_quality_grader] top_score={top:.4f} >= "
            f"{config.RAG_AUTO_SUFFICIENT_SCORE} -> 자동 SUFFICIENT (LLM 판정 생략)",
        )
        return {"rag_sufficient": True, "rag_score": top, "logs": logs}

    chain = EVIDENCE_GRADER_PROMPT | get_llm()
    response = chain.invoke({"question": query, "context": context})
    verdict = response.content.strip().upper()
    sufficient = verdict.startswith("SUFFICIENT")

    logs = _log(
        state,
        f"[rag_quality_grader] top_score={top:.4f}, evidence="
        f"{'SUFFICIENT' if sufficient else 'INSUFFICIENT'}",
    )
    return {"rag_sufficient": sufficient, "rag_score": top, "logs": logs}


def route_after_grader(state: AgentState) -> str:
    """설계서 7번 섹션 Fallback 판단 기준 그대로 구현."""
    if not state.get("rag_documents"):
        return "web_search_agent"
    if state.get("rag_score", 0.0) < config.RERANK_THRESHOLD:
        return "web_search_agent"
    if state.get("rag_sufficient") is False:
        return "web_search_agent"
    return "generate_answer"


# ---------------------------------------------------------------------------
# Node 11. web_search_agent_node : Web Search Tool 호출 (설계서 2번 섹션 Agent 2)
# ---------------------------------------------------------------------------
def web_search_agent_node(state: AgentState) -> dict:
    query = state["query"]
    result = tools.web_search.invoke({"query": query})

    web_results = result.get("results", [])
    logs = _log(state, f"[web_search_agent] 웹 검색 결과 {len(web_results)}건")
    if result.get("error"):
        logs = _log(state, f"[web_search_agent] 경고: {result['error']}")

    web_context = result.get("context", "")
    if web_context:
        context = web_context
        sources = result.get("sources", [])
    else:
        # 웹 검색이 실패했거나(네트워크 차단 등) 결과가 없으면, RAG 단계에서 이미 찾아둔
        # 내부 문서 Context가 있을 경우 그대로 유지합니다(기준 점수 미달/LLM 판정
        # INSUFFICIENT로 여기까지 왔더라도, 완전히 빈 답변보다는 낫습니다).
        # 내부 문서도 전혀 없었다면 빈 Context 그대로 유지되어 generate_answer가
        # "정보를 찾지 못했다"고 안내합니다.
        context = state.get("context", "")
        sources = state.get("sources", [])
        if context:
            logs = _log(state, "[web_search_agent] 웹 검색 결과 없음 -> 기존 내부 문서 Context 유지")

    return {
        "web_results": web_results,
        "context": context,
        "sources": sources,
        "used_web_search": True,
        "logs": logs,
    }


# ---------------------------------------------------------------------------
# Node 12. generate_answer_node : Context 기반 최종 답변 생성
# ---------------------------------------------------------------------------
ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 세탁기/건조기/TV 전자제품 고객지원 상담원입니다. "
            "아래 [참고 문서]만 근거로 사용자 질문에 답변하세요.\n"
            "규칙:\n"
            "1. 참고 문서에 있는 내용만 사용하세요.\n"
            "2. 문서에 없는 내용을 추측하지 마세요.\n"
            "3. 오류 코드, 수치, 절차는 정확하게 유지하세요.\n"
            "4. 친절하고 간결한 한국어로 답변하세요.",
        ),
        ("human", "[사용자 질문]\n{question}\n\n[참고 문서]\n{context}"),
    ]
)


def generate_answer_node(state: AgentState) -> dict:
    query = state["query"]
    context = state.get("context", "")

    chain = ANSWER_PROMPT | get_llm()
    response = chain.invoke({"question": query, "context": context})
    answer = response.content.strip()

    messages = list(state.get("messages", [])) + [{"role": "assistant", "content": answer}]
    logs = _log(state, "[generate_answer] 최종 답변 생성")

    return {"answer": answer, "messages": messages, "logs": logs}


# ---------------------------------------------------------------------------
# Node 13. unsupported_answer_node : 지원하지 않는 제품 안내 (설계서 17번 섹션)
# ---------------------------------------------------------------------------
def unsupported_answer_node(state: AgentState) -> dict:
    answer = (
        "현재 이 고객지원 AI는 세탁기, 건조기, TV 3개 제품만 내부 고객지원 대상으로 지원합니다. "
        "문의하신 내용은 현재 지원 범위에 포함되지 않습니다."
    )
    messages = list(state.get("messages", [])) + [{"role": "assistant", "content": answer}]
    logs = _log(state, "[unsupported_answer] 지원하지 않는 제품 -> 안내 메시지 반환")

    return {"answer": answer, "route": "unsupported", "messages": messages, "logs": logs}


# ---------------------------------------------------------------------------
# LangGraph 구성 (설계서 18번 섹션 최종 LangGraph 설계 그대로)
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("analyze_query", analyze_query_node)
    graph.add_node("basic_query", basic_query_node)
    graph.add_node("multi_query", multi_query_node)
    graph.add_node("hyde", hyde_node)
    graph.add_node("llamaindex_retrieval", llamaindex_retrieval_node)
    graph.add_node("similarity_filter", similarity_filter_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("build_context", build_context_node)
    graph.add_node("rag_quality_grader", rag_quality_grader_node)
    graph.add_node("web_search_agent", web_search_agent_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("unsupported_answer", unsupported_answer_node)

    graph.add_edge(START, "analyze_query")

    # Initial Router
    graph.add_conditional_edges(
        "analyze_query",
        route_after_analyze,
        {
            "unsupported_answer": "unsupported_answer",
            "web_search_agent": "web_search_agent",
            "basic_query": "basic_query",
            "multi_query": "multi_query",
            "hyde": "hyde",
        },
    )

    # hybrid 전략은 multi_query_node를 거친 뒤 hyde_node로 이어집니다.
    graph.add_conditional_edges(
        "multi_query",
        route_after_multi_query,
        {
            "hyde": "hyde",
            "llamaindex_retrieval": "llamaindex_retrieval",
        },
    )

    graph.add_edge("basic_query", "llamaindex_retrieval")
    graph.add_edge("hyde", "llamaindex_retrieval")

    graph.add_edge("llamaindex_retrieval", "similarity_filter")
    graph.add_edge("similarity_filter", "rerank")
    graph.add_edge("rerank", "build_context")
    graph.add_edge("build_context", "rag_quality_grader")

    # RAG Quality Grading -> Web Search Fallback
    graph.add_conditional_edges(
        "rag_quality_grader",
        route_after_grader,
        {
            "web_search_agent": "web_search_agent",
            "generate_answer": "generate_answer",
        },
    )

    graph.add_edge("web_search_agent", "generate_answer")
    graph.add_edge("generate_answer", END)
    graph.add_edge("unsupported_answer", END)

    return graph.compile()
