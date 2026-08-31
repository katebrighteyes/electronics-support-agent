"""
rag.py
--------------------------------
설계서 4번 섹션의 "LlamaIndex 담당" 영역 구현.
  Metadata Filter / Retriever / Similarity Filter / Reranking

Multi-Query / HyDE / Query 생성, Routing, State 관리는 LangGraph(graph.py)가 담당합니다.
(첨부된 AdvancedRAG_LlamaGraph.py 의 LlamaIndex 관련 함수를 참고하여,
 category/region 대신 product/category 메타데이터로 필터링하도록 재작성)
"""
from __future__ import annotations

import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank, SimilarityPostprocessor
from llama_index.core.vector_stores import FilterCondition, MetadataFilter, MetadataFilters
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

import config

_index: VectorStoreIndex | None = None


def setup_settings() -> None:
    """LlamaIndex 전역 설정(LLM / Embedding Model). 앱/스크립트 시작 시 1회 호출."""
    Settings.embed_model = OllamaEmbedding(
        model_name=config.EMBEDDING_MODEL, base_url=config.OLLAMA_BASE_URL
    )
    Settings.llm = Ollama(
        model=config.LLM_MODEL,
        request_timeout=config.OLLAMA_REQUEST_TIMEOUT,
        base_url=config.OLLAMA_BASE_URL,
    )


def get_index() -> VectorStoreIndex:
    """build_db.py 로 생성된 Chroma Collection을 VectorStoreIndex로 lazy 로딩(1회 캐싱)."""
    global _index
    if _index is None:
        chroma_client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        chroma_collection = chroma_client.get_collection(config.COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        _index = VectorStoreIndex.from_vector_store(vector_store)
    return _index


def build_metadata_filters(product: str | None, category: str | None) -> MetadataFilters | None:
    """
    product / category 로 LlamaIndex MetadataFilters 를 구성합니다. (설계서 3번 섹션)
    category가 "general"/"none"/빈 값이면 category 필터는 적용하지 않습니다.
    """
    filters = []
    if product and product in config.SUPPORTED_PRODUCTS:
        filters.append(MetadataFilter(key="product", value=product))
    if category and category not in ("general", "none", ""):
        filters.append(MetadataFilter(key="category", value=category))

    if not filters:
        return None
    return MetadataFilters(filters=filters, condition=FilterCondition.AND)


def retrieve(search_queries: list[str], product: str | None, category: str | None) -> list:
    """
    search_queries(원본 질문 + Multi-Query + HyDE 문서) 전체를 LlamaIndex Retriever로 검색합니다.
    category 필터로 검색 결과가 0건이면 product 필터만으로 재검색합니다(과도한 필터링 방지).
    """
    index = get_index()
    filters = build_metadata_filters(product, category)
    retriever = index.as_retriever(similarity_top_k=config.SIMILARITY_TOP_K, filters=filters)

    retrieved_nodes = []
    for q in search_queries:
        nodes = retriever.retrieve(q)
        if not nodes and category:
            fallback_filters = build_metadata_filters(product, None)
            fallback_retriever = index.as_retriever(
                similarity_top_k=config.SIMILARITY_TOP_K, filters=fallback_filters
            )
            nodes = fallback_retriever.retrieve(q)
        retrieved_nodes.extend(nodes)
    return retrieved_nodes


def similarity_filter(nodes: list) -> list:
    """중복 Node 제거(동일 node_id는 최고 score만 유지) + Similarity Cutoff 적용.

    Cutoff가 후보를 전부 걸러내더라도(Query 전략/paraphrase에 따라 임베딩 유사도가
    예상보다 낮게 나오는 경우가 있음), 원본 후보가 하나라도 있었다면 최상위 1개는
    남겨서 이후 Reranking/RAG Quality Grader가 최종 판단하도록 합니다. 하드 컷오프
    하나가 실제로는 관련 있는 문서까지 파이프라인 초반에 완전히 날려버리는 것을
    방지하기 위함이며, 진짜 관련 없는 문서라면 이후 RERANK_THRESHOLD나 Grader 판정
    단계에서 여전히 걸러집니다.
    """
    best_by_id = {}
    for n in nodes:
        node_id = n.node.node_id
        if node_id not in best_by_id or n.score > best_by_id[node_id].score:
            best_by_id[node_id] = n
    deduped = list(best_by_id.values())
    deduped.sort(key=lambda n: n.score, reverse=True)

    postprocessor = SimilarityPostprocessor(similarity_cutoff=config.SIMILARITY_CUTOFF)
    filtered = postprocessor.postprocess_nodes(deduped)
    filtered.sort(key=lambda n: n.score, reverse=True)

    if not filtered and deduped:
        filtered = deduped[:1]

    return filtered


def rerank(question: str, nodes: list) -> list:
    """Cross-Encoder Reranking (config.RERANK_MODEL) 후 상위 RERANK_TOP_N개만 반환."""
    if not nodes:
        return []
    reranker = SentenceTransformerRerank(model=config.RERANK_MODEL, top_n=config.RERANK_TOP_N)
    return reranker.postprocess_nodes(nodes, query_str=question)


def build_context(nodes: list) -> tuple[str, list[str]]:
    """Reranking 상위 Node들을 하나의 Context 문자열로 결합하고, 출처 file_name 목록을 반환."""
    parts = []
    sources: list[str] = []
    for n in nodes:
        file_name = n.node.metadata.get("file_name", "unknown")
        parts.append(f"[{file_name}]\n{n.node.get_content().strip()}")
        if file_name not in sources:
            sources.append(file_name)
    return "\n\n".join(parts), sources


def top_score(nodes: list) -> float:
    return float(nodes[0].score) if nodes else 0.0
