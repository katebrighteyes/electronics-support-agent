"""
build_db.py
--------------------------------
전자제품 고객지원 AI Agent - Chroma Vector DB 생성 전용 스크립트.
(첨부된 AdvancedRAG_LlamaGraph_BuildDB.py 구조를 참고하여 washer/dryer/tv
3개 제품 문서를 하나의 ChromaDB에 Metadata로 구분하여 저장하도록 확장)

담당 기능:
  1) data/washer, data/dryer, data/tv TXT 문서 로딩 (SimpleDirectoryReader) -> Document
  2) Document별 Metadata 추가 (product / category / file_name)
     - product   : 폴더명 (washer / dryer / tv)
     - category  : 파일명 규칙 "<product>_<category>.txt" 에서 자동 추출
     - file_name : 원본 파일명
  3) Node 분할 (SentenceSplitter, config.CHUNK_SIZE / CHUNK_OVERLAP)
  4) Embedding 생성 (Ollama BGE-M3)
  5) Chroma Vector DB 저장 + VectorStoreIndex 생성

흐름:
  data/{washer,dryer,tv}/*.txt -> SimpleDirectoryReader -> Document -> Metadata 추가
  -> SentenceSplitter -> Node 생성 -> Embedding -> Chroma DB 저장
  -> VectorStoreIndex 구성

실행:
  python build_db.py

사전 준비:
  - Ollama 로컬 서버 실행 (ollama serve)
  - ollama pull bge-m3              (config.EMBEDDING_MODEL)
  - ollama pull qwen2.5:14b          (config.LLM_MODEL, 이 스크립트에서는 직접 호출하지
                                       않지만 Settings.llm 초기화를 위해 필요)
  - pip install llama-index llama-index-vector-stores-chroma
                llama-index-embeddings-ollama llama-index-llms-ollama chromadb
  - data/washer, data/dryer, data/tv 폴더에 각 제품 문서(txt)가 존재해야 함
"""
import chromadb
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

import config


# ---------------------------------------------------------------------------
# 공통 출력 helper
# ---------------------------------------------------------------------------
def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# LlamaIndex 전역 설정 (LLM / Embedding Model)
# ---------------------------------------------------------------------------
def setup_settings() -> None:
    Settings.embed_model = OllamaEmbedding(
        model_name=config.EMBEDDING_MODEL, base_url=config.OLLAMA_BASE_URL
    )
    Settings.llm = Ollama(
        model=config.LLM_MODEL,
        request_timeout=config.OLLAMA_REQUEST_TIMEOUT,
        base_url=config.OLLAMA_BASE_URL,
    )


# ---------------------------------------------------------------------------
# Metadata 추출 helper
# ---------------------------------------------------------------------------
def _infer_category(file_name: str, product: str) -> str:
    """
    파일명 규칙 "<product>_<category>.txt" 에서 category를 추출합니다.
    예) washer_error_code.txt (product=washer) -> "error_code"
    규칙에 맞지 않으면 "general" 로 처리합니다.
    """
    stem = file_name.rsplit(".", 1)[0]
    prefix = f"{product}_"
    if stem.startswith(prefix):
        category = stem[len(prefix):]
    else:
        category = "general"

    if category not in config.KNOWN_CATEGORIES:
        print(f"  [경고] '{file_name}' -> category '{category}' 가 KNOWN_CATEGORIES에 없습니다. "
              f"그대로 저장합니다.")
    return category


# ---------------------------------------------------------------------------
# Step 1. Document 로딩 + Metadata 추가 (product 별 폴더를 순회)
# ---------------------------------------------------------------------------
def step1_load_documents():
    print_header("Step 1. Document 로딩 + Metadata 추가 (product / category / file_name)")

    all_documents = []

    for product in config.SUPPORTED_PRODUCTS:
        product_dir = config.PRODUCT_DIRS[product]
        documents = SimpleDirectoryReader(input_dir=product_dir).load_data()

        for doc in documents:
            file_name = doc.metadata.get("file_name", "")
            doc.metadata["product"] = product
            doc.metadata["category"] = _infer_category(file_name, product)
            # file_name 은 SimpleDirectoryReader가 기본으로 채워주지만 명시적으로 유지
            doc.metadata["file_name"] = file_name

        print(f"\n[{product}] 로드된 Document 수: {len(documents)}")
        for doc in documents:
            print(f"  - file_name={doc.metadata.get('file_name')} "
                  f"category={doc.metadata.get('category')}")

        all_documents.extend(documents)

    print(f"\n전체 로드된 Document 수: {len(all_documents)}")
    return all_documents


# ---------------------------------------------------------------------------
# Step 2. Node 분할 (SentenceSplitter)
# ---------------------------------------------------------------------------
def step2_split_nodes(documents):
    print_header("Step 2. Node 분할 (SentenceSplitter)")

    splitter = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    nodes = splitter.get_nodes_from_documents(documents)

    print(f"chunk_size={config.CHUNK_SIZE}, chunk_overlap={config.CHUNK_OVERLAP}")
    print(f"생성된 Node 수: {len(nodes)}")

    for i, node in enumerate(nodes[:3]):
        preview = node.get_content()[:80].replace("\n", " ")
        print(f"\nNode {i}")
        print(f"product   : {node.metadata.get('product')}")
        print(f"file_name : {node.metadata.get('file_name')}")
        print(f"category  : {node.metadata.get('category')}")
        print(f"text[:80] : {preview}...")

    return nodes


# ---------------------------------------------------------------------------
# Step 3. Embedding 생성 + Chroma Vector DB 저장 + VectorStoreIndex 구성
# ---------------------------------------------------------------------------
def step3_build_index(nodes):
    print_header("Step 3. Embedding 생성 + Chroma 저장 + VectorStoreIndex 생성")

    chroma_client = chromadb.PersistentClient(path=config.CHROMA_PATH)

    # 기존 컬렉션이 있으면 삭제 후 재생성 (재실행 시 중복 방지)
    existing = [c.name for c in chroma_client.list_collections()]
    if config.COLLECTION_NAME in existing:
        chroma_client.delete_collection(config.COLLECTION_NAME)
        print(f"기존 컬렉션 '{config.COLLECTION_NAME}' 삭제 후 재생성합니다.")

    chroma_collection = chroma_client.create_collection(config.COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(nodes, storage_context=storage_context)

    print(f"Chroma DB 경로  : {config.CHROMA_PATH}")
    print(f"Collection 이름 : {config.COLLECTION_NAME}")
    print(f"저장된 Node 수  : {chroma_collection.count()}")
    print("\nVectorStoreIndex 생성 완료.")
    return index


# ---------------------------------------------------------------------------
# Step 4. 저장 결과 요약 (product / category 별 Node 개수)
# ---------------------------------------------------------------------------
def step4_summary(nodes):
    print_header("Step 4. product / category 별 Node 개수 요약")

    from collections import Counter

    product_counts = Counter(n.metadata.get("product") for n in nodes)
    category_counts = Counter(
        (n.metadata.get("product"), n.metadata.get("category")) for n in nodes
    )

    print("Product 별 Node 수:")
    for product in config.SUPPORTED_PRODUCTS:
        print(f"  {product:8s}: {product_counts.get(product, 0)}개")

    print("\nProduct + Category 별 Node 수:")
    for (product, category), count in sorted(category_counts.items()):
        print(f"  {product:8s} / {category:15s}: {count}개")


def main():
    setup_settings()
    documents = step1_load_documents()
    nodes = step2_split_nodes(documents)
    step3_build_index(nodes)
    step4_summary(nodes)

    print_header("완료")
    print("Chroma DB가 생성되었습니다. 이제 4번 단계(멀티 에이전트 코드)에서 "
          "이 DB를 로드하여 테스트할 수 있습니다.")


if __name__ == "__main__":
    main()
