"""
config.py

전자제품 고객지원 AI Agent 프로젝트의 공통 설정값 모음.
electronics_customer_support_ai_agent_design.md 설계서를 기준으로 작성.

이후 단계(4. 멀티 에이전트, 6. FastAPI/Streamlit)에서도 이 파일을 그대로
import 하여 사용합니다. 값 변경이 필요하면 이 파일만 수정하면 됩니다.

6번 단계부터 이 파일은 backend/ 폴더 안에 위치합니다. data/, chroma_db/ 는
설계서 16번 섹션 폴더 구조대로 backend/ 의 부모 폴더(프로젝트 루트)에 그대로
두므로, BASE_DIR은 이 파일 기준 한 단계 위(프로젝트 루트)를 가리킵니다.
"""
import os

# ── 경로 설정 ────────────────────────────────────────────────
# 프로젝트 루트 = backend/ 의 부모 폴더 (config.py는 backend/config.py)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# -------------------------
# 지원 제품 / 문서 폴더
# -------------------------
# data/<product>/ 폴더명이 곧 product 메타데이터 값이 됩니다.
SUPPORTED_PRODUCTS = ["washer", "dryer", "tv"]

PRODUCT_DIRS = {
    product: os.path.join(DATA_DIR, product) for product in SUPPORTED_PRODUCTS
}

# 문서 파일명 규칙: "<product>_<category>.txt" (예: washer_error_code.txt)
# build_db.py 는 이 규칙으로 category 메타데이터를 자동 추출합니다.
# 규칙에 맞지 않는 파일은 category="general" 로 처리됩니다.
KNOWN_CATEGORIES = [
    "manual",
    "error_code",
    "troubleshooting",
    "installation",
    "cleaning",
    "faq",
]

# 제품명 한글 표시(라우팅 안내 메시지 등에서 사용)
PRODUCT_DISPLAY_NAME = {
    "washer": "세탁기",
    "dryer": "건조기",
    "tv": "TV",
}

# -------------------------
# Vector DB
# -------------------------
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "electronics_support_rag"

# -------------------------
# Chunk
# -------------------------
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# -------------------------
# Retrieval
# -------------------------
SIMILARITY_TOP_K = 5      # 검색 Query 1개당 Retriever가 가져오는 Top K
SIMILARITY_CUTOFF = 0.5   # 이 점수 미만 Node는 Similarity Filter에서 제거

# -------------------------
# Reranking
# -------------------------
RERANK_TOP_N = 3
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"   # 로컬 HuggingFace CrossEncoder (다국어)

# RAG Quality Grader가 Web Search로 Fallback 할지 판단할 때 사용하는
# Reranking 최상위 점수 임계값 (설계서 7번 섹션 참조)
RERANK_THRESHOLD = 0.4

# Reranking 최상위 점수가 이 값 이상이면, RAG Quality Grader의 LLM 판정을 거치지 않고
# 곧바로 SUFFICIENT로 처리합니다. 로컬 소형 모델(예: qwen2.5:14b)이 명백히 관련성 높은
# 문서(예: FAQ 문항과 질문이 거의 동일한 경우)도 이따금 INSUFFICIENT로 잘못 판정해
# 불필요하게 Web Search로 빠지는 경우가 있어, 점수가 이미 충분히 높으면 LLM 판정을
# 생략해 이런 오탐을 줄입니다.
RAG_AUTO_SUFFICIENT_SCORE = float(os.environ.get("RAG_AUTO_SUFFICIENT_SCORE", "0.6"))

# -------------------------
# Multi-Query
# -------------------------
MULTI_QUERY_COUNT = 3

# -------------------------
# Model (Ollama)
# -------------------------
LLM_MODEL = "qwen2.5:14b"         # 사용자 환경에 맞게 변경 (ollama pull qwen2.5:14b 필요)
EMBEDDING_MODEL = "bge-m3"        # 사용자 환경에 맞게 변경 (ollama pull bge-m3 필요)
OLLAMA_REQUEST_TIMEOUT = 180.0    # 초 단위, 로컬 LLM 응답이 느릴 수 있어 넉넉히 설정

# Ollama 서버 주소. 로컬 실행 시에는 기본값(localhost)을 그대로 사용하고,
# Docker 컨테이너 안에서는 호스트에 떠 있는 Ollama에 접근해야 하므로
# docker-compose.yml에서 OLLAMA_BASE_URL=http://host.docker.internal:11434 로
# 덮어씁니다(하드코딩 금지, 환경변수로 관리).
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# -------------------------
# RAGAS 평가 실행 설정 (evaluate_ragas.py 의 RunConfig)
# -------------------------
# ragas의 evaluate()는 기본적으로 timeout=180초, max_workers=16(동시 실행)으로 동작합니다.
# 로컬 Ollama 서버(특히 14b급 모델)는 요청을 사실상 순차 처리하는 경우가 많아,
# 16개 작업이 한꺼번에 몰리면 뒤쪽 작업들이 180초 안에 자기 차례가 오지 못해
# TimeoutError로 실패하고 해당 지표가 NaN으로 남는 문제가 생깁니다.
# -> 동시 실행 수를 낮추고(RAGAS_MAX_WORKERS) 타임아웃을 넉넉히(RAGAS_TIMEOUT) 주어
#    한 번에 하나씩(또는 소수씩) 순서대로 처리되도록 합니다.
# 서버 성능/GPU 상황에 맞게 환경변수로 조정 가능합니다.
RAGAS_MAX_WORKERS = int(os.environ.get("RAGAS_MAX_WORKERS", "2"))   # 동시 실행 작업 수
RAGAS_TIMEOUT = int(os.environ.get("RAGAS_TIMEOUT", "600"))          # 초 단위, 작업 1개당 최대 대기 시간
RAGAS_MAX_WAIT = int(os.environ.get("RAGAS_MAX_WAIT", "90"))         # 초 단위, 재시도 사이 최대 대기(지수 백오프 상한)
RAGAS_MAX_RETRIES = int(os.environ.get("RAGAS_MAX_RETRIES", "3"))    # 실패 시 재시도 횟수

# -------------------------
# Web Search (4번 이후 단계에서 사용)
# -------------------------
WEB_SEARCH_MAX_RESULTS = 5
# Tavily 등 사용 시 환경변수(TAVILY_API_KEY)로 관리 (하드코딩 금지)
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

# -------------------------
# API / SSE (6번 단계에서 사용)
# -------------------------
# 환경변수로 덮어쓸 수 있습니다 (같은 서버를 여러 명이 공유할 때 포트 충돌 회피 등).
#   예) BACKEND_PORT=8010 python main.py
#   예) BACKEND_PORT=8010 FRONTEND_PORT=8510 streamlit run app.py --server.port 8510
BACKEND_PORT = int(os.environ.get("BACKEND_PORT", "8000"))
FRONTEND_PORT = int(os.environ.get("FRONTEND_PORT", "8501"))

# -------------------------
# Debug
# -------------------------
SHOW_LOGS = True
SHOW_RETRIEVED_NODES = True
