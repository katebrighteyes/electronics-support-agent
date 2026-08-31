# 전자제품 고객지원 AI Agent

세탁기 / 건조기 / TV 3개 제품을 지원하는 LangGraph 기반 멀티 에이전트 고객지원 챗봇입니다.
Advanced RAG(LlamaIndex + ChromaDB) + RAG Quality Grader + Web Search Fallback 구조로 동작하며,
FastAPI 백엔드 + Streamlit 프론트엔드로 서비스되고, Docker / GitHub Actions로 H200 서버에 배포됩니다.

이 문서는 아래 5가지를 순서대로 진행하는 방법을 정리한 것입니다.

1. [Chroma DB 빌드 (build_db.py)](#1-chroma-db-빌드-build_dbpy)
2. [에이전트 단독 테스트 (agent_main.py)](#2-에이전트-단독-테스트-agent_mainpy)
3. [RAGAS 정량 평가 (evaluate_ragas.py)](#3-ragas-정량-평가-evaluate_ragaspy)
4. [백엔드 / 프론트엔드 실행 (포트 옵션 포함)](#4-백엔드--프론트엔드-실행-포트-옵션-포함)
5. [CI/CD (GitHub Actions → H200 배포)](#5-cicd-github-actions--h200-배포)

---

## 0. 사전 준비

```bash
# 프로젝트 루트에서 (backend/, frontend/, data/ 와 같은 위치)
pip install -r requirements.txt
```

- Ollama 로컬 서버 실행 및 모델 pull (필수)

  ```bash
  ollama serve
  ollama pull qwen2.5:14b   # config.LLM_MODEL
  ollama pull bge-m3        # config.EMBEDDING_MODEL
  ```

- 이후 모든 파이썬 스크립트는 **`backend/` 폴더 안에서** 실행합니다 (`config.py`, `main.py` 등이 모두 이 폴더에 있음).

  ```bash
  cd backend
  ```

- Ollama가 `localhost:11434`가 아닌 다른 주소에 떠 있다면 환경변수로 지정할 수 있습니다.

  ```bash
  export OLLAMA_BASE_URL=http://내주소:11434
  ```

---

## 1. Chroma DB 빌드 (build_db.py)

`data/washer`, `data/dryer`, `data/tv` 의 문서를 읽어 Chroma Vector DB(`../chroma_db`)를 생성합니다.
**2~4번을 실행하기 전에 반드시 한 번은 먼저 실행해야 합니다.**

```bash
cd backend
python build_db.py
```

- 실행하면 Step 1(문서 로딩+메타데이터) → Step 2(청크 분할) → Step 3(임베딩+Chroma 저장) →
  Step 4(product/category별 개수 요약) 순서로 로그가 출력됩니다.
- 이미 DB가 있어도 다시 실행하면 기존 컬렉션을 삭제하고 새로 만듭니다(문서를 수정한 뒤 재실행해도 안전).
- 생성 위치는 `config.CHROMA_PATH` (프로젝트 루트의 `chroma_db/` 폴더)입니다.

---

## 2. 에이전트 단독 테스트 (agent_main.py)

FastAPI 없이, LangGraph 멀티 에이전트만 CLI로 바로 테스트할 수 있습니다.

```bash
cd backend

# 방법 A) ragas_qa_dataset.json 10개 질문을 while 루프로 순회 (기본)
python agent_main.py

# 방법 B) 위 10개 + 최신정보/미문서화 오류코드 케이스 2개 추가 (총 12개)
python agent_main.py --extra

# 방법 C) 질문 1개만 즉석 실행
python agent_main.py 세탁기가 탈수가 안돼요
```

- 방법 A/B는 각 질문마다 `[분석]`(product/category) → `[라우팅]`(expected vs actual) →
  `[출처]` → `[답변]` → `[소요시간]`을 출력하고, 마지막에 라우팅 일치 개수를 요약해줍니다.
- 방법 C는 단일 질문에 대한 product/category/route/answer/source만 바로 출력합니다.
- 이 단계에서 문제없이 답변이 나오면 3번(RAGAS), 4번(백엔드/프론트엔드) 모두 정상 동작할 가능성이 높습니다 —
  가장 먼저 확인해보시길 권장합니다.

---

## 3. RAGAS 정량 평가 (evaluate_ragas.py)

`agent_main.run_agent()`를 호출해 10개 질문의 답변/근거문서를 수집한 뒤, RAGAS 5개 지표
(faithfulness / answer_relevancy / context_precision / context_recall / answer_correctness)로 평가합니다.

```bash
cd backend
python evaluate_ragas.py
```

- 결과는 터미널에 지표별 평균 점수 표 + 질문별 상세 표로 출력되고,
  `ragas_eval_results.csv` / `ragas_eval_results.xlsx` 로도 저장됩니다.
- **로컬 Ollama는 요청을 사실상 순차 처리**하기 때문에, 기본 동시성으로는 다수 작업이
  타임아웃날 수 있어 `config.py`의 아래 값으로 동시 실행 수를 낮추고 대기 시간을 늘려뒀습니다.

  | 환경변수 | 기본값 | 의미 |
  |---|---|---|
  | `RAGAS_MAX_WORKERS` | 2 | 동시에 실행하는 평가 작업 수 |
  | `RAGAS_TIMEOUT` | 600 | 작업 1개당 최대 대기 시간(초) |
  | `RAGAS_MAX_WAIT` | 90 | 재시도 사이 최대 대기(초) |
  | `RAGAS_MAX_RETRIES` | 3 | 실패 시 재시도 횟수 |

  그래도 타임아웃이 나면 서버 성능에 맞게 조정해서 재실행하세요.

  ```bash
  RAGAS_MAX_WORKERS=1 RAGAS_TIMEOUT=900 python evaluate_ragas.py
  ```

- `ragas<0.4` 가 필요합니다(0.4.x는 `langchain_community.chat_models.vertexai` 관련 알려진 호환성
  버그가 있음). `requirements.txt`에 이미 `ragas<0.4`로 고정되어 있으니 별도 조치는 필요 없습니다.
  설치 내용 더 필요.

---

## 4. 백엔드 / 프론트엔드 실행 (포트 옵션 포함)

### 4-1. 백엔드 (FastAPI, 기본 포트 8000)

```bash
cd backend

# 기본 포트(8000)로 실행
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 포트를 바꾸고 싶을 때 (예: 8010) - uvicorn CLI 옵션으로 바로 지정
uvicorn main:app --reload --host 0.0.0.0 --port 8010

# 또는 python main.py 로 실행할 경우, 환경변수 BACKEND_PORT로 지정
BACKEND_PORT=8010 python main.py
```

- `GET /health` : 상태 확인
- `POST /chat` : 일반 답변 (JSON `{ "message": "..." }`)
- `POST /chat/stream` : SSE 스트리밍 답변 (`event: status/token/source/done`)

### 4-2. 프론트엔드 (Streamlit, 기본 포트 8501)

```bash
cd frontend

# 기본 포트(8501)로 실행
streamlit run app.py --server.port 8501

# 포트를 바꾸고 싶을 때 (예: 8510) - streamlit CLI 옵션으로 바로 지정
streamlit run app.py --server.port 8510
```

- 프론트엔드는 기본적으로 `http://localhost:{backend 포트}` 로 백엔드를 호출합니다
  (`backend/config.py`의 `BACKEND_PORT`를 그대로 재사용).
- 백엔드를 8000이 아닌 다른 포트로 띄웠다면, 프론트엔드 실행 전에 **환경변수 두 가지 모두** 맞춰주세요.

  ```bash
  # 예: 백엔드를 8010으로 띄운 경우
  BACKEND_PORT=8010 python main.py                 # (backend/ 에서)
  API_URL=http://localhost:8010 streamlit run app.py --server.port 8510   # (frontend/ 에서)
  ```

  `API_URL`을 지정하면 프론트엔드가 `BACKEND_PORT` 계산 대신 이 주소를 그대로 사용합니다
  (Docker Compose에서 `API_URL=http://backend:8000` 으로 서비스명 DNS를 쓰는 것과 같은 방식입니다).

### 4-3. Docker Compose로 한 번에 실행 (로컬에서 배포 전 테스트할 때)

```bash
# 프로젝트 루트에서, .env 파일에 아래 값들을 미리 준비해두고 실행
#   COMPOSE_PROJECT_NAME=electronics-agent
#   BACKEND_PORT=8000
#   FRONTEND_PORT=8501
#   NVIDIA_VISIBLE_DEVICES=<본인에게 할당된 MIG UUID 또는 all>
docker compose up -d --build

# 컨테이너 상태 확인
docker compose ps

# 로그 확인 (백엔드가 build_db.py를 먼저 실행하므로 초기 기동에 시간이 걸릴 수 있습니다)
docker compose logs -f backend
```

- 이 방식은 `.env`의 `BACKEND_PORT`/`FRONTEND_PORT` 값이 곧 호스트에 노출되는 포트입니다
  (컨테이너 내부 포트는 8000/8501로 고정, `docker-compose.yml`이 매핑을 담당).
- 백엔드는 시작 시 `build_db.py`를 자동 실행해 Chroma DB를 새로 만들기 때문에, **Ollama가 호스트에서
  미리 떠 있고 `bge-m3`/`qwen2.5:14b`가 pull되어 있어야** 정상 기동합니다.

---

## 5. CI/CD (GitHub Actions → H200 배포)

`.github/workflows/deploy.yml` 이 정의하는 흐름입니다.

```
PC에서 main 브랜치에 commit/push
  → GitHub Actions "test" job (GitHub-hosted runner: 의존성 설치 + 문법 검사)
  → GitHub Actions "deploy" job (PC의 self-hosted runner: h200-deploy 라벨)
      1) ssh h200 접속 확인
      2) 레포 전체를 tar로 압축해 H200으로 scp 전송
      3) H200에서: 기존 컨테이너 제거 → 배포 디렉터리(~/llm/electronics-support-agent) 초기화
         → 압축 해제 → 학생별 .env(~/llm/.h200-deploy/.env) 복사 → docker compose up -d --build
      4) docker compose ps 로 기동 결과 확인
```

### 5-1. 사전 준비 (최초 1회)

- 배포용 PC에 self-hosted GitHub Actions 러너를 등록하고, 라벨에 `h200-deploy`가 포함되어 있는지 확인
- 그 PC의 `~/.ssh/config`에 H200 서버 접속용 `h200` 이라는 Host 별칭이 설정되어 있어야 함 (`ssh h200`으로 접속 가능해야 함)
- H200 서버의 `~/llm/.h200-deploy/.env` 파일에 아래 값들이 준비되어 있어야 함 (git에는 올라가지 않음)

  ```
  COMPOSE_PROJECT_NAME=electronics-agent
  BACKEND_PORT=8001        # 다른 학생과 겹치지 않는 값으로
  FRONTEND_PORT=8501       # 다른 학생과 겹치지 않는 값으로
  NVIDIA_VISIBLE_DEVICES=MIG-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx   # 본인에게 할당된 MIG 조각
  ```

- H200 서버에도 Ollama가 떠 있고 `qwen2.5:14b`/`bge-m3`가 pull되어 있어야 함 (배포된 컨테이너가 접속해서 씀)

### 5-2. 배포 트리거

- `main` 브랜치에 push 하면 자동으로 실행됩니다.
- GitHub 저장소의 **Actions 탭 → AI Service CI/CD → Run workflow** 로 수동 실행(`workflow_dispatch`)도 가능합니다.

### 5-3. 배포 확인

- GitHub Actions 실행 로그의 마지막 "Check containers" 단계에서 `docker compose ps` 출력으로 확인
- 또는 H200에 직접 접속해서 확인

  ```bash
  ssh h200
  cd ~/llm/electronics-support-agent
  docker compose ps
  docker compose logs -f backend
  ```

- 컨테이너/이미지 이름은 `electronics-agent-backend` / `electronics-agent-frontend` 로 고정되어 있습니다.
  같은 H200을 쓰는 다른 학생의 배포와 이름이 겹치면 `docker-compose.yml`의 `image:`/`container_name:` 과
  `deploy.yml`의 해당 이름을 원하는 접두어로 바꿔서 재배포하세요.
