"""
main.py
--------------------------------
6번 단계: FastAPI 백엔드 (설계서 15번 섹션).

이 파일은 REST/SSE 엔드포인트 정의만 담당합니다. 실제 에이전트 호출(LangGraph 실행)은
전부 agent_main.py 에 위임합니다 (agent_main.run_agent / agent_main.run_agent_stream).

엔드포인트:
  POST /chat         - 일반 답변 (JSON)
  POST /chat/stream   - SSE 스트리밍 답변 (event: status/token/source/done)
  GET  /health        - 상태 확인

실행:
  cd backend
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
  (또는 python main.py)
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

import agent_main
import config


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 기동 시 LlamaIndex/Chroma/LangGraph를 미리 초기화해 첫 요청 지연을 줄입니다.
    agent_main.get_app()
    yield


app = FastAPI(title="전자제품 고객지원 AI Agent API", lifespan=lifespan)

# 개발 환경 기준 CORS 설정 (frontend/app.py 가 다른 포트(8501)에서 호출)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    source: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "supported_products": config.SUPPORTED_PRODUCTS}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    result = agent_main.run_agent(request.message)
    return ChatResponse(answer=result["answer"], source=result["source"])


@app.post("/chat/stream")
def chat_stream(request: ChatRequest) -> EventSourceResponse:
    def event_generator():
        for event_name, data in agent_main.run_agent_stream(request.message):
            yield {"event": event_name, "data": json.dumps(data, ensure_ascii=False)}

    return EventSourceResponse(event_generator())


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=config.BACKEND_PORT, reload=False)
