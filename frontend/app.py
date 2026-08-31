"""
app.py
--------------------------------
6번 단계: Streamlit 프론트엔드 (설계서 13~15번 섹션).

FastAPI 백엔드(main.py)의 /chat, /chat/stream 을 호출합니다. 백엔드 URL/포트는
backend/config.py 의 BACKEND_PORT 값을 그대로 재사용합니다(포트 번호를 두 곳에
따로 관리하지 않기 위함).

실행:
  1) 백엔드 먼저 실행 (다른 터미널)
       cd backend && uvicorn main:app --reload --port 8000
  2) 프론트엔드 실행
       cd frontend && streamlit run app.py --server.port 8501
"""
from __future__ import annotations

import json
import os
import sys

import requests
import streamlit as st

# backend/config.py 재사용 (포트 번호 등 설정을 한 곳에서만 관리)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))
import config  # noqa: E402

BACKEND_URL = os.environ.get("API_URL") or f"http://localhost:{config.BACKEND_PORT}"
CHAT_ENDPOINT = f"{BACKEND_URL}/chat"
CHAT_STREAM_ENDPOINT = f"{BACKEND_URL}/chat/stream"
REQUEST_TIMEOUT = 180

st.set_page_config(page_title="전자제품 고객지원 AI Agent", page_icon="🛠️")

st.title("전자제품 고객지원 AI Agent")
st.caption("지원 제품: " + " · ".join(config.PRODUCT_DISPLAY_NAME[p] for p in config.SUPPORTED_PRODUCTS))

if "messages" not in st.session_state:
    # {"role": "user" | "assistant", "content": str, "sources": list[str]}
    st.session_state.messages = []


def render_message(role: str, content: str, sources: list[str]) -> None:
    with st.chat_message(role):
        st.write(content)
        if sources:
            with st.expander("📄 참고 문서"):
                for s in sources:
                    st.write(f"- {s}")


for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"], msg.get("sources", []))


# ---------------------------------------------------------------------------
# 백엔드 호출
# ---------------------------------------------------------------------------
def call_chat(message: str) -> dict:
    resp = requests.post(CHAT_ENDPOINT, json={"message": message}, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def parse_sse_lines(lines):
    """
    text/event-stream 라인들("event: ...", "data: ...", 빈 줄로 이벤트 구분)을
    (event_name, data_dict) 튜플로 하나씩 변환합니다. requests의 iter_lines() 결과를
    그대로 넣으면 됩니다.
    """
    event_name = None
    for raw_line in lines:
        if raw_line is None or raw_line == "":
            continue
        if raw_line.startswith("event:"):
            event_name = raw_line[len("event:"):].strip()
        elif raw_line.startswith("data:"):
            data_str = raw_line[len("data:"):].strip()
            try:
                data = json.loads(data_str) if data_str else {}
            except json.JSONDecodeError:
                data = {}
            yield event_name, data


def stream_chat(message: str, status_box, answer_box) -> tuple[str, list[str]]:
    """SSE 스트리밍 응답을 실시간으로 화면에 반영하고, 최종 (답변, 출처 목록)을 반환합니다."""
    answer_text = ""
    sources: list[str] = []

    with requests.post(
        CHAT_STREAM_ENDPOINT, json={"message": message}, stream=True, timeout=REQUEST_TIMEOUT
    ) as resp:
        resp.raise_for_status()
        for event_name, data in parse_sse_lines(resp.iter_lines(decode_unicode=True)):
            if event_name == "status":
                status_box.update(label=data.get("message", ""), state="running")
            elif event_name == "token":
                answer_text += data.get("content", "")
                answer_box.markdown(answer_text)
            elif event_name == "source":
                src = data.get("source")
                if src and src not in sources:
                    sources.append(src)
            elif event_name == "done":
                status_box.update(label="답변 생성 완료", state="complete")

    return answer_text, sources


# ---------------------------------------------------------------------------
# 입력 영역 (질문 입력창 + 답변 방식 선택, 설계서 13번 섹션 GUI)
# ---------------------------------------------------------------------------
_, mode_col = st.columns([3, 1])
with mode_col:
    mode = st.selectbox("답변 방식", ["일반 답변", "SSE 스트리밍"], label_visibility="collapsed")

question = st.chat_input("질문을 입력하세요...")

if question:
    st.session_state.messages.append({"role": "user", "content": question, "sources": []})
    render_message("user", question, [])

    with st.chat_message("assistant"):
        if mode == "일반 답변":
            with st.spinner("답변을 생성하고 있습니다..."):
                try:
                    result = call_chat(question)
                except requests.RequestException as e:
                    st.error(f"백엔드 호출에 실패했습니다: {e}")
                else:
                    answer = result.get("answer", "")
                    sources = [s.strip() for s in (result.get("source") or "").split(",") if s.strip()]
                    st.write(answer)
                    if sources:
                        with st.expander("📄 참고 문서"):
                            for s in sources:
                                st.write(f"- {s}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
        else:
            status_box = st.status("질문을 분석하고 있습니다.", expanded=True)
            answer_box = st.empty()
            try:
                answer, sources = stream_chat(question, status_box, answer_box)
            except requests.RequestException as e:
                status_box.update(label="오류 발생", state="error")
                st.error(f"백엔드 호출에 실패했습니다: {e}")
            else:
                if sources:
                    with st.expander("📄 참고 문서"):
                        for s in sources:
                            st.write(f"- {s}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
