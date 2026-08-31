"""
agent_main.py
--------------------------------
6번 단계: 실제 멀티 에이전트(graph.py) 호출을 전담하는 모듈.

main.py(FastAPI)는 REST/SSE 엔드포인트 정의만 담당하고, 실제 에이전트 실행은
이 파일의 run_agent() / run_agent_stream() 을 호출합니다. evaluate_ragas.py도
동일하게 이 파일의 run_agent()를 호출합니다(그래프를 직접 만들지 않음).

이 파일은 다른 모듈에 의존하지 않고도 단독으로 실행할 수 있습니다.

실행:
  python agent_main.py                       # ragas_qa_dataset.json 10개 질문 while 루프 테스트
  python agent_main.py --extra                # 위 10개 + 최신정보/미문서화 오류코드 2문항 추가
  python agent_main.py 세탁기가 탈수가 안돼요   # 질문 1개만 즉석 실행

사전 준비:
  - build_db.py 로 Chroma DB(config.CHROMA_PATH) 생성 완료
  - Ollama 로컬 서버 실행, qwen2.5:14b / bge-m3 pull 완료
"""
from __future__ import annotations

import json
import sys
import time

import config
import rag
from graph import build_graph, new_state

# ---------------------------------------------------------------------------
# 그래프/설정 lazy 초기화 (FastAPI 등에서 반복 호출 시 재사용하기 위한 모듈 singleton)
# ---------------------------------------------------------------------------
_app = None
_settings_ready = False


def get_app():
    """LlamaIndex 설정 + 컴파일된 LangGraph를 최초 1회만 초기화하고 이후 재사용합니다."""
    global _app, _settings_ready
    if not _settings_ready:
        rag.setup_settings()
        _settings_ready = True
    if _app is None:
        _app = build_graph()
    return _app


# ---------------------------------------------------------------------------
# 일반 답변 (POST /chat 이 사용)
# ---------------------------------------------------------------------------
def run_agent(question: str) -> dict:
    """
    하나의 질문에 대해 멀티 에이전트 그래프를 끝까지 실행하고, main.py / evaluate_ragas.py /
    main_test.py 등에서 바로 쓸 수 있는 형태의 결과 dict를 반환합니다.
    """
    app = get_app()
    result = app.invoke(new_state(question))

    route = "unsupported" if result.get("route") == "unsupported" else (
        "web_search" if result.get("used_web_search") else "product_rag"
    )
    sources = result.get("sources") or []

    return {
        "answer": result.get("answer", ""),
        "source": ", ".join(sources),   # 설계서 15번 섹션 응답 형식과 동일한 키(문자열)
        "sources": sources,             # 프론트엔드/평가 스크립트가 쓰기 편하도록 리스트도 함께 제공
        "route": route,
        "product": result.get("product", ""),
        "category": result.get("category", ""),
        "rag_documents": result.get("rag_documents", []),
        "web_results": result.get("web_results", []),
        "context": result.get("context", ""),
    }


# ---------------------------------------------------------------------------
# SSE 스트리밍 답변 (POST /chat/stream 이 사용, 설계서 11~12번 섹션)
# ---------------------------------------------------------------------------
def _status_message(node_name: str, final_result: dict) -> str:
    product = final_result.get("product", "")
    product_kr = config.PRODUCT_DISPLAY_NAME.get(product, "제품")

    if node_name == "web_search_agent":
        # rag_quality_grader를 이미 거쳐온 경우 = RAG 근거 부족 -> Fallback
        if "rag_sufficient" in final_result:
            return "내부 문서에서 충분한 정보를 찾지 못해 웹 검색을 수행합니다."
        return "최신 정보 확인을 위해 웹 검색을 수행합니다."

    messages = {
        "analyze_query": "질문을 분석하고 있습니다.",
        "basic_query": "검색 질의를 준비하고 있습니다.",
        "multi_query": "다양한 표현으로 검색 질의를 확장하고 있습니다.",
        "hyde": "가상 문서를 생성해 검색 질의를 보강하고 있습니다.",
        "llamaindex_retrieval": f"{product_kr} 고객지원 문서를 검색하고 있습니다.",
        "similarity_filter": "검색 결과의 유사도를 확인하고 있습니다.",
        "rerank": "검색 결과의 관련도를 재정렬하고 있습니다.",
        "build_context": "참고 문서를 정리하고 있습니다.",
        "rag_quality_grader": "검색 문서의 관련도를 평가하고 있습니다.",
        "generate_answer": "답변을 생성하고 있습니다.",
        "unsupported_answer": "지원 범위를 확인하고 있습니다.",
    }
    return messages.get(node_name, f"{node_name} 처리 중입니다.")


def _chunk_answer(answer: str, words_per_chunk: int = 2):
    """
    생성된 답변을 SSE token 이벤트로 잘라 보내기 위한 청크 분할입니다.
    주의: generate_answer_node가 LangGraph 노드 단위(한 번의 완성된 LLM 호출)로 동작하기
    때문에, 이는 LLM을 실시간으로 토큰 스트리밍하는 것이 아니라 이미 생성이 끝난 답변
    문자열을 SSE 프로토콜에 맞춰 점진적으로 전송하는 방식입니다. 실제 LLM 토큰 스트리밍이
    필요하다면 generate_answer_node를 우회해 LLM의 stream() API를 이 함수 위치에서
    직접 호출하도록 확장하면 됩니다.
    """
    if not answer:
        return
    words = answer.split(" ")
    for i in range(0, len(words), words_per_chunk):
        chunk = " ".join(words[i : i + words_per_chunk])
        if i + words_per_chunk < len(words):
            chunk += " "
        yield chunk


def run_agent_stream(question: str):
    """
    질문 1개를 실행하며 (event_name, data) 튜플을 순서대로 yield 합니다.
    event_name 은 설계서 12번 섹션의 status/token/source/done 4종입니다.
    """
    app = get_app()
    state = new_state(question)
    final_result: dict = {}

    for update in app.stream(state, stream_mode="updates"):
        for node_name, node_output in update.items():
            final_result.update(node_output)
            yield ("status", {"message": _status_message(node_name, final_result)})

    answer = final_result.get("answer", "")
    for chunk in _chunk_answer(answer):
        yield ("token", {"content": chunk})

    for source in final_result.get("sources") or []:
        yield ("source", {"source": source})

    yield ("done", {})


# ---------------------------------------------------------------------------
# 단독 실행용 main() : 10개 질문 while 루프 테스트 (+ --extra) / 단일 질문 실행
# ---------------------------------------------------------------------------
QA_PATH = "ragas_qa_dataset.json"

# 설계서 6번 섹션 사례 C(최신정보 직행) / D(내부 문서 근거 부족 -> Web Search Fallback,
# 건조기 문서에는 의도적으로 E90 코드를 넣지 않았음) 확인용 선택 실행 문항
EXTRA_TEST_QUESTIONS = [
    {"id": "extra_latest_tv", "expected_route": "web_search", "question": "이 TV 모델 최신 펌웨어 문제가 있나요?"},
    {"id": "extra_dryer_unknown_error", "expected_route": "web_search", "question": "건조기에서 E90이라는 오류가 나요."},
]


def _run_single_question(question: str) -> None:
    result = run_agent(question)
    print(f"\n[product] {result['product']} / [category] {result['category']} / [route] {result['route']}")
    print(f"[answer]\n{result['answer']}")
    print(f"[source] {result['source'] or '(없음)'}")


def _run_qa_dataset(extra: bool = False) -> None:
    """ragas_qa_dataset.json 10개 질문(+선택적 --extra 2문항)을 while 문으로 순회하며
    실제 라우팅 결과를 expected_route와 비교합니다. (4번 단계 main_test.py 로직을 그대로 흡수)"""
    with open(QA_PATH, encoding="utf-8") as f:
        items = json.load(f)["items"]

    if extra:
        items = items + EXTRA_TEST_QUESTIONS

    results = []
    i = 0
    total = len(items)
    while i < total:
        item = items[i]
        question = item["question"]
        print("\n" + "#" * 70)
        print(f"# [{item['id']}] {question}")
        print("#" * 70)

        start = time.time()
        result = run_agent(question)
        elapsed = time.time() - start

        expected_route = item.get("expected_route", "")
        route_ok = (expected_route == "") or (result["route"] == expected_route)

        print(f"\n[분석] product={result['product']} category={result['category']}")
        print(
            f"[라우팅] expected={expected_route or '-'} / actual={result['route']} "
            f"-> {'OK' if route_ok else 'MISMATCH'}"
        )
        print(f"[출처] {result['sources'] or '(없음)'}")
        print(f"[답변]\n{result['answer']}")
        print(f"[소요시간] {elapsed:.2f}s")

        results.append(
            {
                "id": item["id"],
                "expected_route": expected_route,
                "actual_route": result["route"],
                "route_ok": route_ok,
                "elapsed": elapsed,
            }
        )
        i += 1

    print("\n" + "#" * 70)
    print("# 테스트 결과 요약")
    print("#" * 70)
    ok_count = 0
    for r in results:
        status = "OK" if r["route_ok"] else "MISMATCH"
        if r["route_ok"]:
            ok_count += 1
        print(
            f"  [{status:8s}] {r['id']:22s} expected={(r['expected_route'] or '-'):12s} "
            f"actual={r['actual_route']:12s} ({r['elapsed']:.2f}s)"
        )
    print(f"\n라우팅 일치: {ok_count}/{len(results)}")


def main() -> None:
    args = sys.argv[1:]
    if args and not args[0].startswith("--"):
        _run_single_question(" ".join(args))
        return

    _run_qa_dataset(extra="--extra" in args)


if __name__ == "__main__":
    main()
