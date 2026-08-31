"""
evaluate_ragas.py
--------------------------------
5번 단계: RAGAS를 이용한 정량 평가.

ragas_qa_dataset.json 의 10개 질문에 대해 agent_main.run_agent() 를 호출하여
answer/context를 수집한 뒤, RAGAS 지표로 평가하고 결과를 터미널 표 + CSV/Excel
파일로 저장합니다. (6번 단계부터는 그래프를 이 파일이 직접 만들지 않고, main.py와
동일하게 agent_main.py 를 통해서만 에이전트를 호출합니다)

평가지표 (설계서에 특정 지표가 명시되어 있지 않아, RAG 품질 평가에서 가장 널리
쓰이는 5개 표준 지표를 사용합니다. 10개 질문 모두 ground_truth가 있어 계산 가능):
  - faithfulness       : 답변이 검색된 문서(context)에 근거하는 정도 (환각 여부)
  - answer_relevancy    : 답변이 질문과 얼마나 관련 있는지
  - context_precision    : 검색된 문서 중 실제로 정답에 필요한 문서의 비율(순서 고려)
  - context_recall       : 정답(ground_truth)에 필요한 정보가 검색된 문서에 얼마나 포함됐는지
  - answer_correctness   : 정답과 비교한 답변의 사실적 정확성(의미 유사도 + 사실 일치)

미지원 제품 질문(unsupported_1)은 RAG 검색을 아예 수행하지 않으므로 context 관련
지표(context_precision/recall)는 낮게(혹은 0으로) 나오는 것이 정상입니다.

실행:
  python evaluate_ragas.py

사전 준비:
  - build_db.py 로 Chroma DB 생성 완료
  - Ollama 로컬 서버 실행, qwen2.5:14b / bge-m3 pull 완료
  - pip install "ragas<0.4" datasets pandas openpyxl langchain-ollama
    (ragas 0.4.x는 langchain-community의 ChatVertexAI 관련 알려진 호환성 버그가
     있어 0.4 미만 버전을 권장합니다. 자세한 내용은 아래 "ragas 임포트 실패 시"
     참고)

출력:
  - 터미널에 지표별 평균 점수 표 + 질문별 상세 표
  - ragas_eval_results.csv / ragas_eval_results.xlsx

TimeoutError / NaN 점수가 많이 나올 때:
  로컬 Ollama 서버는 요청을 사실상 순차 처리하는 경우가 많아, ragas의 기본 동시
  실행 설정(동시 16개, 작업당 180초 대기)으로는 뒤쪽 작업들이 제 시간에 처리되지
  못해 TimeoutError로 실패하고 해당 지표가 NaN으로 남을 수 있습니다.
  이 스크립트는 config.py 의 RAGAS_MAX_WORKERS(기본 2) / RAGAS_TIMEOUT(기본 600초)
  값으로 동시 실행 수를 낮추고 대기 시간을 늘려 이 문제를 완화합니다. 그래도 여전히
  타임아웃이 발생하면 환경변수로 더 조정하세요, 예:
    RAGAS_MAX_WORKERS=1 RAGAS_TIMEOUT=900 python evaluate_ragas.py
"""
from __future__ import annotations

import json
import sys
import time

import pandas as pd

import config
from agent_main import run_agent

QA_PATH = "ragas_qa_dataset.json"
CSV_OUTPUT_PATH = "ragas_eval_results.csv"
XLSX_OUTPUT_PATH = "ragas_eval_results.xlsx"

METRIC_LABELS = {
    "faithfulness": "충실성(Faithfulness)",
    "answer_relevancy": "답변 관련성(Answer Relevancy)",
    "context_precision": "문맥 정밀도(Context Precision)",
    "context_recall": "문맥 재현율(Context Recall)",
    "answer_correctness": "답변 정확성(Answer Correctness)",
}


def _import_ragas():
    """
    ragas import를 한 곳에 모아, 실패 시 원인과 해결 방법을 바로 알려줍니다.
    (ragas 0.4.x는 langchain-community가 ChatVertexAI를 langchain-google-vertexai로
     이전하면서 생긴 알려진 호환성 버그가 있습니다 -> pip install "ragas<0.4" 권장)
    """
    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            AnswerCorrectness,
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
        from ragas.run_config import RunConfig
    except ImportError as e:  # pragma: no cover - 환경 진단용
        print(
            "\n[오류] ragas 관련 모듈을 import하지 못했습니다.\n"
            f"  원인: {e}\n\n"
            "이 오류가 'langchain_community.chat_models.vertexai' 를 언급한다면 "
            "ragas 0.4.x의 알려진 호환성 버그입니다.\n"
            "  해결: pip install \"ragas<0.4\"  (0.3.9 등 0.4 미만 버전 사용)\n"
            "그 외의 경우 requirements.txt 의 5단계 패키지가 설치되어 있는지 확인해 주세요.\n"
            "  pip install \"ragas<0.4\" datasets pandas openpyxl langchain-ollama\n"
        )
        raise SystemExit(1) from e

    return {
        "EvaluationDataset": EvaluationDataset,
        "SingleTurnSample": SingleTurnSample,
        "evaluate": evaluate,
        "LangchainEmbeddingsWrapper": LangchainEmbeddingsWrapper,
        "LangchainLLMWrapper": LangchainLLMWrapper,
        "AnswerCorrectness": AnswerCorrectness,
        "AnswerRelevancy": AnswerRelevancy,
        "ContextPrecision": ContextPrecision,
        "ContextRecall": ContextRecall,
        "Faithfulness": Faithfulness,
        "RunConfig": RunConfig,
    }


def load_qa_dataset(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["items"]


def _extract_contexts(result: dict) -> list[str]:
    """그래프 실행 결과에서 RAGAS에 넣을 retrieved_contexts(list[str])를 추출합니다."""
    rag_documents = result.get("rag_documents") or []
    if rag_documents:
        return [d["text"] for d in rag_documents]

    web_results = result.get("web_results") or []
    if web_results:
        return [f"{r.get('title', '')}\n{r.get('content', '')}".strip() for r in web_results]

    # 미지원 제품 등 검색을 전혀 수행하지 않은 경우: 빈 리스트를 그대로 넘기면
    # 일부 RAGAS 지표가 에러를 낼 수 있어 빈 문자열 하나로 채웁니다.
    # (이 경우 context_precision/recall은 의미상 낮게 나오는 것이 정상입니다)
    return [""]


def run_agent_and_collect(items: list[dict]) -> list[dict]:
    """10개 질문에 대해 agent_main.run_agent()를 호출하고 RAGAS 평가용 필드를 수집합니다."""
    rows = []
    for item in items:
        question = item["question"]
        print(f"[실행] {item['id']}: {question}")

        start = time.time()
        result = run_agent(question)
        elapsed = time.time() - start

        contexts = _extract_contexts(result)

        rows.append(
            {
                "id": item["id"],
                "product": item.get("product", ""),
                "expected_route": item.get("expected_route", ""),
                "actual_route": result.get("route", ""),
                "user_input": question,
                "response": result.get("answer", ""),
                "retrieved_contexts": contexts,
                "reference": item["ground_truth"],
                "sources": result.get("source", ""),
                "elapsed_sec": round(elapsed, 2),
            }
        )
    return rows


def build_ragas_dataset(rows: list[dict], ragas_mod: dict):
    SingleTurnSample = ragas_mod["SingleTurnSample"]
    EvaluationDataset = ragas_mod["EvaluationDataset"]

    samples = [
        SingleTurnSample(
            user_input=r["user_input"],
            response=r["response"],
            retrieved_contexts=r["retrieved_contexts"],
            reference=r["reference"],
        )
        for r in rows
    ]
    return EvaluationDataset(samples=samples)


def get_ragas_metrics(ragas_mod: dict) -> list:
    return [
        ragas_mod["Faithfulness"](),
        ragas_mod["AnswerRelevancy"](),
        ragas_mod["ContextPrecision"](),
        ragas_mod["ContextRecall"](),
        ragas_mod["AnswerCorrectness"](),
    ]


def get_ragas_llm_and_embeddings(ragas_mod: dict):
    """RAGAS 평가에 사용할 LLM/Embedding을 Ollama로 래핑합니다 (건물 답변 생성에 쓰는
    모델과 동일하게 config.LLM_MODEL / config.EMBEDDING_MODEL 을 사용)."""
    from langchain_ollama import ChatOllama, OllamaEmbeddings

    llm = ChatOllama(model=config.LLM_MODEL, temperature=0, base_url=config.OLLAMA_BASE_URL)
    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL, base_url=config.OLLAMA_BASE_URL)

    return (
        ragas_mod["LangchainLLMWrapper"](llm),
        ragas_mod["LangchainEmbeddingsWrapper"](embeddings),
    )


def get_ragas_run_config(ragas_mod: dict):
    """
    ragas evaluate()의 실행 동시성/타임아웃 설정.

    기본값(timeout=180초, max_workers=16)은 클라우드 LLM API처럼 다수 요청을
    동시에 처리할 수 있는 환경을 가정합니다. 로컬 Ollama 서버(특히 14b급 모델)는
    요청을 사실상 순차 처리하므로, 16개 작업이 한꺼번에 몰리면 뒤에서 대기하는
    작업들이 180초 안에 처리되지 못해 TimeoutError가 나고 해당 지표가 NaN으로
    남습니다 (질문 10개 x 지표 5개 = 50개 작업이 동시에 큐에 쌓이는 구조).

    config.RAGAS_MAX_WORKERS(기본 2)로 동시 실행 수를 크게 낮추고,
    config.RAGAS_TIMEOUT(기본 600초)으로 작업 1개당 대기 시간을 넉넉히 주어
    순서대로 처리되도록 합니다. 서버가 더 빠르거나 느리면 환경변수
    (RAGAS_MAX_WORKERS / RAGAS_TIMEOUT / RAGAS_MAX_WAIT / RAGAS_MAX_RETRIES)로
    조정하세요.
    """
    RunConfig = ragas_mod["RunConfig"]
    return RunConfig(
        timeout=config.RAGAS_TIMEOUT,
        max_workers=config.RAGAS_MAX_WORKERS,
        max_wait=config.RAGAS_MAX_WAIT,
        max_retries=config.RAGAS_MAX_RETRIES,
    )


def run_ragas_evaluation(rows: list[dict], ragas_mod: dict):
    dataset = build_ragas_dataset(rows, ragas_mod)
    metrics = get_ragas_metrics(ragas_mod)
    ragas_llm, ragas_embeddings = get_ragas_llm_and_embeddings(ragas_mod)
    run_config = get_ragas_run_config(ragas_mod)

    evaluate = ragas_mod["evaluate"]
    return evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config,
    )


def print_summary_table(merged_df: pd.DataFrame) -> None:
    metric_cols = [c for c in METRIC_LABELS if c in merged_df.columns]

    print("\n" + "=" * 70)
    print("RAGAS 평가지표별 평균 점수 (10개 질문 기준)")
    print("=" * 70)
    summary = merged_df[metric_cols].mean(numeric_only=True).round(4)
    for metric in metric_cols:
        label = METRIC_LABELS.get(metric, metric)
        print(f"  {label:32s}: {summary[metric]:.4f}")
    print(f"\n  전체 평균: {summary.mean():.4f}  (n={len(merged_df)})")


def print_detail_table(merged_df: pd.DataFrame) -> None:
    metric_cols = [c for c in METRIC_LABELS if c in merged_df.columns]
    display_cols = ["id", "expected_route", "actual_route"] + metric_cols

    print("\n" + "=" * 70)
    print("질문별 상세 결과")
    print("=" * 70)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(merged_df[display_cols].round(4).to_string(index=False))


def save_outputs(merged_df: pd.DataFrame) -> None:
    merged_df.to_csv(CSV_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    merged_df.to_excel(XLSX_OUTPUT_PATH, index=False)
    print(f"\nCSV 저장 완료  : {CSV_OUTPUT_PATH}")
    print(f"Excel 저장 완료 : {XLSX_OUTPUT_PATH}")


def main() -> None:
    ragas_mod = _import_ragas()

    items = load_qa_dataset(QA_PATH)
    rows = run_agent_and_collect(items)

    print(
        "\n그래프 실행 완료. RAGAS 평가를 시작합니다... "
        "(지표당 다수의 LLM 호출이 발생해 시간이 걸릴 수 있습니다)"
    )
    ragas_result = run_ragas_evaluation(rows, ragas_mod)

    scores_df = ragas_result.to_pandas()  # user_input/retrieved_contexts/response/reference + 지표 점수
    meta_df = pd.DataFrame(rows)[
        ["id", "product", "expected_route", "actual_route", "sources", "elapsed_sec"]
    ]
    merged_df = pd.concat([meta_df, scores_df], axis=1)

    print_summary_table(merged_df)
    print_detail_table(merged_df)
    save_outputs(merged_df)


if __name__ == "__main__":
    main()
