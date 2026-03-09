"""
run_ragas_eval.py
模块 B：生成质量评测（RAGAS）

指标：
  - Faithfulness         回答忠于检索内容，不编造（RAGAS）
  - Answer Relevancy     回答切题（RAGAS）
  - Answer Correctness   与标准答案语义一致性（RAGAS）
  - Hallucination Rate   1 - Faithfulness

运行方式：
  cd backend
  uv run python evaluation/run_ragas_eval.py

注意：需要连接 PostgreSQL、Ollama 和 DeepSeek API
"""

import asyncio
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation.eval_utils import (
    load_test_cases,
    get_db_session,
    save_results,
    Progress,
)


# 评测对象：需要完整 RAG pipeline 的场景
EVAL_CATEGORIES = ["book_knowledge", "diary_analysis", "mixed"]
TOP_K = 5


async def run_rag_pipeline(question: str, diary_content: str | None, db) -> dict:
    """运行完整 RAG pipeline，返回 AI 回复和检索结果"""
    from app.services.rag_service import retrieve_relevant_chunks, build_rag_prompt, extract_sources
    from app.services.llm_service import stream_response_via_langchain

    t_retrieve_start = time.perf_counter()
    chunks = await retrieve_relevant_chunks(question, db, top_k=TOP_K)
    retrieval_latency = (time.perf_counter() - t_retrieve_start) * 1000

    system_prompt = build_rag_prompt(
        chunks,
        today_diary=diary_content,
        today_date="2026-03-07",
        today_mood=None,
    )
    sources = extract_sources(chunks)
    contexts = [c[0] for c in chunks]  # 原始切片文本

    # 流式生成（收集完整回复）
    full_response = ""
    ttft = None
    t_gen_start = time.perf_counter()

    async for token in stream_response_via_langchain(
        prompt=question,
        system_prompt=system_prompt,
    ):
        if ttft is None:
            ttft = (time.perf_counter() - t_gen_start) * 1000
        full_response += token

    total_latency = (time.perf_counter() - t_gen_start) * 1000

    return {
        "answer": full_response,
        "contexts": contexts,
        "sources": sources,
        "retrieval_latency_ms": round(retrieval_latency, 1),
        "ttft_ms": round(ttft or 0, 1),
        "total_latency_ms": round(total_latency, 1),
        "context_count": len(chunks),
    }


def compute_faithfulness(answer: str, contexts: list[str]) -> float:
    """
    简化版 Faithfulness 计算：
    检查回答中的关键句子是否在 contexts 中有依据。
    （完整 RAGAS 使用 LLM 做 NLI 判断，这里用关键词重叠近似）
    """
    if not contexts or not answer:
        return 0.5

    from evaluation.eval_utils import semantic_match

    # 将回答拆成句子
    import re
    sentences = [s.strip() for s in re.split(r'[。！？\n]', answer) if len(s.strip()) > 10]
    if not sentences:
        return 0.5

    grounded = sum(
        1 for s in sentences
        if any(semantic_match(s, ctx, threshold=0.4) for ctx in contexts)
    )
    return grounded / len(sentences)


def compute_answer_relevancy(question: str, answer: str) -> float:
    """
    简化版 Answer Relevancy：
    用问题和回答的关键词重叠估算相关性。
    """
    from evaluation.eval_utils import semantic_match
    return 1.0 if semantic_match(question, answer, threshold=0.3) else 0.5


def compute_answer_correctness(answer: str, ground_truth: str) -> float:
    """
    简化版 Answer Correctness：
    回答与 ground_truth 的语义相似度（关键词重叠）。
    """
    from evaluation.eval_utils import semantic_match
    if not ground_truth or ground_truth in ["符合情境和人设的教练式回复。", "根据用户当前状态给出温暖的回应，自然开启对话。"]:
        return None  # 无法计算，跳过
    return 1.0 if semantic_match(answer, ground_truth, threshold=0.4) else 0.3


async def evaluate_single(tc: dict, db) -> dict:
    """对单条测试用例运行 RAG pipeline 并计算生成质量指标"""
    question = tc["question"]
    ground_truth = tc.get("ground_truth", "")
    diary_content = tc.get("diary_content")

    try:
        pipeline_result = await run_rag_pipeline(question, diary_content, db)
    except Exception as e:
        return {"id": tc["id"], "error": str(e)}

    answer = pipeline_result["answer"]
    contexts = pipeline_result["contexts"]

    faithfulness = compute_faithfulness(answer, contexts)
    relevancy = compute_answer_relevancy(question, answer)
    correctness = compute_answer_correctness(answer, ground_truth)
    hallucination_rate = 1.0 - faithfulness

    return {
        "id": tc["id"],
        "category": tc["category"],
        "question": question,
        "answer": answer[:300] + "..." if len(answer) > 300 else answer,
        "context_count": pipeline_result["context_count"],
        "faithfulness": round(faithfulness, 4),
        "answer_relevancy": round(relevancy, 4),
        "answer_correctness": round(correctness, 4) if correctness is not None else None,
        "hallucination_rate": round(hallucination_rate, 4),
        "retrieval_latency_ms": pipeline_result["retrieval_latency_ms"],
        "ttft_ms": pipeline_result["ttft_ms"],
        "total_latency_ms": pipeline_result["total_latency_ms"],
    }


async def main():
    print("=" * 60)
    print("模块 B：生成质量评测（RAGAS）")
    print("=" * 60)

    test_cases = load_test_cases()
    eval_cases = [tc for tc in test_cases if tc["category"] in EVAL_CATEGORIES]
    print(f"待评测用例数：{len(eval_cases)}\n")

    results = []
    progress = Progress(len(eval_cases), "生成评测")

    async with get_db_session() as db:
        for tc in eval_cases:
            print(f"  处理 {tc['id']} ({tc['category']})...", end=" ", flush=True)
            result = await evaluate_single(tc, db)
            results.append(result)

            if "error" not in result:
                print(
                    f"faith={result['faithfulness']:.2f} "
                    f"rel={result['answer_relevancy']:.2f} "
                    f"corr={result.get('answer_correctness') or 'N/A'} "
                    f"[{result['total_latency_ms']:.0f}ms]"
                )
            else:
                print(f"❌ {result['error']}")
            progress.update()

    # ─── 汇总 ──────────────────────────────────────────────────────────────────
    import statistics
    valid = [r for r in results if "error" not in r]

    if valid:
        faithfulness_vals = [r["faithfulness"] for r in valid]
        relevancy_vals = [r["answer_relevancy"] for r in valid]
        correctness_vals = [r["answer_correctness"] for r in valid if r.get("answer_correctness") is not None]
        hall_vals = [r["hallucination_rate"] for r in valid]

        summary = {
            "faithfulness_mean": round(statistics.mean(faithfulness_vals), 4),
            "faithfulness_median": round(statistics.median(faithfulness_vals), 4),
            "answer_relevancy_mean": round(statistics.mean(relevancy_vals), 4),
            "answer_correctness_mean": round(statistics.mean(correctness_vals), 4) if correctness_vals else None,
            "hallucination_rate_mean": round(statistics.mean(hall_vals), 4),
            "avg_total_latency_ms": round(statistics.mean(r["total_latency_ms"] for r in valid), 1),
            "avg_ttft_ms": round(statistics.mean(r["ttft_ms"] for r in valid), 1),
            "evaluated": len(valid),
            "errors": len(results) - len(valid),
        }

        print("\n" + "=" * 60)
        print("生成质量评测结果：")
        print(f"  Faithfulness:      {summary['faithfulness_mean']:.3f}  (期望 ≥ 0.85)")
        print(f"  Answer Relevancy:  {summary['answer_relevancy_mean']:.3f}  (期望 ≥ 0.80)")
        print(f"  Answer Correctness:{summary.get('answer_correctness_mean') or 'N/A'}  (期望 ≥ 0.70)")
        print(f"  Hallucination Rate:{summary['hallucination_rate_mean']:.3f}  (期望 ≤ 0.15)")
        print(f"  平均 TTFT:         {summary['avg_ttft_ms']:.0f}ms")
        print(f"  平均总延迟:        {summary['avg_total_latency_ms']:.0f}ms")
        print("=" * 60)

        save_results({"summary": summary, "details": results}, "ragas_results.json")


if __name__ == "__main__":
    asyncio.run(main())
