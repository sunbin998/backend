"""
run_retrieval_eval.py
模块 A：检索质量评测

指标：
  - Context Precision   Top-K 中相关片段排在前面的比例
  - Context Recall      ground_truth_contexts 被检索覆盖的比例
  - Hit Rate@5          Top-5 中至少含一个相关片段的比例
  - MRR@5               第一个相关片段排名的倒数均值

运行方式：
  cd backend
  uv run python evaluation/run_retrieval_eval.py

注意：需要连接 PostgreSQL 和 Ollama（用于 embed_query）
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
    semantic_match,
)


# 只评测有 ground_truth_contexts 的场景
EVAL_CATEGORIES = ["book_knowledge", "mixed"]
TOP_K = 5


async def evaluate_single(tc: dict, db) -> dict:
    """对单条测试用例进行检索评测"""
    from app.services.rag_service import retrieve_relevant_chunks

    question = tc["question"]
    ground_truth_ctxs = tc.get("ground_truth_contexts", [])

    t0 = time.perf_counter()
    retrieved = await retrieve_relevant_chunks(question, db, top_k=TOP_K)
    latency_ms = (time.perf_counter() - t0) * 1000

    retrieved_texts = [r[0] for r in retrieved]  # (content, score, metadata)

    # ─── 判断相关性 ──────────────────────────────────────────────────────────
    def is_relevant(retrieved_text: str) -> bool:
        """判断检索片段是否与任一 ground_truth_context 相关"""
        if not ground_truth_ctxs:
            return False
        return any(semantic_match(retrieved_text, gt) for gt in ground_truth_ctxs)

    relevance = [is_relevant(t) for t in retrieved_texts]

    # ─── Hit Rate@5 ──────────────────────────────────────────────────────────
    hit = any(relevance)

    # ─── MRR@5 ───────────────────────────────────────────────────────────────
    mrr = 0.0
    for rank, rel in enumerate(relevance, start=1):
        if rel:
            mrr = 1.0 / rank
            break

    # ─── Context Precision ───────────────────────────────────────────────────
    # 检索结果中相关片段占比，且相关片段在前面的加权
    if retrieved_texts:
        relevant_count = sum(relevance)
        precision = relevant_count / len(retrieved_texts)

        # 加权 precision（相关片段越靠前越好）
        weighted_sum = sum(
            sum(relevance[:i+1]) / (i + 1)
            for i, rel in enumerate(relevance)
            if rel
        )
        context_precision = weighted_sum / max(relevant_count, 1) if relevant_count > 0 else 0.0
    else:
        precision = 0.0
        context_precision = 0.0

    # ─── Context Recall ───────────────────────────────────────────────────────
    # ground_truth_contexts 中被检索覆盖的比例
    if ground_truth_ctxs:
        covered = sum(
            1 for gt in ground_truth_ctxs
            if any(semantic_match(rt, gt) for rt in retrieved_texts)
        )
        context_recall = covered / len(ground_truth_ctxs)
    else:
        context_recall = None  # 无 ground truth，跳过

    return {
        "id": tc["id"],
        "category": tc["category"],
        "question": question,
        "retrieved_count": len(retrieved_texts),
        "relevance_flags": relevance,
        "hit_rate": 1.0 if hit else 0.0,
        "mrr": mrr,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "retrieval_latency_ms": round(latency_ms, 1),
    }


async def main():
    print("=" * 60)
    print("模块 A：检索质量评测")
    print("=" * 60)

    test_cases = load_test_cases()
    # 只评测有 ground_truth_contexts 的测试用例
    eval_cases = [
        tc for tc in test_cases
        if tc["category"] in EVAL_CATEGORIES and tc.get("ground_truth_contexts")
    ]
    print(f"待评测用例数：{len(eval_cases)}（{EVAL_CATEGORIES}）\n")

    results = []
    progress = Progress(len(eval_cases), "检索评测")

    async with get_db_session() as db:
        for tc in eval_cases:
            try:
                result = await evaluate_single(tc, db)
                results.append(result)
                print(
                    f"  {result['id']} | hit={result['hit_rate']:.0f} "
                    f"mrr={result['mrr']:.2f} "
                    f"prec={result['context_precision']:.2f} "
                    f"recall={result.get('context_recall') or 'N/A'} "
                    f"[{result['retrieval_latency_ms']:.0f}ms]"
                )
            except Exception as e:
                print(f"  ❌ {tc['id']} 失败: {e}")
                results.append({"id": tc["id"], "error": str(e)})
            progress.update()

    # ─── 汇总统计 ─────────────────────────────────────────────────────────────
    valid = [r for r in results if "error" not in r]
    if valid:
        import statistics

        hit_rate  = statistics.mean(r["hit_rate"] for r in valid)
        mrr       = statistics.mean(r["mrr"] for r in valid)
        precision = statistics.mean(r["context_precision"] for r in valid)

        recall_valid = [r["context_recall"] for r in valid if r.get("context_recall") is not None]
        recall = statistics.mean(recall_valid) if recall_valid else None

        latencies = [r["retrieval_latency_ms"] for r in valid]
        avg_latency = statistics.mean(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

        summary = {
            "hit_rate_at_5": round(hit_rate, 4),
            "mrr_at_5": round(mrr, 4),
            "context_precision": round(precision, 4),
            "context_recall": round(recall, 4) if recall is not None else None,
            "avg_retrieval_latency_ms": round(avg_latency, 1),
            "p95_retrieval_latency_ms": round(p95_latency, 1),
            "evaluated": len(valid),
            "errors": len(results) - len(valid),
        }

        print("\n" + "=" * 60)
        print("检索质量评测结果：")
        print(f"  Hit Rate@5:        {summary['hit_rate_at_5']:.3f}  (期望 ≥ 0.85)")
        print(f"  MRR@5:             {summary['mrr_at_5']:.3f}  (期望 ≥ 0.70)")
        print(f"  Context Precision: {summary['context_precision']:.3f}  (期望 ≥ 0.75)")
        print(f"  Context Recall:    {summary.get('context_recall') or 'N/A'}  (期望 ≥ 0.80)")
        print(f"  平均检索延迟:      {summary['avg_retrieval_latency_ms']:.1f}ms")
        print(f"  P95 检索延迟:      {summary['p95_retrieval_latency_ms']:.1f}ms")
        print("=" * 60)

        save_results({"summary": summary, "details": results}, "retrieval_results.json")
    else:
        print("❌ 没有有效结果")


if __name__ == "__main__":
    asyncio.run(main())
