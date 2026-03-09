"""
run_performance_eval.py
模块 E：性能评测

指标：
  - Embedding 延迟      embed_query() 耗时
  - 检索延迟            retrieve_relevant_chunks() 耗时
  - TTFT               首 token 延迟（发出请求到收到第一个 token）
  - 完整响应时间        发出请求到收到完整回复
  - Token 使用量        估算输入/输出 token 数

运行方式：
  cd backend
  uv run python evaluation/run_performance_eval.py [--samples N]

N 默认为 20（从测试集随机抽样），可调小以快速测试。
"""

import asyncio
import sys
import os
import time
import statistics
import argparse
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation.eval_utils import (
    load_test_cases,
    get_db_session,
    save_results,
)


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数（中文约 1.5字/token，英文约 4字/token）"""
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars
    return int(chinese_chars / 1.5 + other_chars / 4)


async def benchmark_embedding(question: str) -> float:
    """测量 embed_query 延迟（毫秒）"""
    import asyncio
    from app.services.embedding_service import embed_query

    t0 = time.perf_counter()
    await asyncio.to_thread(embed_query, question)
    return (time.perf_counter() - t0) * 1000


async def benchmark_retrieval(question: str, db) -> tuple[list, float]:
    """测量检索延迟（毫秒），返回 (chunks, latency_ms)"""
    from app.services.rag_service import retrieve_relevant_chunks

    t0 = time.perf_counter()
    chunks = await retrieve_relevant_chunks(question, db, top_k=5)
    latency_ms = (time.perf_counter() - t0) * 1000
    return chunks, latency_ms


async def benchmark_generation(question: str, system_prompt: str) -> dict:
    """测量生成延迟（TTFT + 完整响应），返回性能指标"""
    from app.services.llm_service import stream_response_via_langchain

    full_response = ""
    ttft_ms = None
    t0 = time.perf_counter()

    async for token in stream_response_via_langchain(
        prompt=question,
        system_prompt=system_prompt,
    ):
        if ttft_ms is None:
            ttft_ms = (time.perf_counter() - t0) * 1000
        full_response += token

    total_ms = (time.perf_counter() - t0) * 1000

    return {
        "ttft_ms": round(ttft_ms or 0, 1),
        "total_latency_ms": round(total_ms, 1),
        "response_length": len(full_response),
        "estimated_output_tokens": estimate_tokens(full_response),
    }


async def benchmark_single(tc: dict, db) -> dict | None:
    """对一条测试用例进行完整性能基准测试"""
    from app.services.rag_service import build_rag_prompt

    question = tc["question"]

    try:
        # 1. Embedding 延迟
        embed_latency = await benchmark_embedding(question)

        # 2. 检索延迟
        chunks, retrieval_latency = await benchmark_retrieval(question, db)

        # 3. 构建 System Prompt（计算输入 token）
        system_prompt = build_rag_prompt(
            chunks,
            today_diary=tc.get("diary_content"),
            today_date="2026-03-07",
        )
        estimated_input_tokens = (
            estimate_tokens(system_prompt) + estimate_tokens(question)
        )

        # 4. 生成延迟
        gen_metrics = await benchmark_generation(question, system_prompt)

        return {
            "id": tc["id"],
            "category": tc["category"],
            "question_length": len(question),
            "context_count": len(chunks),
            "embed_latency_ms": round(embed_latency, 1),
            "retrieval_latency_ms": round(retrieval_latency, 1),
            "ttft_ms": gen_metrics["ttft_ms"],
            "total_latency_ms": gen_metrics["total_latency_ms"],
            "response_length": gen_metrics["response_length"],
            "estimated_input_tokens": estimated_input_tokens,
            "estimated_output_tokens": gen_metrics["estimated_output_tokens"],
        }

    except Exception as e:
        print(f"  ❌ {tc['id']} 失败: {e}")
        return None


async def main():
    parser = argparse.ArgumentParser(description="RAG 性能评测")
    parser.add_argument("--samples", type=int, default=20, help="测试样本数量（默认 20）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    print("=" * 60)
    print(f"模块 E：性能评测（{args.samples} 条样本）")
    print("=" * 60)

    all_cases = load_test_cases()
    # 从有意义的场景中随机抽样
    sample_pool = [
        tc for tc in all_cases
        if tc["category"] in ["book_knowledge", "diary_analysis", "mixed"]
    ]
    random.seed(args.seed)
    sample_cases = random.sample(sample_pool, min(args.samples, len(sample_pool)))
    print(f"测试样本：{len(sample_cases)} 条\n")

    results = []

    async with get_db_session() as db:
        for i, tc in enumerate(sample_cases, 1):
            print(f"  [{i}/{len(sample_cases)}] {tc['id']} {tc['question'][:40]}...", end=" ", flush=True)
            result = await benchmark_single(tc, db)
            if result:
                results.append(result)
                print(
                    f"embed={result['embed_latency_ms']:.0f}ms "
                    f"retr={result['retrieval_latency_ms']:.0f}ms "
                    f"ttft={result['ttft_ms']:.0f}ms "
                    f"total={result['total_latency_ms']:.0f}ms"
                )
            else:
                print("跳过")

    if not results:
        print("❌ 无有效结果")
        return

    # ─── 汇总统计 ─────────────────────────────────────────────────────────────
    def agg(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        if not vals:
            return {}
        return {
            "mean": round(statistics.mean(vals), 1),
            "median": round(statistics.median(vals), 1),
            "p95": round(sorted(vals)[int(len(vals) * 0.95)], 1),
            "min": round(min(vals), 1),
            "max": round(max(vals), 1),
        }

    summary = {
        "samples": len(results),
        "embed_latency_ms": agg("embed_latency_ms"),
        "retrieval_latency_ms": agg("retrieval_latency_ms"),
        "ttft_ms": agg("ttft_ms"),
        "total_latency_ms": agg("total_latency_ms"),
        "estimated_input_tokens": agg("estimated_input_tokens"),
        "estimated_output_tokens": agg("estimated_output_tokens"),
        "avg_context_count": round(statistics.mean(r["context_count"] for r in results), 1),
    }

    print("\n" + "=" * 60)
    print("性能评测汇总：")
    thresholds = {
        "embed_latency_ms": ("≤ 200ms", 200),
        "retrieval_latency_ms": ("≤ 300ms", 300),
        "ttft_ms": ("≤ 2000ms", 2000),
        "total_latency_ms": ("≤ 15000ms", 15000),
    }
    for metric, data in summary.items():
        if not isinstance(data, dict):
            print(f"  {metric}: {data}")
            continue
        th_str, th_val = thresholds.get(metric, ("", None))
        status = ""
        if th_val and data.get("mean"):
            status = "✅" if data["mean"] <= th_val else "❌"
        print(f"  {metric:<30} {status}")
        print(f"    均值={data['mean']}  中位={data['median']}  P95={data['p95']}  {th_str}")

    print("=" * 60)
    save_results({"summary": summary, "details": results}, "performance_results.json")


if __name__ == "__main__":
    asyncio.run(main())
