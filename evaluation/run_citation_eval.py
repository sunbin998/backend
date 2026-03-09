"""
run_citation_eval.py
模块 D：引用准确性评测

指标：
  - 引用覆盖率     使用书籍检索内容时，标注了 [来源:...] 的比例
  - 引用正确率     标注的来源能追溯到实际检索片段的比例
  - 幻觉引用率     引用了 sources 中不存在的来源的比例

运行方式：
  cd backend
  uv run python evaluation/run_citation_eval.py

依赖：ragas_results.json（需先运行 run_ragas_eval.py），
      或实时生成 AI 回复。
"""

import asyncio
import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation.eval_utils import (
    load_test_cases,
    get_db_session,
    save_results,
    load_results,
    Progress,
    semantic_match,
)


# 引用格式：[来源: 书名] 或 [来源: 日记 YYYY-MM-DD]
CITATION_PATTERN = re.compile(r'\[来源[：:]\s*([^\]]+)\]')


def extract_citations(text: str) -> list[str]:
    """从 AI 回复中提取所有引用标注"""
    return [m.strip() for m in CITATION_PATTERN.findall(text)]


def check_citation_valid(citation: str, sources: list[dict]) -> bool:
    """
    检查引用是否能在实际 sources 中找到对应内容
    sources 格式: [{"filename": "...", "score": ..., "preview": "..."}]
    """
    if not sources:
        return False

    for src in sources:
        filename = src.get("filename", "")
        preview = src.get("preview", "")
        if semantic_match(citation, filename, threshold=0.4) or semantic_match(citation, preview, threshold=0.3):
            return True
    return False


async def get_full_pipeline_result(tc: dict, db) -> dict | None:
    """运行完整 RAG pipeline，获取 AI 回复和 sources"""
    from app.services.rag_service import retrieve_relevant_chunks, build_rag_prompt, extract_sources
    from app.services.llm_service import stream_response_via_langchain

    try:
        chunks = await retrieve_relevant_chunks(tc["question"], db, top_k=5)
        sources = extract_sources(chunks)
        system_prompt = build_rag_prompt(
            chunks,
            today_diary=tc.get("diary_content"),
            today_date="2026-03-07",
        )
        answer = ""
        async for token in stream_response_via_langchain(
            prompt=tc["question"],
            system_prompt=system_prompt,
        ):
            answer += token

        return {"answer": answer, "sources": sources, "context_count": len(chunks)}
    except Exception as e:
        return {"error": str(e)}


def evaluate_citations(answer: str, sources: list[dict], has_contexts: bool) -> dict:
    """分析单条回复的引用质量"""
    citations = extract_citations(answer)
    citation_count = len(citations)

    # ─── 引用覆盖率 ────────────────────────────────────────────────────────────
    # 定义：当有检索内容时，是否标注了来源
    if has_contexts and len(sources) > 0:
        # 回复中是否有任何引用
        coverage = 1.0 if citation_count > 0 else 0.0
    else:
        coverage = None  # 无检索内容，不评价覆盖率

    # ─── 引用正确率 ────────────────────────────────────────────────────────────
    if citation_count > 0:
        valid_count = sum(1 for c in citations if check_citation_valid(c, sources))
        correctness = valid_count / citation_count
    else:
        correctness = None  # 无引用，不评价正确率

    # ─── 幻觉引用率 ────────────────────────────────────────────────────────────
    if citation_count > 0 and sources is not None:
        hallucination_count = sum(1 for c in citations if not check_citation_valid(c, sources))
        hallucination_rate = hallucination_count / citation_count
    else:
        hallucination_rate = 0.0

    return {
        "citations": citations,
        "citation_count": citation_count,
        "coverage": coverage,
        "correctness": correctness,
        "hallucination_rate": hallucination_rate,
    }


async def main():
    print("=" * 60)
    print("模块 D：引用准确性评测")
    print("=" * 60)

    test_cases = load_test_cases()
    # 主要评测有书籍内容的场景
    eval_cases = [
        tc for tc in test_cases
        if tc["category"] in ["book_knowledge", "mixed", "diary_analysis"]
    ]
    print(f"待评测用例数：{len(eval_cases)}\n")

    results = []
    progress = Progress(len(eval_cases), "引用评测")

    async with get_db_session() as db:
        for tc in eval_cases:
            print(f"  处理 {tc['id']}...", end=" ", flush=True)

            pipeline = await get_full_pipeline_result(tc, db)
            if "error" in pipeline:
                print(f"❌ {pipeline['error']}")
                results.append({"id": tc["id"], "error": pipeline["error"]})
                progress.update()
                continue

            answer = pipeline["answer"]
            sources = pipeline["sources"]
            has_contexts = pipeline["context_count"] > 0

            citation_metrics = evaluate_citations(answer, sources, has_contexts)

            citations_str = ", ".join(citation_metrics["citations"]) or "（无引用）"
            print(
                f"引用数={citation_metrics['citation_count']} "
                f"覆盖={citation_metrics['coverage'] or 'N/A'} "
                f"正确率={citation_metrics['correctness'] or 'N/A'} "
                f"幻觉={citation_metrics['hallucination_rate']:.2f}"
            )

            results.append({
                "id": tc["id"],
                "category": tc["category"],
                "question": tc["question"],
                "answer_preview": answer[:150] + "..." if len(answer) > 150 else answer,
                "has_retrieval": has_contexts,
                "source_count": len(sources),
                **citation_metrics,
            })
            progress.update()

    # ─── 汇总 ──────────────────────────────────────────────────────────────────
    import statistics
    valid = [r for r in results if "error" not in r]

    if valid:
        coverage_vals = [r["coverage"] for r in valid if r.get("coverage") is not None]
        correctness_vals = [r["correctness"] for r in valid if r.get("correctness") is not None]
        hallucination_vals = [r["hallucination_rate"] for r in valid]
        citation_counts = [r["citation_count"] for r in valid]

        summary = {
            "citation_coverage": round(statistics.mean(coverage_vals), 4) if coverage_vals else None,
            "citation_correctness": round(statistics.mean(correctness_vals), 4) if correctness_vals else None,
            "hallucination_rate": round(statistics.mean(hallucination_vals), 4),
            "avg_citations_per_response": round(statistics.mean(citation_counts), 2),
            "responses_with_citations": sum(1 for r in valid if r["citation_count"] > 0),
            "evaluated": len(valid),
            "errors": len(results) - len(valid),
        }

        print("\n" + "=" * 60)
        print("引用准确性评测结果：")
        print(f"  引用覆盖率:        {summary['citation_coverage'] or 'N/A'}  (有检索内容时是否标注来源)")
        print(f"  引用正确率:        {summary['citation_correctness'] or 'N/A'}  (引用能追溯到实际来源的比例)")
        print(f"  幻觉引用率:        {summary['hallucination_rate']:.3f}  (期望 ≤ 0.10)")
        print(f"  平均引用数/回复:   {summary['avg_citations_per_response']:.1f}")
        print(f"  含引用的回复数:    {summary['responses_with_citations']}/{len(valid)}")
        print("=" * 60)

        save_results({"summary": summary, "details": results}, "citation_results.json")


if __name__ == "__main__":
    asyncio.run(main())
