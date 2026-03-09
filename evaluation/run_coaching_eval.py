"""
run_coaching_eval.py
模块 C：教练质量评测（LLM-as-Judge）

对每条测试用例的 AI 回复，使用 6 个 Judge 分别打分：
  - 共情度 (empathy)
  - 日记整合度 (diary_integration)
  - 理论桥接 (theory_bridge)
  - 行动可行性 (actionability)
  - 结构完整度 (structure)
  - 安全性 (safety)

依赖 run_ragas_eval.py 的输出（ragas_results.json）中生成的 AI 回复。
若结果文件不存在，会先实时生成 AI 回复再评分。

运行方式：
  cd backend
  uv run python evaluation/run_coaching_eval.py

注意：需要 DeepSeek API 密钥，每条用例调用 6 次 API
"""

import asyncio
import sys
import os
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation.eval_utils import (
    load_test_cases,
    get_db_session,
    save_results,
    load_results,
    Progress,
)
from evaluation.judges import (
    empathy_judge,
    diary_judge,
    theory_bridge_judge,
    actionability_judge,
    structure_judge,
    safety_judge,
    infer_scene_type,
)


async def get_answer_for_case(tc: dict, db, cached_answers: dict) -> str | None:
    """获取测试用例的 AI 回复（优先使用缓存）"""
    if tc["id"] in cached_answers:
        return cached_answers[tc["id"]]

    # 实时生成
    from app.services.rag_service import retrieve_relevant_chunks, build_rag_prompt
    from app.services.llm_service import stream_response_via_langchain

    try:
        chunks = await retrieve_relevant_chunks(tc["question"], db, top_k=5)
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
        return answer
    except Exception as e:
        print(f"  生成回复失败 {tc['id']}: {e}")
        return None


def score_with_all_judges(
    question: str,
    answer: str,
    diary: str | None,
    contexts: list[str] | None,
    category: str,
    tags: list[str],
) -> dict:
    """用全部 6 个 Judge 对一条回复打分"""
    scene_type = infer_scene_type(category, tags)

    scores = {}

    for judge_fn, name in [
        (empathy_judge,       "empathy"),
        (diary_judge,         "diary_integration"),
        (theory_bridge_judge, "theory_bridge"),
        (actionability_judge, "actionability"),
        (structure_judge,     "structure"),
    ]:
        try:
            result = judge_fn(
                question=question,
                answer=answer,
                diary=diary,
                contexts=contexts,
            )
            scores[name] = result
        except Exception as e:
            scores[name] = {"metric": name, "score": None, "error": str(e)}

    # Safety 需要额外的 scene_type 参数
    try:
        safety_result = safety_judge(
            question=question,
            answer=answer,
            diary=diary,
            scene_type=scene_type,
        )
        scores["safety"] = safety_result
    except Exception as e:
        scores["safety"] = {"metric": "safety", "score": None, "error": str(e)}

    return scores


async def main():
    print("=" * 60)
    print("模块 C：教练质量评测（LLM-as-Judge × 6）")
    print("=" * 60)

    # 1. 加载或生成 AI 回复缓存
    cached_answers = {}
    ragas_path = os.path.join(os.path.dirname(__file__), "results", "ragas_results.json")
    if os.path.exists(ragas_path):
        try:
            ragas_data = load_results("ragas_results.json")
            details = ragas_data.get("details", ragas_data)
            for item in details:
                if "answer" in item and "error" not in item:
                    # ragas 结果里的 answer 可能被截断，只作参考
                    cached_answers[item["id"]] = item["answer"]
            print(f"  从 ragas_results.json 加载了 {len(cached_answers)} 条缓存回复")
        except Exception:
            pass

    # 2. 加载测试用例（所有分类都参与教练评测）
    test_cases = load_test_cases()
    print(f"  待评测用例数：{len(test_cases)}\n")

    results = []
    progress = Progress(len(test_cases), "教练评测")

    async with get_db_session() as db:
        for tc in test_cases:
            print(f"\n  [{tc['id']}] {tc['question'][:50]}...")

            # 获取 AI 回复
            answer = await get_answer_for_case(tc, db, cached_answers)
            if not answer:
                results.append({"id": tc["id"], "error": "无法获取 AI 回复"})
                progress.update()
                continue

            # 6 个 Judge 评分
            t0 = time.time()
            judge_scores = score_with_all_judges(
                question=tc["question"],
                answer=answer,
                diary=tc.get("diary_content"),
                contexts=tc.get("ground_truth_contexts"),
                category=tc["category"],
                tags=tc.get("tags", []),
            )
            judge_time = time.time() - t0

            # 汇总得分
            valid_scores = {k: v["score"] for k, v in judge_scores.items() if v.get("score") is not None}
            avg_score = sum(valid_scores.values()) / len(valid_scores) if valid_scores else None

            print(f"         " + " | ".join(f"{k}={v:.1f}" for k, v in valid_scores.items()))
            print(f"         平均分: {avg_score:.2f}  [{judge_time:.1f}s]")

            results.append({
                "id": tc["id"],
                "category": tc["category"],
                "question": tc["question"],
                "answer_preview": answer[:150] + "..." if len(answer) > 150 else answer,
                "scores": judge_scores,
                "avg_coaching_score": round(avg_score, 4) if avg_score else None,
            })
            progress.update()

    # ─── 汇总统计 ─────────────────────────────────────────────────────────────
    import statistics
    JUDGE_METRICS = ["empathy", "diary_integration", "theory_bridge", "actionability", "structure", "safety"]
    valid = [r for r in results if "error" not in r and r.get("avg_coaching_score")]

    if valid:
        metric_avgs = {}
        for metric in JUDGE_METRICS:
            vals = [r["scores"][metric]["score"] for r in valid
                    if r["scores"].get(metric, {}).get("score") is not None]
            metric_avgs[metric] = round(statistics.mean(vals), 4) if vals else None

        # 按分类汇总
        from collections import defaultdict
        by_cat = defaultdict(list)
        for r in valid:
            by_cat[r["category"]].append(r["avg_coaching_score"])
        cat_avgs = {cat: round(statistics.mean(scores), 3) for cat, scores in by_cat.items()}

        summary = {
            "metrics": metric_avgs,
            "overall_avg": round(statistics.mean(r["avg_coaching_score"] for r in valid), 4),
            "by_category": cat_avgs,
            "evaluated": len(valid),
            "errors": len(results) - len(valid),
        }

        print("\n" + "=" * 60)
        print("教练质量评测结果（各指标均值）：")
        for metric, avg in metric_avgs.items():
            bar = "█" * int((avg or 0) * 4)
            print(f"  {metric:<20} {avg or 'N/A':.2f}  {bar}")
        print(f"\n  综合平均分:       {summary['overall_avg']:.2f} / 5.00")
        print("\n  按场景分类：")
        for cat, avg in cat_avgs.items():
            print(f"    {cat}: {avg:.2f}")
        print("=" * 60)

        save_results({"summary": summary, "details": results}, "coaching_results.json")


if __name__ == "__main__":
    asyncio.run(main())
