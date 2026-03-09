"""
generate_report.py
阶段 4：汇总评测报告生成器

读取 5 个评测模块的结果文件，生成：
  - 五维雷达图        reports/radar_chart.png
  - 详细评测报告      reports/evaluation_report.md
    ├── 总分概览（雷达图嵌入）
    ├── 指标明细表
    ├── 场景对比表
    ├── 低分案例分析（得分 ≤ 2 的 coaching cases）
    └── 优化建议

运行方式：
  cd backend
  uv run python evaluation/generate_report.py

注意：需要先运行 5 个评测模块生成 results/*.json 文件。
      如果结果文件不存在，会生成占位示例报告。
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation.eval_utils import RESULTS_DIR, REPORTS_DIR


# ─── 加载结果 ─────────────────────────────────────────────────────────────────

def load_result_safe(filename: str) -> dict | None:
    """安全加载结果文件，不存在时返回 None"""
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def extract_summary(data: dict | None) -> dict:
    """从结果文件中提取 summary 字段"""
    if data is None:
        return {}
    if "summary" in data:
        return data["summary"]
    return data


def extract_details(data: dict | None) -> list:
    """从结果文件中提取 details 列表"""
    if data is None:
        return []
    if "details" in data:
        return data["details"]
    if isinstance(data, list):
        return data
    return []


# ─── 分数归一化（各指标统一映射到 0-1 范围）─────────────────────────────────

def normalize_retrieval(summary: dict) -> float:
    """检索综合分（4 个指标的加权均值，转为 0-100）"""
    vals = [
        summary.get("hit_rate_at_5"),
        summary.get("mrr_at_5"),
        summary.get("context_precision"),
        summary.get("context_recall"),
    ]
    valid = [v for v in vals if v is not None]
    return (sum(valid) / len(valid) * 100) if valid else 0.0


def normalize_generation(summary: dict) -> float:
    """生成质量综合分（3 个指标均值，转为 0-100）"""
    vals = [
        summary.get("faithfulness_mean"),
        summary.get("answer_relevancy_mean"),
        summary.get("answer_correctness_mean"),
    ]
    valid = [v for v in vals if v is not None]
    return (sum(valid) / len(valid) * 100) if valid else 0.0


def normalize_coaching(summary: dict) -> float:
    """教练质量综合分（6 judge 均值映射到 0-100）"""
    overall = summary.get("overall_avg")
    if overall is not None:
        return (overall / 5.0) * 100
    metrics = summary.get("metrics", {})
    vals = [v for v in metrics.values() if v is not None]
    return (sum(vals) / len(vals) / 5.0 * 100) if vals else 0.0


def normalize_citation(summary: dict) -> float:
    """引用质量综合分"""
    coverage = summary.get("citation_coverage") or 0.5  # 默认 0.5 若无数据
    correctness = summary.get("citation_correctness") or 0.5
    hallucination = summary.get("hallucination_rate") or 0.0
    score = (coverage * 0.4 + correctness * 0.4 + (1 - hallucination) * 0.2)
    return score * 100


def normalize_performance(summary: dict, thresholds: dict = None) -> float:
    """
    性能综合分：基于各延迟指标与期望阈值的比较
    thresholds: {指标: 期望上限ms}
    """
    if thresholds is None:
        thresholds = {
            "embed_latency_ms": 200,
            "retrieval_latency_ms": 300,
            "ttft_ms": 2000,
            "total_latency_ms": 15000,
        }
    scores = []
    for metric, limit in thresholds.items():
        sub = summary.get(metric, {})
        if isinstance(sub, dict):
            mean_val = sub.get("mean")
        else:
            mean_val = None
        if mean_val is not None:
            score = min(1.0, limit / mean_val) if mean_val > 0 else 1.0
            scores.append(score)
    return (sum(scores) / len(scores) * 100) if scores else 0.0


# ─── 雷达图 ───────────────────────────────────────────────────────────────────

def draw_radar_chart(scores: dict[str, float], output_path: Path) -> bool:
    """
    绘制五维雷达图
    scores: {"检索质量": 82.0, "生成质量": 75.0, ...}  (0-100)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, message="Glyph.*missing")

        # 使用英文标签（避免 Linux 无中文字体的警告）
        label_map = {
            "检索质量": "Retrieval",
            "生成质量": "Generation",
            "教练质量": "Coaching",
            "引用质量": "Citation",
            "性能表现": "Performance",
        }
        labels = [label_map.get(k, k) for k in scores.keys()]
        values = list(scores.values())
        N = len(labels)

        # 计算角度
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        values_plot = values + [values[0]]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        # 背景网格
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20", "40", "60", "80", "100"], color="#888", fontsize=8)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, color="white", fontsize=12, fontweight="bold")
        ax.grid(color="#444", linewidth=0.5, linestyle="--")

        # 雷达面和轮廓
        ax.plot(angles, values_plot, color="#7c3aed", linewidth=2.5, linestyle="-")
        ax.fill(angles, values_plot, color="#7c3aed", alpha=0.25)

        # 标注各点分值
        for angle, value, label in zip(angles[:-1], values, labels):
            x = angle
            y = value + 6
            ax.annotate(
                f"{value:.1f}",
                xy=(x, y),
                color="#c4b5fd",
                fontsize=10,
                ha="center",
                va="center",
            )

        ax.set_title(
            "觉醒教练 RAG 系统评测总览",
            color="white",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"✅ 雷达图已保存: {output_path}")
        return True

    except ImportError as e:
        print(f"⚠️  matplotlib 未安装，跳过雷达图生成: {e}")
        return False
    except Exception as e:
        print(f"⚠️  雷达图生成失败: {e}")
        return False


# ─── 报告生成 ─────────────────────────────────────────────────────────────────

def find_low_score_cases(coaching_details: list, threshold: float = 2.5) -> list:
    """找出教练评测中平均分 ≤ threshold 的案例"""
    low = []
    for item in coaching_details:
        if "error" in item:
            continue
        avg = item.get("avg_coaching_score")
        if avg is not None and avg <= threshold:
            low.append(item)
    return sorted(low, key=lambda x: x["avg_coaching_score"])


def generate_optimization_suggestions(
    retrieval_summary: dict,
    generation_summary: dict,
    coaching_summary: dict,
    citation_summary: dict,
    performance_summary: dict,
) -> list[str]:
    """根据各模块分数生成针对性优化建议"""
    suggestions = []

    # 检索
    hit_rate = retrieval_summary.get("hit_rate_at_5", 1.0)
    recall = retrieval_summary.get("context_recall", 1.0)
    if hit_rate < 0.85:
        suggestions.append(
            f"**检索命中率偏低** ({hit_rate:.2f} < 0.85)：建议尝试降低相似度阈值（当前 0.3），"
            "或对知识库切片做更细粒度切分（chunk_size 从 500 → 300）"
        )
    if recall and recall < 0.80:
        suggestions.append(
            f"**Context Recall 偏低** ({recall:.2f} < 0.80)：考虑增大 top_k（从 5 → 8），"
            "或对 query 做 HyDE（假设文档扩展）再检索"
        )

    # 生成
    faithfulness = generation_summary.get("faithfulness_mean", 1.0)
    hallucination = generation_summary.get("hallucination_rate_mean", 0.0)
    if faithfulness < 0.85:
        suggestions.append(
            f"**Faithfulness 不足** ({faithfulness:.2f} < 0.85)：在 System Prompt 中增加"
            "「只基于参考资料回答，不允许凭空推断」的明确约束"
        )
    if hallucination > 0.15:
        suggestions.append(
            f"**幻觉率偏高** ({hallucination:.2f} > 0.15)：引入输出后处理——"
            "对每个声明做 NLI 验证，不在 context 中的声明标注「教练建议」"
        )

    # 教练质量
    if coaching_summary:
        metrics = coaching_summary.get("metrics", {})
        for metric, val in metrics.items():
            if val is not None and val < 3.0:
                tips = {
                    "empathy": "在 System Prompt 中明确要求「先共情再建议」，避免直接给方案",
                    "diary_integration": "强化日记注入时的「洞察提示」，要求教练从日记中提取 1-2 个具体细节",
                    "theory_bridge": "在理论桥接部分加入「用两句话说明为什么这个理论与你相关」的要求",
                    "actionability": "要求回复包含「今天就能做的第一步」，并给出时间/地点/方式三要素",
                    "structure": "在 response_framework 中为每个步骤加入示例，强化五步结构输出",
                    "safety": "完善 guardrails 规则，对危机关键词做前置检测并强制触发危机回复模板",
                }
                suggestions.append(
                    f"**{metric} 评分偏低** ({val:.2f}/5)：{tips.get(metric, '需要针对性改进')}"
                )

    # 引用
    hallucination_citation = citation_summary.get("hallucination_rate", 0.0)
    if hallucination_citation is not None and hallucination_citation > 0.1:
        suggestions.append(
            f"**幻觉引用率偏高** ({hallucination_citation:.2f} > 0.10)：考虑在回复后"
            "自动验证 `[来源:]` 标注是否在实际 sources 列表中，过滤无效引用"
        )

    # 性能
    ttft = performance_summary.get("ttft_ms", {})
    if isinstance(ttft, dict) and ttft.get("mean", 0) > 2000:
        suggestions.append(
            f"**TTFT 偏高** (均值 {ttft['mean']:.0f}ms > 2000ms)："
            "考虑切换到 qwen-turbo-online（更小延迟），或对常见问题做回复预热"
        )

    if not suggestions:
        suggestions.append("✅ 所有指标均达到期望阈值，系统表现良好！")

    return suggestions


def generate_markdown_report(
    scores: dict[str, float],
    retrieval_summary: dict,
    generation_summary: dict,
    coaching_summary: dict,
    citation_summary: dict,
    performance_summary: dict,
    coaching_details: list,
    radar_path: Path | None,
    output_path: Path,
    generated_at: str,
) -> str:
    """生成完整 Markdown 报告"""
    lines = []

    # 标题
    lines += [
        "# 觉醒教练 RAG 系统 · 评测报告",
        "",
        f"> 生成时间：{generated_at}",
        "> 数据来源：《认知觉醒》+ 《认知天性》（PostgreSQL pgvector 知识库）",
        "> 评测集：62 条（真实书籍切片生成）",
        "",
        "---",
        "",
    ]

    # 总览雷达图
    lines += ["## 综合得分总览", ""]
    if radar_path and radar_path.exists():
        rel_path = radar_path.name
        lines.append(f"![RAG 评测雷达图]({rel_path})")
    else:
        lines.append("_(雷达图未生成，请确认 matplotlib 已安装)_")
    lines.append("")

    # 五维得分表
    lines += [
        "| 维度 | 综合得分 | 期望 | 状态 |",
        "|------|---------|------|------|",
    ]
    thresholds = {
        "检索质量": 80,
        "生成质量": 75,
        "教练质量": 72,  # 3.6/5 * 100
        "引用质量": 75,
        "性能表现": 80,
    }
    for dim, score in scores.items():
        th = thresholds.get(dim, 75)
        status = "✅" if score >= th else ("⚠️" if score >= th * 0.8 else "❌")
        lines.append(f"| {dim} | {score:.1f}/100 | ≥{th} | {status} |")
    lines += ["", "---", ""]

    # 检索质量
    lines += ["## 模块 A：检索质量", ""]
    if retrieval_summary:
        expected = {
            "hit_rate_at_5": ("Hit Rate@5", "≥ 0.85"),
            "mrr_at_5": ("MRR@5", "≥ 0.70"),
            "context_precision": ("Context Precision", "≥ 0.75"),
            "context_recall": ("Context Recall", "≥ 0.80"),
        }
        lines += ["| 指标 | 均值 | 期望 |", "|------|------|------|"]
        for key, (name, exp) in expected.items():
            val = retrieval_summary.get(key)
            val_str = f"{val:.3f}" if val is not None else "N/A"
            lines.append(f"| {name} | {val_str} | {exp} |")
        latency = retrieval_summary.get("avg_retrieval_latency_ms")
        if latency:
            lines.append(f"| 平均检索延迟 | {latency:.0f}ms | ≤ 300ms |")
    else:
        lines.append("_未找到结果文件，请先运行 `run_retrieval_eval.py`_")
    lines += ["", "---", ""]

    # 生成质量
    lines += ["## 模块 B：生成质量（RAGAS）", ""]
    if generation_summary:
        gen_metrics = {
            "faithfulness_mean": ("Faithfulness", "≥ 0.85"),
            "answer_relevancy_mean": ("Answer Relevancy", "≥ 0.80"),
            "answer_correctness_mean": ("Answer Correctness", "≥ 0.70"),
            "hallucination_rate_mean": ("Hallucination Rate", "≤ 0.15"),
        }
        lines += ["| 指标 | 均值 | 期望 |", "|------|------|------|"]
        for key, (name, exp) in gen_metrics.items():
            val = generation_summary.get(key)
            val_str = f"{val:.3f}" if val is not None else "N/A"
            lines.append(f"| {name} | {val_str} | {exp} |")
    else:
        lines.append("_未找到结果文件，请先运行 `run_ragas_eval.py`_")
    lines += ["", "---", ""]

    # 教练质量
    lines += ["## 模块 C：教练质量（LLM-as-Judge）", ""]
    if coaching_summary:
        metrics = coaching_summary.get("metrics", {})
        overall = coaching_summary.get("overall_avg")
        lines += [
            f"**综合平均分：{overall:.2f}/5.00**" if overall else "",
            "",
            "| 指标 | 得分 | 满分 |",
            "|------|------|------|",
        ]
        metric_names = {
            "empathy": "共情度",
            "diary_integration": "日记整合度",
            "theory_bridge": "理论桥接",
            "actionability": "行动可行性",
            "structure": "结构完整度",
            "safety": "安全性",
        }
        for key, val in metrics.items():
            name = metric_names.get(key, key)
            val_str = f"{val:.2f}" if val is not None else "N/A"
            bar = "█" * int((val or 0) * 2)
            lines.append(f"| {name} | {val_str} `{bar}` | 5.00 |")

        # 场景对比
        by_cat = coaching_summary.get("by_category", {})
        if by_cat:
            lines += [
                "",
                "**按场景分类得分：**",
                "",
                "| 场景 | 平均分 |",
                "|------|--------|",
            ]
            cat_names = {
                "book_knowledge": "纯书籍知识",
                "diary_analysis": "日记分析",
                "mixed": "混合场景",
                "chitchat": "闲聊",
                "boundary": "边界/安全",
                "multi_turn": "多轮对话",
            }
            for cat, avg in by_cat.items():
                lines.append(f"| {cat_names.get(cat, cat)} | {avg:.2f}/5 |")
    else:
        lines.append("_未找到结果文件，请先运行 `run_coaching_eval.py`_")
    lines += ["", "---", ""]

    # 引用质量
    lines += ["## 模块 D：引用准确性", ""]
    if citation_summary:
        lines += ["| 指标 | 数值 | 说明 |", "|------|------|------|"]
        lines.append(f"| 引用覆盖率 | {citation_summary.get('citation_coverage') or 'N/A'} | 有检索内容时是否标注来源 |")
        lines.append(f"| 引用正确率 | {citation_summary.get('citation_correctness') or 'N/A'} | 引用可追溯到实际 sources |")
        lines.append(f"| 幻觉引用率 | {citation_summary.get('hallucination_rate', 0):.3f} | 期望 ≤ 0.10 |")
        lines.append(f"| 平均引用数/回复 | {citation_summary.get('avg_citations_per_response', 0):.1f} | |")
    else:
        lines.append("_未找到结果文件，请先运行 `run_citation_eval.py`_")
    lines += ["", "---", ""]

    # 性能表现
    lines += ["## 模块 E：性能表现", ""]
    if performance_summary:
        perf_metrics = {
            "embed_latency_ms": ("Embedding 延迟", "≤ 200ms"),
            "retrieval_latency_ms": ("检索延迟", "≤ 300ms"),
            "ttft_ms": ("首 Token 延迟 (TTFT)", "≤ 2000ms"),
            "total_latency_ms": ("完整响应时间", "≤ 15000ms"),
        }
        lines += ["| 指标 | 均值 | 中位 | P95 | 期望 |", "|------|------|------|-----|------|"]
        for key, (name, exp) in perf_metrics.items():
            data = performance_summary.get(key, {})
            if isinstance(data, dict):
                mean = data.get("mean", "N/A")
                median = data.get("median", "N/A")
                p95 = data.get("p95", "N/A")
                mean_str = f"{mean:.0f}ms" if isinstance(mean, (int, float)) else mean
                median_str = f"{median:.0f}ms" if isinstance(median, (int, float)) else median
                p95_str = f"{p95:.0f}ms" if isinstance(p95, (int, float)) else p95
            else:
                mean_str = median_str = p95_str = "N/A"
            lines.append(f"| {name} | {mean_str} | {median_str} | {p95_str} | {exp} |")
        samples = performance_summary.get("samples")
        if samples:
            lines.append(f"\n> 基于 {samples} 条随机样本测量")
    else:
        lines.append("_未找到结果文件，请先运行 `run_performance_eval.py`_")
    lines += ["", "---", ""]

    # 低分案例分析
    low_score_cases = find_low_score_cases(coaching_details, threshold=2.5)
    lines += ["## 低分案例分析", ""]
    if low_score_cases:
        lines.append(f"以下 {len(low_score_cases)} 条用例教练综合得分 ≤ 2.5，需要重点关注：\n")
        for item in low_score_cases[:5]:  # 最多展示 5 条
            scores_detail = {
                k: v.get("score") for k, v in item.get("scores", {}).items()
                if v.get("score") is not None
            }
            score_str = " | ".join(f"{k}={v:.1f}" for k, v in scores_detail.items())
            lines += [
                f"### {item['id']} — {item['category']}",
                f"**问题**：{item['question']}",
                f"**综合分**：{item['avg_coaching_score']:.2f}/5  `{score_str}`",
                f"**回复片段**：{item.get('answer_preview', 'N/A')}",
                "",
            ]
    else:
        lines.append("✅ 暂无教练综合得分 ≤ 2.5 的案例。")
    lines += ["", "---", ""]

    # 优化建议
    suggestions = generate_optimization_suggestions(
        retrieval_summary, generation_summary, coaching_summary,
        citation_summary, performance_summary,
    )
    lines += [
        "## 优化建议",
        "",
        "根据评测数据，建议按以下优先级优化：",
        "",
    ]
    for i, s in enumerate(suggestions, 1):
        lines.append(f"{i}. {s}")
        lines.append("")

    lines += [
        "---",
        "",
        "_本报告由 `generate_report.py` 自动生成，评测系统版本 v1.0_",
    ]

    content = "\n".join(lines)

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return content


# ─── 主程序 ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("阶段 4：评测报告生成器")
    print("=" * 60)

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 加载各模块结果
    print("\n加载评测结果文件...")
    retrieval_data   = load_result_safe("retrieval_results.json")
    ragas_data       = load_result_safe("ragas_results.json")
    coaching_data    = load_result_safe("coaching_results.json")
    citation_data    = load_result_safe("citation_results.json")
    performance_data = load_result_safe("performance_results.json")

    for name, data in [
        ("retrieval_results.json", retrieval_data),
        ("ragas_results.json", ragas_data),
        ("coaching_results.json", coaching_data),
        ("citation_results.json", citation_data),
        ("performance_results.json", performance_data),
    ]:
        status = "✅" if data else "⚠️  (未找到，将使用默认值)"
        print(f"  {name}: {status}")

    # 提取各模块 summary
    retrieval_summary   = extract_summary(retrieval_data)
    generation_summary  = extract_summary(ragas_data)
    coaching_summary    = extract_summary(coaching_data)
    citation_summary    = extract_summary(citation_data)
    performance_summary = extract_summary(performance_data)
    coaching_details    = extract_details(coaching_data)

    # 计算五维综合分（0-100）
    scores = {
        "检索质量": normalize_retrieval(retrieval_summary),
        "生成质量": normalize_generation(generation_summary),
        "教练质量": normalize_coaching(coaching_summary),
        "引用质量": normalize_citation(citation_summary),
        "性能表现": normalize_performance(performance_summary),
    }

    print(f"\n五维综合得分：")
    for dim, score in scores.items():
        bar = "█" * int(score / 5)
        print(f"  {dim:<8} {score:5.1f}/100  {bar}")

    # 生成雷达图
    radar_path = REPORTS_DIR / "radar_chart.png"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    draw_radar_chart(scores, radar_path)

    # 生成 Markdown 报告
    report_path = REPORTS_DIR / "evaluation_report.md"
    content = generate_markdown_report(
        scores=scores,
        retrieval_summary=retrieval_summary,
        generation_summary=generation_summary,
        coaching_summary=coaching_summary,
        citation_summary=citation_summary,
        performance_summary=performance_summary,
        coaching_details=coaching_details,
        radar_path=radar_path,
        output_path=report_path,
        generated_at=generated_at,
    )

    print(f"\n✅ 报告已生成：{report_path}")
    print(f"   字数：{len(content)} 字")
    print("=" * 60)


if __name__ == "__main__":
    main()
