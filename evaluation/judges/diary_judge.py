"""
judges/diary_judge.py
日记整合度评分 Judge

评估教练回复是否恰当地引用和整合了用户的日记内容。
评分标准 1-5 分，使用 DeepSeek Chat 作为评分模型。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from evaluation.eval_utils import call_llm_judge, parse_score

# ─── Prompt 模板 ──────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """请评估以下 AI 教练回复对用户**日记内容的整合度**（1-5 分）。

**用户问题**：{question}

**用户日记**：{diary}

**AI 教练回复**：
{answer}

**评分标准**：
- **1 分** = 有日记但完全忽视，没有任何引用或关联
- **2 分** = 仅泛泛提及日记存在，没有具体内容引用
- **3 分** = 引用了日记内容，但引用生硬，关联不够自然
- **4 分** = 自然地将日记细节融入回答，帮助用户从日记中看到规律或洞察
- **5 分** = 精准捕捉日记中的关键信号，引用准确且有深度，帮助用户发现自己没注意到的模式或亮点

**特殊情况**：
- 如果**没有提供日记**（"无日记"），且教练没有强行引用虚构内容，则给 **3 分**（中性）
- 如果没有日记但教练**编造了日记内容**，给 **1 分**

**注意**：只评价日记整合度，请只输出一个整数（1、2、3、4 或 5），不要加任何解释"""


def judge(
    question: str,
    answer: str,
    diary: str | None = None,
    **kwargs,
) -> dict:
    """
    评估日记整合度

    Args:
        question: 用户问题
        answer:   AI 教练回复
        diary:    用户日记内容（可选，无日记时测试模型是否编造）

    Returns:
        {"score": float, "raw": str, "metric": "diary_integration"}
    """
    prompt = PROMPT_TEMPLATE.format(
        question=question,
        diary=diary or "（无日记）",
        answer=answer,
    )

    raw = call_llm_judge(prompt)
    score = parse_score(raw)

    return {
        "metric": "diary_integration",
        "score": score,
        "raw": raw,
    }


if __name__ == "__main__":
    result = judge(
        question="帮我分析今天的日记，有什么值得关注的吗？",
        answer="我注意到你今天提到「永远做不完的事」——这个词很有意思，说明你感到的不只是忙碌，而是一种无法追上的无力感。这可能是一个值得深入看的模式。",
        diary="加班到深夜，感觉永远做不完的事，领导还觉得进度慢。",
    )
    print(f"日记整合度评分：{result}")
