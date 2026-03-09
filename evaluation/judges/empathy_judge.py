"""
judges/empathy_judge.py
共情度评分 Judge

评估教练回复是否具体回应了用户的情绪和处境。
评分标准 1-5 分，使用 DeepSeek Chat 作为评分模型。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from evaluation.eval_utils import call_llm_judge, parse_score

# ─── Prompt 模板 ──────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """请评估以下 AI 教练回复的**共情度**（1-5 分）。

**用户问题**：{question}

**用户日记**（可为空）：{diary}

**AI 教练回复**：
{answer}

**评分标准**：
- **1 分** = 完全没有共情，直接给建议或讲道理，忽视用户情绪
- **2 分** = 有通用共情语句但不具体（如"我理解你的感受"），感觉机械
- **3 分** = 能回应用户情绪，但没有关联到用户描述的具体情境
- **4 分** = 具体回应了用户的情绪和处境，有温度，用户会感到被理解
- **5 分** = 深度共情，准确捕捉了用户话语背后的真实感受，共情表达自然而有力

**注意**：
- 只评价共情度，不评价其他维度
- 如果用户没有提供日记，共情应基于问题本身的情绪信号
- 请只输出一个整数（1、2、3、4 或 5），不要加任何解释"""


def judge(
    question: str,
    answer: str,
    diary: str | None = None,
    **kwargs,
) -> dict:
    """
    评估共情度

    Args:
        question: 用户问题
        answer:   AI 教练回复
        diary:    用户日记内容（可选）

    Returns:
        {"score": float, "raw": str, "metric": "empathy"}
    """
    prompt = PROMPT_TEMPLATE.format(
        question=question,
        diary=diary or "（无日记）",
        answer=answer,
    )

    raw = call_llm_judge(prompt)
    score = parse_score(raw)

    return {
        "metric": "empathy",
        "score": score,
        "raw": raw,
    }


if __name__ == "__main__":
    # 快速测试
    result = judge(
        question="我今天加班到很晚，感觉永远做不完的事，很累很焦虑。",
        answer="我注意到你今天承受了非常大的压力——不仅是体力上的疲惫，更是一种「永远追不上」的无力感。这种焦虑是真实的，不是你意志力不够强。",
        diary="加班到深夜，永远做不完。",
    )
    print(f"共情度评分：{result}")
