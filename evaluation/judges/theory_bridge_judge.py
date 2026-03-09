"""
judges/theory_bridge_judge.py
理论桥接评分 Judge

评估教练回复是否有效地将书籍理论与用户的实际情况相结合。
评分标准 1-5 分，使用 DeepSeek Chat 作为评分模型。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from evaluation.eval_utils import call_llm_judge, parse_score

# ─── Prompt 模板 ──────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """请评估以下 AI 教练回复的**理论桥接质量**（1-5 分）。

理论桥接是指：将书籍中的知识/理论，与用户当前的真实处境相结合，而非机械照本宣科。

**用户问题**：{question}

**用户日记**（可为空）：{diary}

**参考资料/书籍内容**（可为空）：{contexts}

**AI 教练回复**：
{answer}

**评分标准**：
- **1 分** = 没有引用任何理论，或引用了与用户处境完全无关的内容
- **2 分** = 引用了理论，但只是原文照搬，没有与用户情况关联
- **3 分** = 引用了相关理论，并简单解释了和用户的关系，但关联较浅
- **4 分** = 理论引用恰当，用教练语言重新表达，并清晰说明了「为什么这个理论与你的情况相关」
- **5 分** = 理论与用户处境深度融合，不是在讲书，而是在用书中的智慧照亮用户的真实经历，令人有「原来如此」的顿悟感

**特殊情况**：
- 如果用户是闲聊或边界场景，无需理论引用，给 **3 分**（中性）
- 如果没有参考资料但教练编造了书籍引用，给 **1 分**

**注意**：只评价理论桥接，请只输出一个整数（1、2、3、4 或 5），不要加任何解释"""


def judge(
    question: str,
    answer: str,
    diary: str | None = None,
    contexts: list[str] | None = None,
    **kwargs,
) -> dict:
    """
    评估理论桥接质量

    Args:
        question: 用户问题
        answer:   AI 教练回复
        diary:    用户日记（可选）
        contexts: 检索到的书籍内容列表（可选）

    Returns:
        {"score": float, "raw": str, "metric": "theory_bridge"}
    """
    contexts_text = "\n".join(f"- {c}" for c in contexts) if contexts else "（无参考资料）"

    prompt = PROMPT_TEMPLATE.format(
        question=question,
        diary=diary or "（无日记）",
        contexts=contexts_text,
        answer=answer,
    )

    raw = call_llm_judge(prompt)
    score = parse_score(raw)

    return {
        "metric": "theory_bridge",
        "score": score,
        "raw": raw,
    }


if __name__ == "__main__":
    result = judge(
        question="我总是忍不住刷手机，有什么书里的方法能帮我？",
        answer="你描述的状态和《认知觉醒》里提到的「心智带宽被占用」很像——当手机的即时反馈不断消耗你的注意力资源，你会越来越难以主动选择做更重要的事。书中建议的方法是：**把手机从触手可及的地方移走**，因为「提示」是习惯触发的第一个环节，消除触发点比靠意志力更有效。",
        contexts=["心智带宽被即时反馈消耗，导致注意力资源匮乏。提示是习惯的第一环节。"],
    )
    print(f"理论桥接评分：{result}")
