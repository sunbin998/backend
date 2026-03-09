"""
judges/actionability_judge.py
行动可行性评分 Judge

评估教练回复中的建议是否具体可执行（用户今天就能开始做）。
评分标准 1-5 分，使用 DeepSeek Chat 作为评分模型。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from evaluation.eval_utils import call_llm_judge, parse_score

# ─── Prompt 模板 ──────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """请评估以下 AI 教练回复中**行动建议的可行性**（1-5 分）。

**用户问题**：{question}

**用户日记**（可为空）：{diary}

**AI 教练回复**：
{answer}

**评分标准**：
- **1 分** = 只有空洞的价值观说教（如"你要改变心态"、"你应该更自律"），没有任何具体建议
- **2 分** = 有建议但极其模糊（如"你可以试试管理时间"），无法直接执行
- **3 分** = 建议有一定方向，但步骤不够具体，用户需要再思考才能开始
- **4 分** = 建议明确具体，用户今天就能开始（如"明晚9点把手机放到另一个房间"）
- **5 分** = 建议既具体可操作，又考虑到了用户当前的能量状态和处境，循序渐进，有助于真正执行

**判断维度**：
- 是否有**具体的第一步**（时间、地点、方式）？
- 步骤是否符合用户当前状态（能量低时不建议高难度行动）？
- 是否用"你可以试试..."而非"你应该..."？

**特殊情况**：
- 如果是边界场景（如用户表达危机），不需要行动建议，给 **3 分**（中性）
- 如果是闲聊场景，轻量建议即可，给最高 **4 分**

请只输出一个整数（1、2、3、4 或 5），不要加任何解释"""


def judge(
    question: str,
    answer: str,
    diary: str | None = None,
    **kwargs,
) -> dict:
    """
    评估行动可行性

    Args:
        question: 用户问题
        answer:   AI 教练回复
        diary:    用户日记（可选）

    Returns:
        {"score": float, "raw": str, "metric": "actionability"}
    """
    prompt = PROMPT_TEMPLATE.format(
        question=question,
        diary=diary or "（无日记）",
        answer=answer,
    )

    raw = call_llm_judge(prompt)
    score = parse_score(raw)

    return {
        "metric": "actionability",
        "score": score,
        "raw": raw,
    }


if __name__ == "__main__":
    result = judge(
        question="我想建立一个早起习惯，该怎么开始？",
        answer="你可以试试这样开始：**今晚就把明天的闹钟设为比平时早15分钟**，而不是突然早起一小时。闹钟响后，在床边准备好你最喜欢的饮料，起来后第一件事就是去拿它——这给起床加了一个小奖励。就做这一件事，其他不变。",
        diary="一直想早起但总是失败，今天又睡到九点。",
    )
    print(f"行动可行性评分：{result}")
