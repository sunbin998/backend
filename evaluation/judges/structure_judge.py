"""
judges/structure_judge.py
结构完整度评分 Judge

评估教练回复是否遵循设计的五步框架：
共情锚定 → 日记洞察 → 理论桥接 → 行动阶梯 → 反思提问

评分标准 1-5 分，使用 DeepSeek Chat 作为评分模型。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from evaluation.eval_utils import call_llm_judge, parse_score

# ─── Prompt 模板 ──────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """请评估以下 AI 教练回复的**结构完整度**（1-5 分）。

「觉醒教练」的理想回复结构（五步框架）：
1. **共情锚定**：回应用户情绪，让 ta 感到被理解
2. **日记洞察**：从日记中提取用户没注意到的模式或亮点（有日记时）
3. **理论桥接**：引用书籍理论并与用户情况关联
4. **行动阶梯**：给出 1-3 个具体可执行的步骤
5. **反思提问**：留一个开放性问题引导用户继续思考（可选）

**用户问题**：{question}

**用户日记**（可为空）：{diary}

**AI 教练回复**：
{answer}

**评分标准**：
- **1 分** = 结构混乱，缺失超过 3 个步骤，或只有单一维度（如只给建议）
- **2 分** = 有 2 个步骤，结构不完整
- **3 分** = 有 3 个步骤，基本结构存在
- **4 分** = 有 4 个步骤，结构清晰，各部分过渡自然
- **5 分** = 完整五步，比例合理，整体流畅自然，不机械

**弹性原则**：
- 闲聊/追问场景不需要完整框架，评估适应性表现，最高 **3 分**
- 边界场景（危机）应以共情+建议专业帮助为主，不需完整框架，最高 **4 分**
- 没有日记时，日记洞察步骤可省略

请只输出一个整数（1、2、3、4 或 5），不要加任何解释"""


def judge(
    question: str,
    answer: str,
    diary: str | None = None,
    **kwargs,
) -> dict:
    """
    评估结构完整度

    Args:
        question: 用户问题
        answer:   AI 教练回复
        diary:    用户日记（可选）

    Returns:
        {"score": float, "raw": str, "metric": "structure"}
    """
    prompt = PROMPT_TEMPLATE.format(
        question=question,
        diary=diary or "（无日记）",
        answer=answer,
    )

    raw = call_llm_judge(prompt)
    score = parse_score(raw)

    return {
        "metric": "structure",
        "score": score,
        "raw": raw,
    }


if __name__ == "__main__":
    result = judge(
        question="最近总感到焦虑，书里有什么方法可以帮我？",
        answer="""我听到你了——焦虑的感觉很真实，它是你的身体在告诉你某些事情需要关注 💙

我注意到你今天日记里写到「永远做不完的事」，这和焦虑很有关系——当任务看起来没有终点，大脑会持续处于高警觉状态。

《认知觉醒》里提到，焦虑往往来自「模糊感」——任务边界不清晰时，大脑无法判断何时可以放松。

你可以试试今天就做这一件事：**把明天要做的事写下来，只写三件**。不需要全部列出，就三件。这给大脑一个清晰的边界。

一个小问题留给你思考：在这些焦虑里，有哪一件事是你今天就能掌控的？""",
        diary="加班到深夜，感觉永远做不完的事，很焦虑。",
    )
    print(f"结构完整度评分：{result}")
