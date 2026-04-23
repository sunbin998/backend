# backend/app/services/llm_service.py
"""
LLM 调用服务（DeepSeek deepseek-reasoner，流式）
支持：
- 自定义 system prompt（来自 RAG 检索结果）
- 对话历史上下文
"""
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import AsyncGenerator, List, Optional
from dotenv import load_dotenv

load_dotenv()

llm = ChatDeepSeek(
    model="deepseek-reasoner",
    temperature=0.7,
    streaming=True,
)

# 默认 system prompt（无 RAG 上下文时使用）
DEFAULT_SYSTEM_PROMPT = "你是一个个人成长教练，用温暖、鼓励但诚实的语气帮助用户成长。"

# 历史消息最多携带条数
MAX_HISTORY_MESSAGES = 10


def _build_langchain_messages(
    prompt: str,
    system_prompt: Optional[str] = None,
    history: Optional[List[dict]] = None,
) -> list:
    """
    构建 LangChain 消息列表

    Args:
        prompt: 当前用户输入
        system_prompt: RAG 注入的 system prompt
        history: 历史消息 [{"role": "user"|"assistant", "content": "..."}]

    Returns:
        LangChain Message 列表
    """
    messages = []

    # 1. System prompt
    messages.append(SystemMessage(content=system_prompt or DEFAULT_SYSTEM_PROMPT))

    # 2. 历史消息（最多 MAX_HISTORY_MESSAGES 条）
    if history:
        recent_history = history[-MAX_HISTORY_MESSAGES:]
        for msg in recent_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

    # 3. 当前用户消息
    messages.append(HumanMessage(content=prompt))

    return messages


async def stream_response_via_langchain(
    prompt: str,
    system_prompt: Optional[str] = None,
    history: Optional[List[dict]] = None,
) -> AsyncGenerator[str, None]:
    """
    流式调用 DeepSeek

    Args:
        prompt: 当前用户输入
        system_prompt: RAG 生成的 system prompt（含检索上下文）
        history: 历史对话消息列表

    Yields:
        str: 每个 token 的文本内容
    """
    messages = _build_langchain_messages(prompt, system_prompt, history)

    try:
        async for chunk in llm.astream(messages):
            if chunk.content:
                yield chunk.content
    except Exception as e:
        print(f"流式生成出错: {e}")
        yield f"[系统错误: {str(e)}]"


async def generate_title(first_message: str) -> str:
    """
    根据用户第一条消息，生成简短的对话标题（≤10字）
    非流式调用，快速返回
    """
    messages = [
        SystemMessage(content="你是一个标题生成器。根据用户的消息，生成一个简洁的中文对话标题，不超过10个字。只输出标题本身，不要加引号、标点或其他任何内容。"),
        HumanMessage(content=first_message),
    ]

    try:
        result = await llm.ainvoke(messages)
        title = result.content.strip().strip('"\'""''')
        # 确保标题不超过 20 字符（兜底）
        if len(title) > 20:
            title = title[:20]
        return title or "新对话"
    except Exception as e:
        print(f"生成标题失败: {e}")
        return "新对话"

async def generate_session_summary(chat_history: List[dict]) -> str:
    """
    根据历史对话记录，生成用于长期记忆的抽象总结。
    非流式调用，可能会耗时 1-3 秒。
    """
    if not chat_history:
        return ""
        
    chat_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    
    sys_prompt = (
        "你是一个记忆提炼师，负责为个人成长教练系统生成长效记忆。\n"
        "你需要从提供的这通对话流水记录中，提炼出具有高浓度的总结。\n"
        "【总结核心】：\n"
        "1. 遇到了什么核心困惑/问题？\n"
        "2. 最终给出了哪些建设性的认知方案或微行动？\n"
        "3. 这段对话揭示了该用户怎样的个人特质或行为模式？\n"
        "【格式要求】：不用编号，使用平稳的陈述性段落，尽量控制在 200 字以内。只输出摘要本身。"
    )
    
    messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=f"【对话记录如下】：\n{chat_text}"),
    ]

    try:
        result = await llm.ainvoke(messages)
        return result.content.strip()
    except Exception as e:
        print(f"生成记忆摘要失败: {e}")
        return ""