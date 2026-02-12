# backend/app/services/llm_service.py
"""
LLM 调用服务（通义千问 qwen-turbo，流式）
支持：
- 自定义 system prompt（来自 RAG 检索结果）
- 对话历史上下文
"""
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import AsyncGenerator, List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatTongyi(
    model="qwen-turbo",
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
    流式调用通义千问

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