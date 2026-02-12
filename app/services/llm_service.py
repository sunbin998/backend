# backend/app/services/llm_service.py
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage
from typing import AsyncGenerator
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatTongyi(
    model="qwen-turbo",
    temperature=0.7,
    streaming=True, # <--- 关键开启
)

async def stream_response_via_langchain(prompt: str) -> AsyncGenerator[str, None]:
    """
    流式调用通义千问
    Yields:
        str: 每个 token 的文本内容
    """
    messages = [
        SystemMessage(content="你是一个残酷诚实、技术精湛的 AGI 开发顾问。"),
        HumanMessage(content=prompt)
    ]
    
    # 使用 astream 异步流式获取
    try:
        async for chunk in llm.astream(messages):
            if chunk.content:
                yield chunk.content
    except Exception as e:
        print(f"流式生成出错: {e}")
        yield f"[系统错误: {str(e)}]"