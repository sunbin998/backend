# backend/app/api/endpoints/chat.py
"""
聊天 API（已集成 RAG 管线 + 对话历史上下文）
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
import uuid

from app.services.llm_service import stream_response_via_langchain
from app.services.rag_service import retrieve_relevant_chunks, build_rag_prompt
from app.database import get_session
from app.models import Message, ChatSession
from app.schemas import MessageCreate, MessageRead
from fastapi.responses import StreamingResponse

router = APIRouter()

# 历史消息加载条数
HISTORY_LIMIT = 10


async def _load_history(session_id: uuid.UUID, db: AsyncSession) -> List[dict]:
    """
    从数据库加载当前会话的最近历史消息
    """
    statement = (
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(desc(Message.created_at))
        .limit(HISTORY_LIMIT)
    )
    result = await db.exec(statement)
    messages = result.all()

    # 数据库查询是倒序的，反转为正序
    messages.reverse()

    return [{"role": msg.role, "content": msg.content} for msg in messages]


# 1. 获取某个会话的所有消息
@router.get("/messages", response_model=List[MessageRead])
async def get_messages(
    session_id: uuid.UUID,
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_session),
):
    session = await db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    statement = (
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at)
    )
    result = await db.exec(statement)
    return result.all()


# 2. 辅助函数：流式生成器（处理流 + 存库）
async def generate_and_save(
    session_id: uuid.UUID,
    prompt: str,
    db: AsyncSession,
    system_prompt: str,
    history: List[dict],
):
    full_response = ""

    # 1. 开始流式生成（传入 RAG prompt + 历史消息）
    async for token in stream_response_via_langchain(
        prompt=prompt,
        system_prompt=system_prompt,
        history=history,
    ):
        full_response += token
        # SSE 格式：data: <content>\n\n
        yield f"data: {token}\n\n"

    # 2. 流结束，发送结束信号
    yield "data: [DONE]\n\n"

    # 3. 生成结束后，异步存库
    print(f"流式结束，完整回复长度: {len(full_response)} 字符")

    try:
        ai_msg = Message(
            session_id=session_id,
            role="assistant",
            content=full_response,
        )
        db.add(ai_msg)

        # 更新会话时间
        session = await db.get(ChatSession, session_id)
        if session:
            session.updated_at = ai_msg.created_at
            db.add(session)

        await db.commit()
    except Exception as e:
        print(f"存库失败: {e}")


# 3. 流式发送消息（主要接口，已集成 RAG）
@router.post("/messages/stream")
async def send_message_stream(
    message_in: MessageCreate,
    db: AsyncSession = Depends(get_session),
):
    # 1. 先保存用户消息
    user_msg = Message(
        session_id=message_in.session_id,
        role="user",
        content=message_in.content,
    )
    db.add(user_msg)
    await db.commit()
    await db.refresh(user_msg)

    # 2. 加载历史消息（不含刚保存的用户消息，避免重复）
    history = await _load_history(message_in.session_id, db)
    # 移除最后一条（刚保存的用户消息），因为 prompt 参数已经包含它
    if history and history[-1]["role"] == "user" and history[-1]["content"] == message_in.content:
        history = history[:-1]

    # 3. RAG 检索：基于用户问题检索相关文档
    try:
        relevant_chunks = await retrieve_relevant_chunks(message_in.content, db)
        system_prompt = build_rag_prompt(relevant_chunks)
        if relevant_chunks:
            print(f"RAG 检索到 {len(relevant_chunks)} 个相关片段")
        else:
            print("RAG 未检索到相关内容，使用 fallback prompt")
    except Exception as e:
        print(f"RAG 检索失败，降级为无检索模式: {e}")
        system_prompt = None

    # 4. 返回流式响应
    return StreamingResponse(
        generate_and_save(message_in.session_id, message_in.content, db, system_prompt, history),
        media_type="text/event-stream",
    )