# backend/app/api/endpoints/chat.py
"""
聊天 API（已集成 RAG 管线 + 对话历史上下文 + 自动标题生成）
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
import uuid
import json
import asyncio

from app.services.llm_service import stream_response_via_langchain, generate_title
from app.services.rag_service import retrieve_relevant_chunks, build_rag_prompt, extract_sources
from app.database import get_session
from app.models import Message, ChatSession, DiaryEntry
from app.schemas import MessageCreate, MessageRead
from fastapi.responses import StreamingResponse
from datetime import date

router = APIRouter()

# 历史消息加载条数
HISTORY_LIMIT = 10


async def _load_history(session_id: uuid.UUID, db: AsyncSession) -> List[dict]:
    """从数据库加载当前会话的最近历史消息"""
    statement = (
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(desc(Message.created_at))
        .limit(HISTORY_LIMIT)
    )
    result = await db.exec(statement)
    messages = result.all()
    messages.reverse()
    return [{"role": msg.role, "content": msg.content} for msg in messages]


async def _is_first_message(session_id: uuid.UUID, db: AsyncSession) -> bool:
    """判断是否为该会话的第一条用户消息"""
    statement = (
        select(Message)
        .where(Message.session_id == session_id, Message.role == "user")
    )
    result = await db.exec(statement)
    messages = result.all()
    return len(messages) <= 1  # 刚保存了当前消息，所以 <=1


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
    sources: List[dict],
    is_first: bool,
):
    full_response = ""

    # 1. 先发送 sources 事件（如果有检索结果）
    if sources:
        yield f"event: sources\ndata: {json.dumps(sources, ensure_ascii=False)}\n\n"

    # 2. 流式生成 AI 回复
    async for token in stream_response_via_langchain(
        prompt=prompt,
        system_prompt=system_prompt,
        history=history,
    ):
        full_response += token
        # SSE 规范：多行 data 需要每行加 data: 前缀，因此用 JSON 编码保留换行符
        escaped = json.dumps(token, ensure_ascii=False)
        yield f"data: {escaped}\n\n"

    # 3. 流结束信号
    yield "data: [DONE]\n\n"

    # 4. 存库
    print(f"流式结束，完整回复长度: {len(full_response)} 字符")

    try:
        ai_msg = Message(
            session_id=session_id,
            role="assistant",
            content=full_response,
            sources=sources if sources else None,
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

    # 5. 如果是第一条消息，异步生成标题
    if is_first:
        try:
            title = await generate_title(prompt)
            print(f"自动生成标题: {title}")

            session = await db.get(ChatSession, session_id)
            if session:
                session.title = title
                db.add(session)
                await db.commit()

            # 通过 SSE 通知前端更新标题
            yield f"event: title\ndata: {title}\n\n"
        except Exception as e:
            print(f"生成标题失败: {e}")


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

    # 2. 判断是否为第一条消息（用于自动生成标题）
    is_first = await _is_first_message(message_in.session_id, db)

    # 3. 加载历史消息
    history = await _load_history(message_in.session_id, db)
    if history and history[-1]["role"] == "user" and history[-1]["content"] == message_in.content:
        history = history[:-1]

    # 4. RAG 检索 + 当天日记注入
    sources = []
    try:
        # 4a. 查询当天日记
        today_str = date.today().isoformat()
        diary_statement = select(DiaryEntry).where(DiaryEntry.date == today_str)
        diary_result = await db.exec(diary_statement)
        today_diary_entry = diary_result.first()

        today_diary = today_diary_entry.content if today_diary_entry else None
        today_mood = today_diary_entry.mood if today_diary_entry else None

        if today_diary:
            print(f"已注入当天日记 ({today_str}), 长度: {len(today_diary)} 字")

        # 4b. 向量检索参考资料
        relevant_chunks = await retrieve_relevant_chunks(message_in.content, db)
        sources = extract_sources(relevant_chunks)

        # 4c. 构建完整 System Prompt
        system_prompt = build_rag_prompt(
            relevant_chunks,
            today_diary=today_diary,
            today_date=today_str,
            today_mood=today_mood,
        )

        if relevant_chunks:
            print(f"RAG 检索到 {len(relevant_chunks)} 个相关片段")
        else:
            print("RAG 未检索到相关内容")
    except Exception as e:
        print(f"RAG 检索失败，降级为无检索模式: {e}")
        await db.rollback()
        system_prompt = None

    # 5. 返回流式响应
    return StreamingResponse(
        generate_and_save(
            message_in.session_id, message_in.content, db,
            system_prompt, history, sources, is_first,
        ),
        media_type="text/event-stream",
    )