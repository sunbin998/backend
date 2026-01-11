# backend/app/api/endpoints/chat.py
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
import uuid

from app.database import get_session
from app.models import Message, ChatSession
from app.schemas import MessageCreate, MessageRead # 稍后在 schemas.py 补充定义

router = APIRouter()

# 1. 获取某个会话的所有消息
@router.get("/messages", response_model=List[MessageRead])
async def get_messages(
    session_id: uuid.UUID,
    skip: int = 0,
    limit: int = 100, # 限制单次加载数量，防止炸内存
    db: AsyncSession = Depends(get_session)
):
    # 验证会话是否存在
    session = await db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    # 查询消息，按时间正序排列（旧的在上，新的在下）
    statement = select(Message).where(Message.session_id == session_id).order_by(Message.created_at)
    result = await db.exec(statement)
    return result.all()

# 2. 发送消息 (User -> Assistant)
@router.post("/messages", response_model=MessageRead)
async def send_message(
    message_in: MessageCreate,
    db: AsyncSession = Depends(get_session)
):
    # 1. 保存用户消息
    user_msg = Message(
        session_id=message_in.session_id,
        role="user",
        content=message_in.content
    )
    db.add(user_msg)
    
    # 2. (Mock) AI 回复逻辑
    # 真正接入模型前，我们先假装思考了 1 秒，然后复读机
    # 只要这个逻辑通了，后面把这里换成 LangChain 调用即可
    ai_msg = Message(
        session_id=message_in.session_id,
        role="assistant",
        content=f"【Mock回复】我收到了你的消息：{message_in.content}"
    )
    db.add(ai_msg)
    
    # 3. 更新会话的 updated_at (让它浮到列表顶部)
    session = await db.get(ChatSession, message_in.session_id)
    if session:
        session.updated_at = ai_msg.created_at
        db.add(session)
        
    await db.commit()
    await db.refresh(user_msg)
    await db.refresh(ai_msg)
    
    # 注意：这里我们返回的是 AI 的回复，因为前端已经乐观展示了用户的消息
    # 实际生产中通常会用 WebSocket 推送，但 HTTP 轮询/返回最简单
    return ai_msg