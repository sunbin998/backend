# backend/app/api/endpoints/chat.py
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
import uuid
from app.services.llm_service import stream_response_via_langchain
from app.database import get_session
from app.models import Message, ChatSession
from app.schemas import MessageCreate, MessageRead # 稍后在 schemas.py 补充定义
from fastapi.responses import StreamingResponse # <--- 引入流式响应

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
    # 1. 保存用户消息 (User Input)
    user_msg = Message(
        session_id=message_in.session_id,
        role="user",
        content=message_in.content
    )
    db.add(user_msg)
    # 先提交一次，确保用户消息入库
    await db.commit() 
    await db.refresh(user_msg)

    # =====================================================
    # 2. 真实调用 AI (通过 LangChain)
    # =====================================================
    # 盲点警告：你现在传给大模型的只有 message_in.content (当前问题)
    # 大模型并不知道这个 session_id 之前聊过什么。
    ai_content = generate_response_via_langchain(prompt=message_in.content)
    
    # 3. 保存 AI 回复
    ai_msg = Message(
        session_id=message_in.session_id,
        role="assistant",
        content=ai_content
    )
    db.add(ai_msg)
    
    # 4. 更新会话时间 (用于列表排序)
    session = await db.get(ChatSession, message_in.session_id)
    if session:
        session.updated_at = ai_msg.created_at
        db.add(session)
        
    await db.commit()
    await db.refresh(ai_msg)
    
    return ai_msg

# 辅助函数：生成器包装器 (处理流 + 存库)
async def generate_and_save(session_id: uuid.UUID, prompt: str, db: AsyncSession):
    full_response = ""
    
    # 1. 开始流式生成
    async for token in stream_response_via_langchain(prompt):
        full_response += token
        # SSE 格式：data: <content>\n\n
        yield f"data: {token}\n\n"
    
    # 2. 流结束，发送结束信号
    yield "data: [DONE]\n\n"
    
    # 3. 【关键】生成结束后，在这里异步存库
    # 注意：因为 Response 已经开始发送，我们必须小心处理 DB 会话
    print(f"流式结束，完整回复: {full_response[:20]}...")
    
    try:
        # 新建一个 AI 消息记录
        ai_msg = Message(
            session_id=session_id,
            role="assistant",
            content=full_response
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

@router.post("/messages/stream") # <--- 新的流式接口路径
async def send_message_stream(
    message_in: MessageCreate,
    db: AsyncSession = Depends(get_session)
):
    # 1. 先保存用户消息 (User) - 这一步必须在开始流之前完成
    user_msg = Message(
        session_id=message_in.session_id,
        role="user",
        content=message_in.content
    )
    db.add(user_msg)
    await db.commit()
    await db.refresh(user_msg)
    
    # 2. 返回流式响应
    return StreamingResponse(
        generate_and_save(message_in.session_id, message_in.content, db),
        media_type="text/event-stream"
    )