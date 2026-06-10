# app/api/endpoints/sessions.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select, desc, or_
from typing import List, Optional
import random

from app.api.deps import get_current_user
from app.database import get_session
from app.models import ChatSession, Category, User, Message
from app.schemas import SessionCreate, SessionRead

router = APIRouter()

# 新建会话时自动发送首条 assistant 欢迎语（引用积极短句）
WELCOME_QUOTES = [
    ("世界上只有一种真正的英雄主义，就是在认清生活的真相后依然热爱生活。", "罗曼·罗兰"),
    ("长风破浪会有时，直挂云帆济沧海。", "李白"),
    ("既然选择了远方，便只顾风雨兼程。", "汪国真"),
    ("我们终将上岸，阳光万里。", "《摆渡人》"),
    ("不要慨叹生活底痛苦，慨叹是弱者。", "高尔基"),
]


def _build_welcome_message(username: Optional[str]) -> str:
    """生成新会话欢迎语：积极引用 + 用户名问候。"""
    user_name = (username or "朋友").strip() or "朋友"
    quote, author = random.choice(WELCOME_QUOTES)
    return (
        f"\"{quote}\" —— {author}\n\n"
        f"嗨，{user_name}，很高兴见到你。"
        "你可以从今天最想聊的一件小事开始，我会认真听你说。"
    )

# 1. 创建新会话
@router.post("", response_model=SessionRead, include_in_schema=False)
@router.post("/", response_model=SessionRead)
async def create_session(
    session_in: SessionCreate,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    # 如果传了 category_id，先校验存在性
    if session_in.category_id:
        statement = select(Category).where(
            Category.id == session_in.category_id,
            Category.user_id == current_user.id,
        )
        category = (await db.exec(statement)).first()
        if not category:
            raise HTTPException(status_code=404, detail="分类不存在")

    # 创建数据库实例
    db_session = ChatSession(
        title=session_in.title,
        category_id=session_in.category_id,
        user_id=current_user.id,
    )

    # 同步创建 assistant 首条欢迎语，确保新会话打开即可看到问候
    welcome_msg = Message(
        session_id=db_session.id,
        role="assistant",
        content=_build_welcome_message(current_user.username),
    )

    db.add(db_session)
    db.add(welcome_msg)
    await db.commit()
    await db.refresh(db_session)
    return db_session

# 2. 获取会话列表 (支持 搜索 + 分类筛选 + 分页)
@router.get("", response_model=List[SessionRead], include_in_schema=False)
@router.get("/", response_model=List[SessionRead])
async def get_sessions(
    skip: int = 0,
    limit: int = 20,
    keyword: Optional[str] = None,
    category_id: Optional[int] = None,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    # 构建查询
    query = select(ChatSession).where(ChatSession.user_id == current_user.id)
    
    # 筛选：分类
    if category_id:
        query = query.where(ChatSession.category_id == category_id)
    
    # 筛选：关键词 (搜索标题或聊天内容)
    if keyword:
        from app.models import Message
        query = query.outerjoin(Message, ChatSession.id == Message.session_id)
        query = query.where(
            or_(
                ChatSession.title.ilike(f"%{keyword}%"),
                ChatSession.summary.ilike(f"%{keyword}%"),
                Message.content.ilike(f"%{keyword}%")
            )
        ).distinct()
    
    # 排序：置顶优先，然后按创建时间倒序
    query = query.order_by(desc(ChatSession.is_pinned), desc(ChatSession.created_at))
    
    # 分页
    query = query.offset(skip).limit(limit)
    
    result = await db.exec(query)
    return result.all()

# 3. 删除会话
@router.delete("/{session_id}/", include_in_schema=False)
@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    statement = select(ChatSession).where(
        ChatSession.id == session_id,
        ChatSession.user_id == current_user.id,
    )
    session = (await db.exec(statement)).first()
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    # NEW: 清理关联的长效记忆向量数据
    from sqlalchemy import text
    await db.execute(
        text(
            "DELETE FROM documents "
            "WHERE user_id = CAST(:user_id AS uuid) "
            "AND metadata->>'source_type' = 'chat_memory' "
            "AND metadata->>'session_id' = :session_id"
        ),
        {"session_id": session_id, "user_id": str(current_user.id)},
    )

    await db.delete(session)
    await db.commit()
    return {"ok": True}


# 4. 更新会话（分类、标题等）
from pydantic import BaseModel


class SessionUpdate(BaseModel):
    category_id: Optional[int] = None
    title: Optional[str] = None
    clear_category: bool = False  # 专门用于清除分类


@router.patch("/{session_id}", response_model=SessionRead)
@router.patch("/{session_id}/", response_model=SessionRead, include_in_schema=False)
async def update_session(
    session_id: str,
    update_in: SessionUpdate,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    statement = select(ChatSession).where(
        ChatSession.id == session_id,
        ChatSession.user_id == current_user.id,
    )
    session = (await db.exec(statement)).first()
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    if update_in.clear_category:
        session.category_id = None
    elif update_in.category_id is not None:
        # 验证分类存在
        category_stmt = select(Category).where(
            Category.id == update_in.category_id,
            Category.user_id == current_user.id,
        )
        category = (await db.exec(category_stmt)).first()
        if not category:
            raise HTTPException(status_code=404, detail="分类不存在")
        session.category_id = update_in.category_id

    if update_in.title is not None:
        session.title = update_in.title

    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session