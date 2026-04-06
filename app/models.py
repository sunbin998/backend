# app/models.py
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from sqlmodel import Field, Relationship, SQLModel
from sqlalchemy import Column, DateTime, func, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB  # 用于存储 JSON
from pgvector.sqlalchemy import Vector  # 用于存储向量


class User(SQLModel, table=True):
    __tablename__ = "users"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    username: str = Field(unique=True, index=True, max_length=50)
    email: Optional[str] = Field(default=None, unique=True, max_length=100)
    hashed_password: str = Field(max_length=255)
    avatar: Optional[str] = Field(default=None, max_length=500)

    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )
    updated_at: datetime = Field(
        sa_column=Column(
            DateTime(timezone=True),
            server_default=func.now(),
            onupdate=func.now(),
        )
    )

# ==========================================
# 1. 分类模型 (Category)
# ==========================================
class Category(SQLModel, table=True):
    __tablename__ = "categories" # 显式指定表名，好习惯
    __table_args__ = (
        UniqueConstraint("user_id", "name", name="uq_categories_user_name"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[uuid.UUID] = Field(default=None, foreign_key="users.id", index=True)
    # 分类名称（用户维度唯一）
    name: str = Field(index=True, max_length=50)
    # 用于前端展示的颜色 (如 #FF0000)
    color_code: str = Field(default="#6366f1", max_length=7)
    
    # 创建时间，默认当前时间
    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )

    # 关系：一个分类下有多个会话
    sessions: List["ChatSession"] = Relationship(back_populates="category")


# ==========================================
# 2. 会话模型 (ChatSession)
# ==========================================
class ChatSession(SQLModel, table=True):
    __tablename__ = "sessions"

    # 使用 UUID 作为主键，比自增 ID 更适合分布式和 URL 里的 ID 隐藏
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: Optional[uuid.UUID] = Field(default=None, foreign_key="users.id", index=True)
    
    title: str = Field(default="新对话", max_length=255)
    
    # 摘要：未来由 LLM 生成，用于快速了解对话内容
    summary: Optional[str] = Field(default=None, sa_column_kwargs={"nullable": True})
    
    # 置顶功能：方便用户找到重要的历史
    is_pinned: bool = Field(default=False)
    
    # 外键：关联到 Category 表
    category_id: Optional[int] = Field(default=None, foreign_key="categories.id")
    
    # 时间戳管理
    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )
    # 更新时间：每次修改自动更新
    updated_at: datetime = Field(
        sa_column=Column(
            DateTime(timezone=True), 
            server_default=func.now(), 
            onupdate=func.now()
        )
    )

    # 关系定义
    category: Optional[Category] = Relationship(back_populates="sessions")
    messages: List["Message"] = Relationship(back_populates="session", sa_relationship_kwargs={"cascade": "all, delete"})


# ==========================================
# 3. 消息模型 (Message)
# ==========================================
class Message(SQLModel, table=True):
    __tablename__ = "messages"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    
    # 关联到哪个会话
    session_id: uuid.UUID = Field(foreign_key="sessions.id")
    
    # 角色：user (用户), assistant (AI), system (系统指令)
    role: str = Field(max_length=20)
    
    # 消息的具体文本内容
    content: str

    # RAG 参考来源（仅 assistant 消息有值）
    sources: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        sa_column=Column("sources", JSONB)
    )
    
    # Embedding 向量字段
    embedding: Optional[List[float]] = Field(
        default=None,
        sa_column=Column(Vector(1536))
    )
    
    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )

    # 关系：属于哪个会话
    session: ChatSession = Relationship(back_populates="messages")


# ==========================================
# 4. 知识库文档模型 (Document - RAG 专用)
# ==========================================
class Document(SQLModel, table=True):
    __tablename__ = "documents"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[uuid.UUID] = Field(default=None, foreign_key="users.id", index=True)
    
    # 文档切片后的具体内容
    content: str
    
    # 【核心修正】元数据 (JSON)
    # 存储如 {"filename": "Paper.pdf", "page": 12, "author": "..."}
    # 使用 PostgreSQL 的 JSONB 类型，支持高效的 JSON 查询
    metadata_: Dict[str, Any] = Field(
        default={},
        sa_column=Column("metadata", JSONB) # 数据库里叫 metadata，Python 里为了避开关键字用 metadata_
    )
    
    # 向量索引 (1024 维，匹配 DashScope text-embedding-v3)
    embedding: Optional[List[float]] = Field(
        default=None,
        sa_column=Column(Vector(1024))
    )


# ==========================================
# 5. 日记模型 (DiaryEntry)
# ==========================================
class DiaryEntry(SQLModel, table=True):
    __tablename__ = "diary_entries"
    __table_args__ = (
        UniqueConstraint("user_id", "date", name="uq_diary_entries_user_date"),
    )

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: Optional[uuid.UUID] = Field(default=None, foreign_key="users.id", index=True)

    # 日期（用户维度唯一）
    date: str = Field(index=True, max_length=10)  # 格式: "2026-02-14"

    # 日记正文
    content: str

    # 心情标签（可选）
    mood: Optional[str] = Field(default=None, max_length=20)  # 如: 开心/焦虑/平静/疲惫

    # 标签（JSON 数组，可选）
    tags: Optional[Dict[str, Any]] = Field(
        default=None,
        sa_column=Column("tags", JSONB)
    )

    # 是否已向量化入库
    is_vectorized: bool = Field(default=False)

    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )
    updated_at: datetime = Field(
        sa_column=Column(
            DateTime(timezone=True),
            server_default=func.now(),
            onupdate=func.now()
        )
    )