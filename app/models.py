# app/models.py
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from sqlmodel import Field, Relationship, SQLModel
from sqlalchemy import Column, DateTime, func
from sqlalchemy.dialects.postgresql import JSONB  # 用于存储 JSON
from pgvector.sqlalchemy import Vector  # 用于存储向量

# ==========================================
# 1. 分类模型 (Category)
# ==========================================
class Category(SQLModel, table=True):
    __tablename__ = "categories" # 显式指定表名，好习惯

    id: Optional[int] = Field(default=None, primary_key=True)
    # 唯一且必填的分类名称
    name: str = Field(unique=True, index=True, max_length=50)
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
    
    # 【核心修正】Embedding 向量字段
    # 使用 pgvector 的 Vector 类型，维度 1536 (对应 OpenAI text-embedding-3-small)
    # 注意：在没有安装 vector 扩展的库里这会报错，但我们已经在 requirement 里装了
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