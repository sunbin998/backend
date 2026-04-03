# app/schemas.py
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import uuid

# --- Session 相关 ---
class SessionCreate(BaseModel):
    category_id: Optional[int] = None # 允许不选分类
    title: str = "新对话"

class SessionRead(BaseModel):
    id: uuid.UUID
    title: str
    summary: Optional[str]
    is_pinned: bool
    category_id: Optional[int]
    created_at: datetime
    updated_at: datetime
    # 这里我们暂不返回 messages，因为列表页不需要加载详情

    class Config:
        from_attributes = True # 让 Pydantic 能读取 SQLModel 对象

# --- Category 相关 (简单定义) ---
class CategoryCreate(BaseModel):
    name: str
    color_code: Optional[str] = None

class CategoryRead(BaseModel):
    id: int
    name: str
    color_code: str
    
    class Config:
        from_attributes = True
    
class MessageCreate(BaseModel):
    session_id: uuid.UUID
    content: str
    book_filter: Optional[List[str]] = None  # 书籍过滤：为空=全部，["书名.epub"]=指定书

class MessageRead(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    sources: Optional[list] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

# --- Diary 相关 ---
class DiaryCreate(BaseModel):
    date: str  # "2026-02-14"
    content: str
    mood: Optional[str] = None
    tags: Optional[list] = None

class DiaryRead(BaseModel):
    id: uuid.UUID
    date: str
    content: str
    mood: Optional[str]
    tags: Optional[list]
    is_vectorized: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True