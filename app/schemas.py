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
class CategoryRead(BaseModel):
    id: int
    name: str
    color_code: str