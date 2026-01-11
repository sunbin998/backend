# backend/app/api/endpoints/categories.py
from fastapi import APIRouter, Depends
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from app.database import get_session
from app.models import Category
from app.schemas import CategoryRead # 需在 schemas.py 定义

router = APIRouter()

@router.get("/", response_model=List[CategoryRead])
async def get_categories(db: AsyncSession = Depends(get_session)):
    statement = select(Category)
    result = await db.exec(statement)
    return result.all()