# backend/app/api/endpoints/categories.py
"""
分类 CRUD API
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from app.api.deps import get_current_user
from app.database import get_session
from app.models import Category, ChatSession, User
from app.schemas import CategoryRead, CategoryCreate

router = APIRouter()


@router.get("", response_model=List[CategoryRead], include_in_schema=False)
@router.get("/", response_model=List[CategoryRead])
async def get_categories(
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    statement = (
        select(Category)
        .where(Category.user_id == current_user.id)
        .order_by(Category.id)
    )
    result = await db.exec(statement)
    return result.all()


@router.post("", response_model=CategoryRead, include_in_schema=False)
@router.post("/", response_model=CategoryRead)
async def create_category(
    cat_in: CategoryCreate,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    existing_stmt = select(Category).where(
        Category.user_id == current_user.id,
        Category.name == cat_in.name,
    )
    if (await db.exec(existing_stmt)).first():
        raise HTTPException(status_code=400, detail="分类名称已存在")

    category = Category(
        user_id=current_user.id,
        name=cat_in.name,
        color_code=cat_in.color_code or "#6366f1",
    )
    db.add(category)
    await db.commit()
    await db.refresh(category)
    return category


@router.put("/{category_id}/", response_model=CategoryRead, include_in_schema=False)
@router.put("/{category_id}", response_model=CategoryRead)
async def update_category(
    category_id: int,
    cat_in: CategoryCreate,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    stmt = select(Category).where(
        Category.id == category_id,
        Category.user_id == current_user.id,
    )
    category = (await db.exec(stmt)).first()
    if not category:
        raise HTTPException(status_code=404, detail="分类不存在")

    duplicate_stmt = select(Category).where(
        Category.user_id == current_user.id,
        Category.name == cat_in.name,
        Category.id != category_id,
    )
    if (await db.exec(duplicate_stmt)).first():
        raise HTTPException(status_code=400, detail="分类名称已存在")

    category.name = cat_in.name
    if cat_in.color_code:
        category.color_code = cat_in.color_code
    db.add(category)
    await db.commit()
    await db.refresh(category)
    return category


@router.delete("/{category_id}/", include_in_schema=False)
@router.delete("/{category_id}")
async def delete_category(
    category_id: int,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    stmt = select(Category).where(
        Category.id == category_id,
        Category.user_id == current_user.id,
    )
    category = (await db.exec(stmt)).first()
    if not category:
        raise HTTPException(status_code=404, detail="分类不存在")

    # 将该分类下的会话解除关联
    statement = select(ChatSession).where(
        ChatSession.category_id == category_id,
        ChatSession.user_id == current_user.id,
    )
    result = await db.exec(statement)
    for session in result.all():
        session.category_id = None
        db.add(session)

    await db.delete(category)
    await db.commit()
    return {"ok": True, "message": f"已删除分类「{category.name}」"}