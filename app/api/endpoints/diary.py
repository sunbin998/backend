# backend/app/api/endpoints/diary.py
"""
日记 CRUD API + 自动向量化入库
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select, desc
from typing import List, Optional
import asyncio

from app.database import get_session
from app.models import DiaryEntry, Document
from app.schemas import DiaryCreate, DiaryRead
from app.services.embedding_service import embed_texts

from langchain_text_splitters import RecursiveCharacterTextSplitter

router = APIRouter()

# 切片配置
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


async def _vectorize_diary(diary: DiaryEntry, db: AsyncSession):
    """
    将日记内容切片 + Embedding + 写入 documents 表
    """
    # 1. 先清除该日记的旧向量数据
    from sqlalchemy import text
    await db.execute(
        text("DELETE FROM documents WHERE metadata->>'source_type' = 'diary' AND metadata->>'diary_date' = :date"),
        {"date": diary.date},
    )

    # 2. 切片
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(diary.content)

    if not chunks:
        return

    # 3. Embedding（同步调用，在后台线程执行）
    embeddings = await asyncio.to_thread(embed_texts, chunks)

    # 4. 批量写入 documents 表
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        doc = Document(
            content=chunk,
            metadata_={
                "source_type": "diary",
                "diary_date": diary.date,
                "mood": diary.mood or "",
                "chunk_index": i,
                "total_chunks": len(chunks),
            },
            embedding=embedding,
        )
        db.add(doc)

    # 5. 标记已向量化
    diary.is_vectorized = True
    db.add(diary)
    await db.commit()
    await db.refresh(diary)

    print(f"日记 {diary.date} 已向量化: {len(chunks)} 个切片")


# 1. 创建/更新日记（按日期 upsert）
@router.post("", response_model=DiaryRead)
async def upsert_diary(
    diary_in: DiaryCreate,
    db: AsyncSession = Depends(get_session),
):
    """创建或更新指定日期的日记"""
    # 查找是否已存在该日期的日记
    statement = select(DiaryEntry).where(DiaryEntry.date == diary_in.date)
    result = await db.exec(statement)
    existing = result.first()

    if existing:
        # 更新
        existing.content = diary_in.content
        existing.mood = diary_in.mood
        existing.tags = diary_in.tags
        existing.is_vectorized = False  # 内容变了，需要重新向量化
        db.add(existing)
        await db.commit()
        await db.refresh(existing)
        diary = existing
    else:
        # 创建
        diary = DiaryEntry(
            date=diary_in.date,
            content=diary_in.content,
            mood=diary_in.mood,
            tags=diary_in.tags,
        )
        db.add(diary)
        await db.commit()
        await db.refresh(diary)

    # 异步向量化（不阻塞响应）
    try:
        await _vectorize_diary(diary, db)
    except Exception as e:
        print(f"日记向量化失败: {e}")
        # 向量化失败不影响日记保存

    return diary


# 2. 获取日记列表
@router.get("", response_model=List[DiaryRead])
async def list_diaries(
    month: Optional[str] = None,  # 格式: "2026-02"
    limit: int = 30,
    db: AsyncSession = Depends(get_session),
):
    """获取日记列表，支持按月筛选"""
    query = select(DiaryEntry)

    if month:
        query = query.where(DiaryEntry.date.startswith(month))

    query = query.order_by(desc(DiaryEntry.date)).limit(limit)

    result = await db.exec(query)
    return result.all()


# 3. 获取指定日期日记
@router.get("/{date}", response_model=Optional[DiaryRead])
async def get_diary(
    date: str,
    db: AsyncSession = Depends(get_session),
):
    """获取指定日期的日记"""
    statement = select(DiaryEntry).where(DiaryEntry.date == date)
    result = await db.exec(statement)
    diary = result.first()

    if not diary:
        return None

    return diary


# 4. 删除日记
@router.delete("/{date}")
async def delete_diary(
    date: str,
    db: AsyncSession = Depends(get_session),
):
    """删除指定日期的日记"""
    statement = select(DiaryEntry).where(DiaryEntry.date == date)
    result = await db.exec(statement)
    diary = result.first()

    if not diary:
        raise HTTPException(status_code=404, detail="该日期没有日记")

    # 同时删除关联的 documents
    from sqlalchemy import text
    await db.execute(
        text("DELETE FROM documents WHERE metadata->>'source_type' = 'diary' AND metadata->>'diary_date' = :date"),
        {"date": date},
    )

    await db.delete(diary)
    await db.commit()

    return {"ok": True, "message": f"已删除 {date} 的日记"}
