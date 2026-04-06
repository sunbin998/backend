# backend/app/api/endpoints/documents.py
"""
文档管理 API
- 上传文件（PDF/EPUB/MOBI/AZW/TXT/MD）
- 查看已上传文档列表
- 删除文档
"""
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import List

from app.api.deps import get_current_user
from app.database import get_session
from app.models import User
from app.services.document_service import process_uploaded_file, SUPPORTED_EXTENSIONS

router = APIRouter()

# 最大上传大小限制 (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024


@router.post("/", include_in_schema=False)
@router.post("")
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    上传文件并自动处理：解析 → 切片 → Embedding → 入库
    支持格式: PDF, EPUB, MOBI, AZW, AZW3, TXT, MD
    """
    # 1. 校验文件名
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    # 2. 校验文件扩展名
    import os
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {ext}。支持: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    # 3. 校验文件大小（读取前先检查 content_length）
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大 ({file.size / 1024 / 1024:.1f}MB)。最大限制: 10MB",
        )

    # 4. 调用处理管线
    try:
        result = await process_uploaded_file(file, db, current_user.id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件处理失败: {str(e)}")


@router.get("/", include_in_schema=False)
@router.get("")
async def list_documents(
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    获取已上传文档列表（按文件名聚合）
    返回每个文件的切片数量和上传时间
    """
    sql = text("""
        SELECT
            metadata->>'filename' AS filename,
            COUNT(*) AS chunk_count,
            MIN(id) AS first_id
        FROM documents
        WHERE user_id = CAST(:user_id AS uuid)
          AND metadata->>'filename' IS NOT NULL
        GROUP BY metadata->>'filename'
        ORDER BY MIN(id) DESC
    """)
    result = await db.execute(sql, {"user_id": str(current_user.id)})
    rows = result.fetchall()

    return [
        {
            "filename": row.filename,
            "chunk_count": row.chunk_count,
        }
        for row in rows
    ]


@router.delete("/{filename}/", include_in_schema=False)
@router.delete("/{filename}")
async def delete_document(
    filename: str,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    删除指定文件名的所有切片
    """
    sql = text("""
        DELETE FROM documents
        WHERE user_id = CAST(:user_id AS uuid)
          AND metadata->>'filename' = :filename
    """)
    result = await db.execute(
        sql,
        {"filename": filename, "user_id": str(current_user.id)},
    )
    await db.commit()

    deleted_count = result.rowcount
    if deleted_count == 0:
        raise HTTPException(status_code=404, detail=f"未找到文件: {filename}")

    return {
        "filename": filename,
        "deleted_chunks": deleted_count,
        "status": "success",
    }
