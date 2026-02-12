# backend/app/services/document_service.py
"""
文档处理管线：解析 → 切片 → Embedding → 入库
支持格式：PDF, EPUB, MOBI, AZW, TXT, MD
"""
import os
import tempfile
import shutil
from typing import List, Dict, Any

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models import Document
from app.services.embedding_service import embed_texts

# 支持的文件扩展名
SUPPORTED_EXTENSIONS = {".pdf", ".epub", ".mobi", ".azw", ".azw3", ".txt", ".md"}

# 切片配置
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
)


def _get_file_extension(filename: str) -> str:
    """获取小写文件扩展名"""
    return os.path.splitext(filename)[1].lower()


def _extract_text_from_pdf(file_path: str) -> str:
    """从 PDF 提取文本"""
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    return "\n\n".join(text_parts)


def _extract_text_from_epub(file_path: str) -> str:
    """从 EPUB 提取文本"""
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup

    book = epub.read_epub(file_path, options={"ignore_ncx": True})
    text_parts = []

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        content = item.get_content()
        soup = BeautifulSoup(content, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        if text.strip():
            text_parts.append(text)

    return "\n\n".join(text_parts)


def _extract_text_from_mobi(file_path: str) -> str:
    """
    从 MOBI/AZW 提取文本
    策略：用 mobi 库解包为临时目录，找到其中的 EPUB/HTML 文件后提取
    """
    import mobi

    # mobi.extract 会解包到临时目录
    tempdir, _ = mobi.extract(file_path)

    text_parts = []
    try:
        # 遍历解压目录，寻找 HTML/EPUB 文件
        for root, dirs, files in os.walk(tempdir):
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                ext = _get_file_extension(fname)

                if ext == ".epub":
                    text_parts.append(_extract_text_from_epub(fpath))
                elif ext in (".html", ".htm", ".xhtml"):
                    from bs4 import BeautifulSoup

                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        soup = BeautifulSoup(f.read(), "html.parser")
                        text = soup.get_text(separator="\n", strip=True)
                        if text.strip():
                            text_parts.append(text)
    finally:
        # 清理临时目录
        if os.path.isdir(tempdir):
            shutil.rmtree(tempdir, ignore_errors=True)

    return "\n\n".join(text_parts)


def _extract_text_from_txt(file_path: str) -> str:
    """从 TXT/MD 提取文本"""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_text(file_path: str, extension: str) -> str:
    """根据文件扩展名，调用对应的解析器提取文本"""
    if extension == ".pdf":
        return _extract_text_from_pdf(file_path)
    elif extension == ".epub":
        return _extract_text_from_epub(file_path)
    elif extension in (".mobi", ".azw", ".azw3"):
        return _extract_text_from_mobi(file_path)
    elif extension in (".txt", ".md"):
        return _extract_text_from_txt(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {extension}")


async def process_uploaded_file(
    file: UploadFile, db: AsyncSession
) -> Dict[str, Any]:
    """
    完整的文件处理管线：
    1. 保存上传文件到临时目录
    2. 提取文本
    3. 切片
    4. 生成 Embedding
    5. 批量写入数据库

    Returns:
        {"filename": str, "chunks_created": int, "status": str}
    """
    filename = file.filename or "unknown"
    extension = _get_file_extension(filename)

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"不支持的文件格式: {extension}。"
            f"支持的格式: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    # 1. 保存到临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # 2. 提取文本
        full_text = extract_text(tmp_path, extension)

        if not full_text.strip():
            return {
                "filename": filename,
                "chunks_created": 0,
                "status": "empty",
                "message": "文件内容为空，未创建任何切片",
            }

        # 3. 切片
        chunks = TEXT_SPLITTER.split_text(full_text)

        if not chunks:
            return {
                "filename": filename,
                "chunks_created": 0,
                "status": "empty",
                "message": "切片后无内容",
            }

        # 4. 生成 Embedding（Ollama bge-m3 支持批量输入）
        embeddings = embed_texts(chunks)

        # 5. 批量入库
        documents = []
        for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            doc = Document(
                content=chunk_text,
                metadata_={
                    "filename": filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
                embedding=embedding,
            )
            documents.append(doc)
            db.add(doc)

        await db.commit()

        return {
            "filename": filename,
            "chunks_created": len(documents),
            "status": "success",
            "message": f"成功处理文件：{len(chunks)} 个切片已入库",
        }

    finally:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
