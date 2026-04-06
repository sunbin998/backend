# backend/app/services/document_service.py
"""
文档处理管线：解析 → 切片 → Embedding → 入库
支持格式：PDF, EPUB, MOBI, AZW, TXT, MD

切块策略（v2 - 结构化 + 语义两级切块）：
  EPUB: HTMLHeaderTextSplitter（按章节标题切） → SemanticChunker（语义精切）
  PDF/TXT/MD: SemanticChunker（语义切块）+ RecursiveCharacterTextSplitter（兜底）
"""
import os
import tempfile
import shutil
import uuid
from typing import List, Dict, Any, Tuple

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    HTMLHeaderTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import OllamaEmbeddings

from app.models import Document
from app.services.embedding_service import embed_texts

# 支持的文件扩展名
SUPPORTED_EXTENSIONS = {".pdf", ".epub", ".mobi", ".azw", ".azw3", ".txt", ".md"}

# SemanticChunker 使用的 Embedding 模型（复用本地 Ollama bge-m3）
_ollama_embeddings = OllamaEmbeddings(
    model="bge-m3",
    base_url="http://localhost:11434",
)

# HTML 标题层级映射（用于 EPUB 结构化切块）
HEADERS_TO_SPLIT_ON = [
    ("h1", "章"),
    ("h2", "节"),
    ("h3", "小节"),
]

# 兜底切块器：当语义切块产生过大的块时，再做一次切割
FALLBACK_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    length_function=len,
    separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
)


def _get_file_extension(filename: str) -> str:
    """获取小写文件扩展名"""
    return os.path.splitext(filename)[1].lower()


# ==========================================
# 文本提取器（按文件类型）
# ==========================================

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


def _extract_html_from_epub(file_path: str) -> List[Tuple[str, str]]:
    """
    从 EPUB 提取原始 HTML 内容（保留标题标签），返回 [(chapter_title, html_content), ...]
    与旧版不同：不做 get_text()，保留 h1/h2/h3 结构供 HTMLHeaderTextSplitter 使用
    """
    import ebooklib
    from ebooklib import epub

    book = epub.read_epub(file_path, options={"ignore_ncx": True})
    html_parts = []

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        content = item.get_content()
        html_str = content.decode("utf-8", errors="ignore")
        if html_str.strip():
            html_parts.append(html_str)

    return html_parts


def _extract_text_from_epub(file_path: str) -> str:
    """从 EPUB 提取纯文本（兼容旧接口）"""
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


# ==========================================
# 两级切块核心逻辑
# ==========================================

def _chunk_epub_two_level(file_path: str) -> List[Dict[str, Any]]:
    """
    EPUB 两级切块：
    第一级：HTMLHeaderTextSplitter 按 h1/h2/h3 章节切
    第二级：SemanticChunker 在大段落内按语义边界精切
    
    返回: [{"content": str, "metadata": {"章": ..., "节": ..., ...}}, ...]
    """
    html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)

    # 提取 EPUB 原始 HTML
    html_parts = _extract_html_from_epub(file_path)

    # 第一级：按标题结构切
    structural_chunks = []
    for html_content in html_parts:
        try:
            docs = html_splitter.split_text(html_content)
            structural_chunks.extend(docs)
        except Exception as e:
            # 如果某个 HTML 片段无法解析（比如目录页），跳过
            print(f"  ⚠️ HTMLHeaderTextSplitter 解析跳过: {e}")
            continue

    print(f"  第一级（结构化切块）: {len(structural_chunks)} 个块")

    # 第二级：对每个结构化块进行语义精切
    semantic_chunker = SemanticChunker(
        _ollama_embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=70,  # 相似度低于 70% 分位数时切割
    )

    final_chunks = []
    for doc in structural_chunks:
        text = doc.page_content.strip()
        if not text or len(text) < 50:
            continue  # 跳过过短的片段（如只有标题）

        # 提取结构化元数据（章节信息）
        chapter_meta = dict(doc.metadata) if doc.metadata else {}

        try:
            # 尝试语义切块
            if len(text) > 300:
                # 只对较长的文本做语义切割，短文本直接保留
                sub_docs = semantic_chunker.create_documents([text])
                for sub_doc in sub_docs:
                    sub_text = sub_doc.page_content.strip()
                    if not sub_text:
                        continue
                    # 如果语义切块后仍然过大，用兜底切割器再切一次
                    if len(sub_text) > 800:
                        fallback_chunks = FALLBACK_SPLITTER.split_text(sub_text)
                        for fc in fallback_chunks:
                            final_chunks.append({
                                "content": fc,
                                "metadata": chapter_meta,
                            })
                    else:
                        final_chunks.append({
                            "content": sub_text,
                            "metadata": chapter_meta,
                        })
            else:
                final_chunks.append({
                    "content": text,
                    "metadata": chapter_meta,
                })
        except Exception as e:
            # 语义切块失败，降级为兜底切割
            print(f"  ⚠️ SemanticChunker 失败，降级兜底切割: {e}")
            fallback_chunks = FALLBACK_SPLITTER.split_text(text)
            for fc in fallback_chunks:
                final_chunks.append({
                    "content": fc,
                    "metadata": chapter_meta,
                })

    print(f"  第二级（语义精切）: {len(final_chunks)} 个最终块")
    return final_chunks


def _chunk_text_semantic(full_text: str) -> List[Dict[str, Any]]:
    """
    非 EPUB 文件的语义切块（PDF/TXT/MD）：
    直接用 SemanticChunker，超大块用兜底切割器再切
    """
    if not full_text.strip():
        return []

    semantic_chunker = SemanticChunker(
        _ollama_embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=70,
    )

    final_chunks = []
    try:
        docs = semantic_chunker.create_documents([full_text])
        for doc in docs:
            text = doc.page_content.strip()
            if not text:
                continue
            if len(text) > 800:
                fallback_chunks = FALLBACK_SPLITTER.split_text(text)
                for fc in fallback_chunks:
                    final_chunks.append({"content": fc, "metadata": {}})
            else:
                final_chunks.append({"content": text, "metadata": {}})
    except Exception as e:
        print(f"  ⚠️ SemanticChunker 失败，使用纯兜底切割: {e}")
        fallback_chunks = FALLBACK_SPLITTER.split_text(full_text)
        for fc in fallback_chunks:
            final_chunks.append({"content": fc, "metadata": {}})

    return final_chunks


# ==========================================
# 主处理管线
# ==========================================

async def process_uploaded_file(
    file: UploadFile,
    db: AsyncSession,
    user_id: uuid.UUID,
) -> Dict[str, Any]:
    """
    完整的文件处理管线（v2 - 两级切块）：
    1. 保存上传文件到临时目录
    2. 根据文件类型选择切块策略
    3. 生成 Embedding
    4. 批量写入数据库（含章节元数据）

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
        print(f"📖 开始处理文件: {filename}")

        # 2. 根据文件类型选择切块策略
        if extension == ".epub":
            # EPUB: 两级切块（结构化 + 语义）
            chunk_list = _chunk_epub_two_level(tmp_path)
        else:
            # 其他格式: 先提取文本，再语义切块
            full_text = extract_text(tmp_path, extension)
            if not full_text.strip():
                return {
                    "filename": filename,
                    "chunks_created": 0,
                    "status": "empty",
                    "message": "文件内容为空，未创建任何切片",
                }
            chunk_list = _chunk_text_semantic(full_text)

        if not chunk_list:
            return {
                "filename": filename,
                "chunks_created": 0,
                "status": "empty",
                "message": "切片后无内容",
            }

        print(f"  ✅ 切块完成，共 {len(chunk_list)} 个块，开始生成 Embedding...")

        # 3. 生成 Embedding
        chunk_texts = [c["content"] for c in chunk_list]
        embeddings = embed_texts(chunk_texts)

        # 4. 批量入库（含章节元数据）
        documents = []
        for i, (chunk_info, embedding) in enumerate(zip(chunk_list, embeddings)):
            # 合并元数据：文件信息 + 章节信息
            metadata = {
                "filename": filename,
                "chunk_index": i,
                "total_chunks": len(chunk_list),
            }
            # 添加章节元数据
            if chunk_info.get("metadata"):
                metadata.update(chunk_info["metadata"])

            doc = Document(
                user_id=user_id,
                content=chunk_info["content"],
                metadata_=metadata,
                embedding=embedding,
            )
            documents.append(doc)
            db.add(doc)

        await db.commit()

        # 统计章节分布
        chapters = set()
        for c in chunk_list:
            chapter = c.get("metadata", {}).get("章", "")
            if chapter:
                chapters.add(chapter)

        chapter_info = f"，覆盖 {len(chapters)} 个章节" if chapters else ""

        return {
            "filename": filename,
            "chunks_created": len(documents),
            "status": "success",
            "message": f"成功处理文件：{len(chunk_list)} 个切片已入库{chapter_info}",
        }

    finally:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
