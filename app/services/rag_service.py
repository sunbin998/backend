# backend/app/services/rag_service.py
"""
RAG 检索 + Prompt 构建服务
1. 将用户查询生成 Embedding
2. 在 pgvector 中做余弦相似度搜索，取 top-K 相关文档片段
3. 将检索结果注入 System Prompt
"""
from typing import List, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.services.embedding_service import embed_query

# 检索配置
TOP_K = 5
SIMILARITY_THRESHOLD = 0.3  # 余弦相似度阈值，低于此值的结果不使用

# 教练风格的 System Prompt 模板
COACH_SYSTEM_PROMPT = """你是一个个人成长教练。请基于以下参考资料回答用户的问题。
如果参考资料中没有相关内容，请基于你的知识回答，但要明确告知用户。
请用温暖、鼓励但诚实的语气，像一个教练那样帮助用户成长。

## 参考资料
{context}
"""

# 无检索结果时的 fallback prompt
FALLBACK_SYSTEM_PROMPT = """你是一个个人成长教练。
知识库中暂时没有找到与用户问题相关的参考资料。
请基于你的专业知识回答，用温暖、鼓励但诚实的语气帮助用户成长。"""


async def retrieve_relevant_chunks(
    query: str, db: AsyncSession, top_k: int = TOP_K
) -> List[Tuple[str, float, dict]]:
    """
    向量相似度检索

    Args:
        query: 用户查询文本
        db: 数据库会话
        top_k: 返回的最大结果数

    Returns:
        [(chunk_content, similarity_score, metadata), ...]
    """
    # 1. 生成查询向量
    query_embedding = embed_query(query)

    # 2. pgvector 余弦相似度搜索
    # 使用 <=> 操作符（余弦距离），值越小越相似
    # 1 - 余弦距离 = 余弦相似度
    sql = text("""
        SELECT content, metadata, 1 - (embedding <=> :query_vec::vector) AS similarity
        FROM documents
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> :query_vec::vector
        LIMIT :top_k
    """)

    result = await db.execute(
        sql,
        {
            "query_vec": str(query_embedding),
            "top_k": top_k,
        },
    )
    rows = result.fetchall()

    # 3. 过滤低相似度结果
    relevant = []
    for row in rows:
        content, metadata, similarity = row
        if similarity >= SIMILARITY_THRESHOLD:
            relevant.append((content, similarity, metadata or {}))

    return relevant


def build_rag_prompt(
    retrieved_chunks: List[Tuple[str, float, dict]],
) -> str:
    """
    根据检索结果构建 System Prompt

    Args:
        retrieved_chunks: retrieve_relevant_chunks 的返回结果

    Returns:
        完整的 system prompt 字符串
    """
    if not retrieved_chunks:
        return FALLBACK_SYSTEM_PROMPT

    # 拼接检索到的文档片段
    context_parts = []
    for i, (content, score, metadata) in enumerate(retrieved_chunks, 1):
        source = metadata.get("filename", "未知来源")
        context_parts.append(f"[{i}] 来源: {source} (相关度: {score:.2f})\n{content}")

    context = "\n\n---\n\n".join(context_parts)
    return COACH_SYSTEM_PROMPT.format(context=context)
