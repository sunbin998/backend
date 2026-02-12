# backend/app/services/embedding_service.py
"""
Embedding 生成服务
使用本地 Ollama bge-m3 模型，1024 维
通过 Ollama REST API (http://localhost:11434/api/embed) 调用
"""
import requests
from typing import List

OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "bge-m3"
EMBEDDING_DIM = 1024
BATCH_SIZE = 32  # 每批处理的文本数量，避免 Ollama 内存溢出


def _call_ollama_embed(texts: List[str]) -> List[List[float]]:
    """调用 Ollama Embedding API"""
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={
            "model": EMBEDDING_MODEL,
            "input": texts,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["embeddings"]


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    批量生成 Embedding（用于文档入库）
    自动分批处理，每批最多 BATCH_SIZE 条
    """
    if not texts:
        return []

    all_embeddings: List[List[float]] = []
    total = len(texts)

    for i in range(0, total, BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        print(f"  Embedding 批次 {i // BATCH_SIZE + 1}/{(total + BATCH_SIZE - 1) // BATCH_SIZE}，"
              f"处理 {len(batch)} 条文本...")
        batch_embeddings = _call_ollama_embed(batch)
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def embed_query(query: str) -> List[float]:
    """
    单条查询文本生成 Embedding（用于检索）
    """
    result = _call_ollama_embed([query])
    return result[0]
