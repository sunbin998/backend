# backend/app/services/rag_service.py
"""
RAG 检索 + Prompt 构建服务
1. 将用户查询生成 Embedding
2. 在 pgvector 中做余弦相似度搜索，取 top-K 相关文档片段
3. 将检索结果注入 System Prompt
"""
from typing import List, Tuple, Optional
from datetime import date
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.services.embedding_service import embed_query

# 检索配置
TOP_K = 10
SIMILARITY_THRESHOLD = 0.3

# ===================================================
# 系统提示词 = 教练角色 + 当天日记 + 参考资料
# ===================================================
SYSTEM_PROMPT_TEMPLATE = """<role>
你是「觉醒教练」——一位融合认知科学、积极心理学和行为设计的个人成长教练。
你拥有 10 年以上一对一教练经验，善于从日常细节中捕捉成长信号。
</role>

<identity>
- 姓名：觉醒教练
- 专长：认知科学 · 习惯设计 · 情绪管理 · 元认知 · 正念觉察
- 价值观：每个人都有自我成长的内在力量，教练的角色是镜子和催化剂，而非说教者
</identity>

<philosophy>
你的工作基于两个核心信念：

1. **日记是人生的实时镜像**
   日记不仅仅是记录——它是人生轨迹的自我可视化。通过持续书写和回顾，一个人可以觉察到自己的行为模式、情绪周期和成长轨迹，从而建立内在秩序。

2. **知识需要与现实碰撞才有力量**
   书籍是前人验证过的智慧结晶，但知识只有与个人的真实经历对照时才能内化。你的使命正是连接这两者——用理论照亮现实，用现实验证理论——帮助用户形成属于自己的成长体系。
</philosophy>

<date>今天是 {today}。用户提到的所有日期都是真实的，日记内容是真实发生过的事。</date>

<thinking_protocol>
在回答前，请在内心完成以下思考（不要输出思考过程）：
1. 用户真正想要的是什么？（表面问题 vs 深层需求）
2. 当天日记中有哪些值得关注的信号？（情绪转折、行为模式、未被用户意识到的亮点）
3. 参考资料中哪些理论能最好地照亮用户的处境？
4. 什么样的行动建议是此刻最有帮助的？（考虑用户的当前状态和承受能力）
</thinking_protocol>

<intent_routing>
首先判断用户的【当前情绪状态】与【发话意图】，并选择对应的回复策略（极度重要！绝对不要机械套用固定格式）：

1. **闲聊模式 (Casual Chat)**
   - 触发条件：日常打招呼、简单的分享（例如“今天天气不错”、“早安”）。
   - 回复策略：必须极度简短、自然、像真人的对话。无需抛出理论、不要总结日记、不要列行动清单，1-2句话即可。

2. **情感倾诉模式 (Emotional Catharsis)**
   - 触发条件：用户在宣泄负面情绪、表达委屈、愤怒、极度失落时。
   - 回复策略：**绝对不要给建议！绝对不要讲大道理！**
   - 行动：提供纯粹的心理支撑（Hold Space），承认并接纳ta的情绪。如果需要，仅用一句温柔的提问进行回应（如“听起来你今天真的很累，想多和我说说那时的感觉吗？”）。

3. **寻求建议模式 (Seeking Advice & Problem Solving)**
   - 触发条件：明确提出困惑、寻求方法改进（如“我该怎么戒掉刷短视频？”）。
   - 回复策略：
     - (a) **共情锚定**：简短回应ta目前的挣扎点。
     - (b) **结合日记与检索理论**：把你从参考资料找到的【知识块】与ta【日记中的症状】融合，用“朋友的口吻”向ta解释其背后的成因。
     - (c) **开出微行动处方**：提供 1-2 个立刻能试的微小行动阶梯（用“你可以试试...”而不是“你应该...”）。

4. **深度复盘模式 (Deep Reflection)**
   - 触发条件：用户自己进行了较长的反思，或希望发掘潜在的行为模式。
   - 回复策略：化身苏格拉底。指出你在ta日记和过往认知中看到的【模式闭环】，并用一个深刻的、一针见血的反思性提问（Reflective Question）结束，引导ta自己悟出答案。
</intent_routing>

<response_guidelines>
- 抛弃一切机械的“1234”编号排版，让文字如活水般自然流淌。
- 只有在【寻求建议】时，才考虑输出“小步行动”的列表，否则使用散文式的自然段落。
- 不要在每句话后面都加上大道理，让知识无痕地融化在关怀之中。
</response_guidelines>

<style>
- 语气：温暖而坦诚，像一位你信任的学长/学姐
- 适度使用 emoji（每段 0-1 个），增加亲和力但不轻浮
- 用 Markdown 格式化回复（加粗关键词、用列表呈现行动步骤）
- 段落简洁，每段不超过 3-4 句话
- 如果用户情绪低落，先给予足够的情感空间，不要急于给建议
</style>

<citation_rules>
- 引用书籍：在相关段落末尾标注 `[来源: 书名]`
- 引用日记：标注 `[来源: 日记 YYYY-MM-DD]`
- 自身专业建议：标注「这是我的教练建议」
- ⚠️ 绝对不要编造日记中没有的内容
- ⚠️ 如果参考资料不足，坦诚说明并基于专业知识回答
</citation_rules>

<guardrails>
- 如果用户表达了严重的心理困扰（自杀倾向、严重抑郁），温和但明确地建议寻求专业心理咨询师帮助，同时给予情感支持
- 不要做医学诊断或开处方
- 如果日记和书籍的建议存在冲突，以理解用户的实际情况为优先
- 如果用户只是闲聊，可以自然对话，不必强行分析
</guardrails>

{diary_section}
{context_section}
"""

# 当天日记段落
DIARY_SECTION = """<today_diary date="{date}">
以下是用户今天的完整日记。请仔细阅读，关注：情绪变化、行为选择、人际互动、自我评价、未被言说的潜在需求。

{diary_content}

{mood_line}
</today_diary>
"""

DIARY_SECTION_EMPTY = """<today_diary>
用户今天尚未写日记。
如果对话涉及今天的经历，可以温和地邀请：「如果你愿意，可以先在日记里梳理一下今天的经历，写下来本身就是一种觉察 ✍️」
</today_diary>
"""

# 参考资料段落
CONTEXT_SECTION = """<references>
以下是从知识库中检索到的与用户问题最相关的内容。请在回答中自然融入这些参考，不要逐条复述。

{context}
</references>
"""

CONTEXT_SECTION_EMPTY = """<references>
知识库中暂未找到直接相关的参考资料。请基于你的专业知识回答，并标注为教练建议。
</references>
"""


def _get_source_name(metadata: dict) -> str:
    """根据 metadata 智能判断来源名称"""
    source_type = metadata.get("source_type", "")

    if source_type == "diary":
        diary_date = metadata.get("diary_date", "未知日期")
        mood = metadata.get("mood", "")
        if mood:
            return f"📔 日记 {diary_date} ({mood})"
        return f"📔 日记 {diary_date}"

    filename = metadata.get("filename", "")
    if filename:
        return f"📖 {filename}"

    return "未知来源"


async def _vector_retrieve(
    query: str, db: AsyncSession, top_k: int = TOP_K,
    book_filter: List[str] = None,
    user_id: Optional[uuid.UUID] = None,
) -> List[Tuple[str, float, dict]]:
    """向量相似度检索（支持元数据预过滤）"""
    query_embedding = embed_query(query)
    query_vec_str = str(query_embedding)

    # 构建 SQL（根据是否有书籍过滤条件）
    if book_filter and user_id:
        sql = text(
            "SELECT content, metadata, "
            "1 - (embedding <=> CAST(:qvec AS vector)) AS similarity "
            "FROM documents "
            "WHERE embedding IS NOT NULL "
            "AND user_id = CAST(:user_id AS uuid) "
            "AND metadata->>'filename' = ANY(:filenames) "
            "ORDER BY embedding <=> CAST(:qvec AS vector) "
            "LIMIT :topk"
        )
        params = {
            "qvec": query_vec_str,
            "topk": top_k,
            "filenames": book_filter,
            "user_id": str(user_id),
        }
    elif book_filter:
        sql = text(
            "SELECT content, metadata, "
            "1 - (embedding <=> CAST(:qvec AS vector)) AS similarity "
            "FROM documents "
            "WHERE embedding IS NOT NULL "
            "AND metadata->>'filename' = ANY(:filenames) "
            "ORDER BY embedding <=> CAST(:qvec AS vector) "
            "LIMIT :topk"
        )
        params = {"qvec": query_vec_str, "topk": top_k, "filenames": book_filter}
    elif user_id:
        sql = text(
            "SELECT content, metadata, "
            "1 - (embedding <=> CAST(:qvec AS vector)) AS similarity "
            "FROM documents "
            "WHERE embedding IS NOT NULL "
            "AND user_id = CAST(:user_id AS uuid) "
            "ORDER BY embedding <=> CAST(:qvec AS vector) "
            "LIMIT :topk"
        )
        params = {
            "qvec": query_vec_str,
            "topk": top_k,
            "user_id": str(user_id),
        }
    else:
        sql = text(
            "SELECT content, metadata, "
            "1 - (embedding <=> CAST(:qvec AS vector)) AS similarity "
            "FROM documents "
            "WHERE embedding IS NOT NULL "
            "ORDER BY embedding <=> CAST(:qvec AS vector) "
            "LIMIT :topk"
        )
        params = {"qvec": query_vec_str, "topk": top_k}

    result = await db.execute(sql, params)
    rows = result.fetchall()

    relevant = []
    for row in rows:
        content, metadata, similarity = row
        if similarity >= SIMILARITY_THRESHOLD:
            relevant.append((content, similarity, metadata or {}))

    return relevant


async def _bm25_retrieve(
    query: str, db: AsyncSession, top_k: int = TOP_K,
    book_filter: List[str] = None,
    user_id: Optional[uuid.UUID] = None,
) -> List[Tuple[str, float, dict]]:
    """BM25 关键词检索（jieba 分词 + rank_bm25）"""
    import jieba
    from rank_bm25 import BM25Okapi

    # 1. 从数据库加载候选文档（支持书籍过滤）
    if book_filter and user_id:
        sql = text(
            "SELECT content, metadata FROM documents "
            "WHERE user_id = CAST(:user_id AS uuid) "
            "AND metadata->>'filename' = ANY(:filenames)"
        )
        result = await db.execute(
            sql,
            {"filenames": book_filter, "user_id": str(user_id)},
        )
    elif book_filter:
        sql = text(
            "SELECT content, metadata FROM documents "
            "WHERE metadata->>'filename' = ANY(:filenames)"
        )
        result = await db.execute(sql, {"filenames": book_filter})
    elif user_id:
        sql = text(
            "SELECT content, metadata FROM documents "
            "WHERE user_id = CAST(:user_id AS uuid)"
        )
        result = await db.execute(sql, {"user_id": str(user_id)})
    else:
        sql = text("SELECT content, metadata FROM documents")
        result = await db.execute(sql)

    rows = result.fetchall()
    if not rows:
        return []

    # 2. jieba 分词建立语料库
    corpus_texts = [row[0] for row in rows]
    corpus_meta = [row[1] or {} for row in rows]
    tokenized_corpus = [list(jieba.cut(text)) for text in corpus_texts]

    # 3. BM25 打分
    bm25 = BM25Okapi(tokenized_corpus)
    query_tokens = list(jieba.cut(query))
    scores = bm25.get_scores(query_tokens)

    # 4. 排序取 Top-K
    scored_indices = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )[:top_k]

    results = []
    for idx in scored_indices:
        if scores[idx] > 0:  # 过滤零分（完全不匹配）
            results.append((corpus_texts[idx], float(scores[idx]), corpus_meta[idx]))

    return results


def _rrf_fusion(
    *ranked_lists: List[Tuple[str, float, dict]],
    k: int = 60,
    top_k: int = TOP_K,
) -> List[Tuple[str, float, dict]]:
    """
    RRF (Reciprocal Rank Fusion) 融合多路检索结果
    
    公式: RRF_score(doc) = Σ 1/(k + rank_i)
    k=60 是业界标准值（Elasticsearch/Pinecone 默认）
    """
    rrf_scores = {}  # content -> {"score": float, "metadata": dict}

    for ranked_list in ranked_lists:
        for rank, (content, _original_score, metadata) in enumerate(ranked_list):
            if content not in rrf_scores:
                rrf_scores[content] = {"score": 0.0, "metadata": metadata}
            rrf_scores[content]["score"] += 1.0 / (k + rank + 1)

    # 按 RRF 分数降序排序
    sorted_results = sorted(
        rrf_scores.items(), key=lambda x: x[1]["score"], reverse=True
    )

    # 理论最高分：该知识块在所有的检索路径中全都排在第一名 (rank=0)
    # 这用于将 RRF 绝对数值（非常小，如 0.05）归一化为 0~1 的百分比相关度，便于前端展示
    if len(ranked_lists) > 0:
        max_possible_score = len(ranked_lists) * (1.0 / (k + 1))
    else:
        max_possible_score = 1.0

    # 转换回标准格式并归一化分数
    final = []
    for content, info in sorted_results[:top_k]:
        normalized_score = min(info["score"] / max_possible_score, 1.0)
        final.append((content, normalized_score, info["metadata"]))

    return final


async def retrieve_relevant_chunks(
    query: str,
    db: AsyncSession,
    diary_content: str = None,
    top_k: int = TOP_K,
    book_filter: List[str] = None,
    user_id: Optional[uuid.UUID] = None,
) -> List[Tuple[str, float, dict]]:
    """
    混合检索主入口：
    1. 元数据预过滤（按书籍）
    2. 向量检索（提问 + 日记双路）
    3. BM25 关键词检索（提问）
    4. RRF 融合所有结果
    
    参数:
        query: 用户的提问
        diary_content: 当天日记内容（可选，用于向量双路检索）
        book_filter: 书籍文件名列表（可选，用于元数据预过滤）
    """
    retrieval_paths = []

    # 路径 1: 向量检索 - 用户提问
    vector_question = await _vector_retrieve(
        query,
        db,
        top_k=top_k * 2,
        book_filter=book_filter,
        user_id=user_id,
    )
    retrieval_paths.append(vector_question)

    # 路径 2: 向量检索 - 日记内容（如果有）
    if diary_content and diary_content.strip():
        diary_query = diary_content.strip()[:500]
        vector_diary = await _vector_retrieve(
            diary_query,
            db,
            top_k=top_k * 2,
            book_filter=book_filter,
            user_id=user_id,
        )
        retrieval_paths.append(vector_diary)

    # 路径 3: BM25 关键词检索 - 用户提问
    bm25_results = await _bm25_retrieve(
        query,
        db,
        top_k=top_k * 2,
        book_filter=book_filter,
        user_id=user_id,
    )
    retrieval_paths.append(bm25_results)

    # RRF 融合所有路径
    final_results = _rrf_fusion(*retrieval_paths, top_k=top_k)

    # 日志
    path_names = ["向量(提问)"]
    if diary_content and diary_content.strip():
        path_names.append("向量(日记)")
    path_names.append("BM25(提问)")
    path_counts = [len(p) for p in retrieval_paths]
    filter_info = f"[书籍过滤: {', '.join(book_filter)}]" if book_filter else "[全部书籍]"
    
    print(f"  混合检索 {filter_info}: {' + '.join(f'{n}={c}' for n, c in zip(path_names, path_counts))} → RRF融合后 {len(final_results)} 个")

    return final_results


def build_rag_prompt(
    retrieved_chunks: List[Tuple[str, float, dict]],
    today_diary: Optional[str] = None,
    today_date: Optional[str] = None,
    today_mood: Optional[str] = None,
) -> str:
    """
    构建完整 System Prompt：
    教练角色 + 当天日记（直接注入） + 参考资料（RAG 检索）
    """
    today = today_date or date.today().isoformat()

    # 1. 当天日记段落
    if today_diary and today_diary.strip():
        mood_line = f"> 今日心情：{today_mood}" if today_mood else ""
        diary_section = DIARY_SECTION.format(
            date=today,
            diary_content=today_diary,
            mood_line=mood_line,
        )
    else:
        diary_section = DIARY_SECTION_EMPTY

    # 2. 参考资料段落
    if retrieved_chunks:
        context_parts = []
        for i, (content, score, metadata) in enumerate(retrieved_chunks, 1):
            source = _get_source_name(metadata)
            context_parts.append(f"[{i}] 来源: {source} (相关度: {score:.2f})\n{content}")
        context = "\n\n---\n\n".join(context_parts)
        context_section = CONTEXT_SECTION.format(context=context)
    else:
        context_section = CONTEXT_SECTION_EMPTY

    # 3. 组装完整 prompt
    return SYSTEM_PROMPT_TEMPLATE.format(
        today=today,
        diary_section=diary_section,
        context_section=context_section,
    )


def extract_sources(
    retrieved_chunks: List[Tuple[str, float, dict]],
) -> List[dict]:
    """从检索结果中提取来源信息（供前端展示）"""
    sources = []
    for content, score, metadata in retrieved_chunks:
        sources.append({
            "filename": _get_source_name(metadata),
            "score": round(score, 3),
            "preview": content[:120] + "..." if len(content) > 120 else content,
        })
    return sources
