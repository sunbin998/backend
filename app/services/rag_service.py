# backend/app/services/rag_service.py
"""
RAG 检索 + Prompt 构建服务
1. 将用户查询生成 Embedding
2. 在 pgvector 中做余弦相似度搜索，取 top-K 相关文档片段
3. 将检索结果注入 System Prompt
"""
from typing import List, Tuple, Optional
from datetime import date
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.services.embedding_service import embed_query

# 检索配置
TOP_K = 5
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

<response_framework>
请按以下结构组织回答（可根据对话自然程度适当调整，不必每次都写出标题）：

1. **共情锚定**（1-2 句）
   - 回应用户的情绪和处境，让ta感到被理解
   - 避免空洞的"我理解你的感受"，要具体指出你理解的是什么

2. **日记洞察**（如有当天日记）
   - 从日记中提取用户可能没注意到的模式、亮点或转折点
   - 用「我注意到你今天...」的方式温和指出
   - 将日记细节与用户的提问建立关联

3. **理论桥接**
   - 引用参考资料中最相关的理论或方法
   - 不要照搬原文，用教练的语言重新表达
   - 解释"为什么这个理论与你的情况相关"

4. **行动阶梯**（1-3 步，由易到难）
   - 每个步骤必须足够具体，用户今天就能开始做
   - 用「你可以试试...」而非「你应该...」
   - 考虑用户当前的能量水平和心情

5. **反思提问**（1 个问题，可选）
   - 留一个开放式问题引导用户继续思考
   - 好的问题比好的答案更有力量
</response_framework>

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


async def retrieve_relevant_chunks(
    query: str, db: AsyncSession, top_k: int = TOP_K
) -> List[Tuple[str, float, dict]]:
    """向量相似度检索"""
    query_embedding = embed_query(query)
    query_vec_str = str(query_embedding)

    sql = text(
        "SELECT content, metadata, "
        "1 - (embedding <=> CAST(:qvec AS vector)) AS similarity "
        "FROM documents "
        "WHERE embedding IS NOT NULL "
        "ORDER BY embedding <=> CAST(:qvec AS vector) "
        "LIMIT :topk"
    )

    result = await db.execute(
        sql,
        {"qvec": query_vec_str, "topk": top_k},
    )
    rows = result.fetchall()

    relevant = []
    for row in rows:
        content, metadata, similarity = row
        if similarity >= SIMILARITY_THRESHOLD:
            relevant.append((content, similarity, metadata or {}))

    return relevant


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
