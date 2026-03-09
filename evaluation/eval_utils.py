"""
eval_utils.py
RAG 评测系统公共工具模块

提供：
- load_test_cases()      加载测试数据集
- get_db_session()       创建独立异步 DB session
- call_llm_judge()       调用 DeepSeek 进行 LLM-as-Judge 评分
- save_results()         保存评测结果到 JSON
- setup_sys_path()       将 backend 加入 Python 路径
"""

import os
import sys
import json
import time
import asyncio
from pathlib import Path
from typing import Any
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from openai import OpenAI

# ─── 路径配置 ────────────────────────────────────────────────────────────────
EVAL_DIR    = Path(__file__).parent
DATASET_DIR = EVAL_DIR / "dataset"
RESULTS_DIR = EVAL_DIR / "results"
REPORTS_DIR = EVAL_DIR / "reports"
BACKEND_DIR = EVAL_DIR.parent

# 加载 .env
load_dotenv(dotenv_path=BACKEND_DIR / ".env")


def setup_sys_path():
    """将 backend 目录加入 sys.path，使 app.* 可以被导入"""
    backend_str = str(BACKEND_DIR)
    if backend_str not in sys.path:
        sys.path.insert(0, backend_str)


# ─── 测试数据集 ────────────────────────────────────────────────────────────────

def load_test_cases(category_filter: str | None = None) -> list[dict]:
    """
    加载测试数据集

    Args:
        category_filter: 可选，按 category 过滤
            ('book_knowledge'|'diary_analysis'|'mixed'|'chitchat'|'boundary'|'multi_turn')

    Returns:
        测试用例列表
    """
    path = DATASET_DIR / "test_cases.json"
    if not path.exists():
        raise FileNotFoundError(f"测试数据集未找到: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if category_filter:
        data = [tc for tc in data if tc["category"] == category_filter]

    return data


# ─── 数据库 Session ───────────────────────────────────────────────────────────

@asynccontextmanager
async def get_db_session():
    """
    创建独立的异步数据库 Session（评测专用，不依赖 FastAPI 依赖注入）

    用法:
        async with get_db_session() as db:
            result = await retrieve_relevant_chunks(query, db)
    """
    setup_sys_path()
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlmodel.ext.asyncio.session import AsyncSession
    from sqlalchemy.orm import sessionmaker

    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:root@localhost:5432/RAGDB",
    )

    engine = create_async_engine(DATABASE_URL, echo=False, future=True)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        try:
            yield session
        finally:
            await engine.dispose()


# ─── DeepSeek LLM Judge ──────────────────────────────────────────────────────

def _get_deepseek_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    if not api_key:
        raise EnvironmentError("未找到 DEEPSEEK_API_KEY，请检查 .env 文件")
    return OpenAI(api_key=api_key, base_url=base_url)


def call_llm_judge(prompt: str, max_retries: int = 3, delay: float = 1.0) -> str:
    """
    调用 DeepSeek Chat 进行 LLM-as-Judge 评分

    Args:
        prompt:      评分提示词（包含评分标准和输入内容）
        max_retries: 最大重试次数
        delay:       重试间隔（秒，指数退避）

    Returns:
        LLM 返回的原始文本（通常是 1-5 的数字评分）
    """
    client = _get_deepseek_client()

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,   # 评分任务确定性要高
                max_tokens=16,     # 只需要输出一个数字
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [LLM Judge 尝试 {attempt+1}/{max_retries}] 失败: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay * (2 ** attempt))

    return ""


def parse_score(raw: str) -> float | None:
    """
    从 LLM 输出中解析数值评分（1-5）

    Args:
        raw: LLM 返回的字符串

    Returns:
        float 评分，或 None（解析失败时）
    """
    import re
    match = re.search(r"[1-5](?:\.\d+)?", raw)
    if match:
        score = float(match.group())
        return max(1.0, min(5.0, score))
    return None


# ─── 结果保存 ─────────────────────────────────────────────────────────────────

def save_results(results: list[dict], filename: str) -> Path:
    """
    保存评测结果到 JSON 文件

    Args:
        results:  评测结果列表
        filename: 输出文件名（如 'retrieval_results.json'）

    Returns:
        保存路径
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ 结果已保存至: {path}")
    return path


def load_results(filename: str) -> list[dict]:
    """加载已有评测结果（用于报告生成阶段）"""
    path = RESULTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"结果文件未找到: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── 进度辅助 ─────────────────────────────────────────────────────────────────

class Progress:
    """轻量级进度追踪器"""

    def __init__(self, total: int, desc: str = ""):
        self.total = total
        self.current = 0
        self.desc = desc
        self.start_time = time.time()

    def update(self, n: int = 1):
        self.current += n
        elapsed = time.time() - self.start_time
        avg = elapsed / self.current if self.current else 0
        eta = avg * (self.total - self.current)
        print(
            f"\r  {self.desc} {self.current}/{self.total} "
            f"[已用 {elapsed:.0f}s, 剩余 ~{eta:.0f}s]",
            end="",
            flush=True,
        )
        if self.current >= self.total:
            print()  # 换行

    def done(self) -> float:
        """返回总耗时"""
        return time.time() - self.start_time


# ─── 语义相似度（用于检索评测） ───────────────────────────────────────────────

def semantic_match(text_a: str, text_b: str, threshold: float = 0.6) -> bool:
    """
    简单的关键词重叠判断（不依赖额外模型）
    若需要更精确可替换为 embedding 余弦相似度

    Args:
        text_a, text_b: 待比较文本
        threshold:      判断为相关的最低重叠率

    Returns:
        bool: 是否语义相关
    """
    def tokenize(text: str) -> set[str]:
        # 按字或词拆分（中文逐字，英文按空格）
        import re
        chars = set(re.findall(r'[\u4e00-\u9fff]', text))
        words = set(w.lower() for w in re.findall(r'[a-zA-Z]+', text) if len(w) > 2)
        return chars | words

    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)

    if not tokens_a or not tokens_b:
        return False

    overlap = len(tokens_a & tokens_b)
    min_len = min(len(tokens_a), len(tokens_b))
    return (overlap / min_len) >= threshold


if __name__ == "__main__":
    # 自检
    print("=== eval_utils 自检 ===")

    # 1. 加载数据集
    cases = load_test_cases()
    print(f"✅ 数据集加载：{len(cases)} 条")

    from collections import Counter
    cats = Counter(tc["category"] for tc in cases)
    for cat, cnt in cats.items():
        print(f"   {cat}: {cnt} 条")

    # 2. 检查 DeepSeek 密钥
    try:
        _get_deepseek_client()
        print("✅ DeepSeek API 密钥已配置")
    except EnvironmentError as e:
        print(f"❌ {e}")

    # 3. 测试 save_results
    test_data = [{"test": True}]
    save_results(test_data, "_selfcheck.json")
    loaded = load_results("_selfcheck.json")
    assert loaded == test_data
    (RESULTS_DIR / "_selfcheck.json").unlink()
    print("✅ 结果存储读写正常")

    print("\n✅ eval_utils 自检通过")
