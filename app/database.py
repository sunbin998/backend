# app/database.py
import os

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

# 读取 backend/.env
load_dotenv()

# 数据库连接字符串（环境变量优先）
# 示例: postgresql+asyncpg://postgres:root@localhost:5432/RAGDB
DEFAULT_DATABASE_URL = "postgresql+asyncpg://postgres:root@localhost:5432/RAGDB"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)

# 兼容部分平台常见写法: postgres:// -> postgresql+asyncpg://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)


def _get_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


SQL_ECHO = _get_bool_env("SQL_ECHO", default=False)

# 是否在启动时清空并重建数据库（仅建议本地临时开发使用）
DB_RECREATE_ON_START = _get_bool_env("DB_RECREATE_ON_START", default=False)

# 创建异步引擎
engine = create_async_engine(
    DATABASE_URL,
    echo=SQL_ECHO,
    future=True,
    pool_pre_ping=True,
)

async_session_factory = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

async def init_db():
    """
    初始化数据库表结构。
    在应用启动时调用，如果表不存在则创建。
    注意：这需要配合 pgvector 扩展使用。
    """
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # 可选重建模式：用于快速重置开发环境
        if DB_RECREATE_ON_START:
            await conn.run_sync(SQLModel.metadata.drop_all)

        await conn.run_sync(SQLModel.metadata.create_all)

async def get_session() -> AsyncSession:
    """
    依赖注入函数：为每个请求提供一个独立的数据库会话。
    使用 yield 确保会话在使用后自动关闭。
    """
    async with async_session_factory() as session:
        yield session