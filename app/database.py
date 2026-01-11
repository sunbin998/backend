# app/database.py
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel
from sqlalchemy import text

# 数据库连接字符串
# 格式: postgresql+asyncpg://用户名:密码@地址:端口/数据库名
# 注意：在 WSL2 docker 中，如果使用了我之前的命令，默认是 postgres 用户
DATABASE_URL = "postgresql+asyncpg://postgres:root@localhost:5432/RAGDB"

# 创建异步引擎
# echo=True 会在控制台打印 SQL 语句，方便调试，生产环境请关闭
engine = create_async_engine(DATABASE_URL, echo=True, future=True)

async def init_db():
    """
    初始化数据库表结构。
    在应用启动时调用，如果表不存在则创建。
    注意：这需要配合 pgvector 扩展使用。
    """
    # 第一步：在单独的事务中创建 vector 扩展
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        
    # 关键:dispose 掉引擎的连接池,强制使用新连接
    await engine.dispose()
    
    # 第二步：在新的事务中创建所有表
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

async def get_session() -> AsyncSession:
    """
    依赖注入函数：为每个请求提供一个独立的数据库会话。
    使用 yield 确保会话在使用后自动关闭。
    """
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session() as session:
        yield session