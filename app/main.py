# app/main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.database import init_db
from app import models
from app.api.endpoints import sessions

# 生命周期管理：应用启动时初始化数据库
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("正在初始化数据库表结构...")
    await init_db()
    print("数据库初始化完成！")
    yield
    print("应用关闭...")

app = FastAPI(
    title="AGI Knowledge Assistant Backend",
    version="0.1.0",
    lifespan=lifespan
)

app.include_router(sessions.router, prefix="/api/sessions", tags=["Sessions"])

@app.get("/")
async def root():
    return {"message": "API 服务已启动", "status": "running"}


# 提示：后续你会在这里引入 api.endpoints 下的 router
# from app.api.endpoints import sessions, chat
# app.include_router(sessions.router, prefix="/api/sessions", tags=["Sessions"])