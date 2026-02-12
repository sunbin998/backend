# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.database import init_db
from app import models
from app.api.endpoints import sessions, chat, categories, documents

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
    lifespan=lifespan,
)

# ==========================================
# 新增：CORS 中间件配置
# ==========================================
app.add_middleware(
    CORSMiddleware,
    # 允许的源：在生产环境要改成具体域名，开发环境用 "*" 偷懒没问题
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], # 允许 GET, POST, DELETE 等所有方法
    allow_headers=["*"],
)

app.include_router(sessions.router, prefix="/api/sessions", tags=["Sessions"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(categories.router, prefix="/api/categories", tags=["Categories"])
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])

@app.get("/")
async def root():
    return {"message": "API 服务已启动", "status": "running"}


# 提示：后续你会在这里引入 api.endpoints 下的 router
# from app.api.endpoints import sessions, chat
# app.include_router(sessions.router, prefix="/api/sessions", tags=["Sessions"])