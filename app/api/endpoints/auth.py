import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.api.deps import get_current_user
from app.database import get_session
from app.models import User
from app.schemas import (
    AuthResponse,
    CommonMessage,
    RefreshTokenRequest,
    TokenPair,
    UserCreate,
    UserLogin,
    UserRead,
    UserUpdate,
)
from app.services.auth_service import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)

router = APIRouter()

MAX_AVATAR_SIZE = 2 * 1024 * 1024
ALLOWED_AVATAR_CONTENT_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
BACKEND_ROOT = Path(__file__).resolve().parents[3]
AVATAR_ROOT = BACKEND_ROOT / "uploads" / "avatars"


@router.post("/register/", response_model=AuthResponse, include_in_schema=False)
@router.post("/register", response_model=AuthResponse)
async def register(user_in: UserCreate, db: AsyncSession = Depends(get_session)):
    exists_stmt = select(User).where(User.username == user_in.username)
    exists = (await db.exec(exists_stmt)).first()
    if exists:
        raise HTTPException(status_code=400, detail="用户名已存在")

    if user_in.email:
        email_exists_stmt = select(User).where(User.email == user_in.email)
        email_exists = (await db.exec(email_exists_stmt)).first()
        if email_exists:
            raise HTTPException(status_code=400, detail="邮箱已被使用")

    user = User(
        username=user_in.username,
        email=user_in.email,
        hashed_password=hash_password(user_in.password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    access_token = create_access_token(str(user.id))
    refresh_token = create_refresh_token(str(user.id))

    return AuthResponse(
        user=user,
        access_token=access_token,
        refresh_token=refresh_token,
    )


@router.post("/login/", response_model=AuthResponse, include_in_schema=False)
@router.post("/login", response_model=AuthResponse)
async def login(login_in: UserLogin, db: AsyncSession = Depends(get_session)):
    stmt = select(User).where(User.username == login_in.username)
    user = (await db.exec(stmt)).first()

    if not user or not verify_password(login_in.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
        )

    access_token = create_access_token(str(user.id))
    refresh_token = create_refresh_token(str(user.id))

    return AuthResponse(
        user=user,
        access_token=access_token,
        refresh_token=refresh_token,
    )


@router.post("/refresh/", response_model=TokenPair, include_in_schema=False)
@router.post("/refresh", response_model=TokenPair)
async def refresh_token(req: RefreshTokenRequest):
    try:
        payload = decode_token(req.refresh_token, expected_type="refresh")
        user_id = payload["sub"]
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="refresh token 无效",
        )

    return TokenPair(
        access_token=create_access_token(user_id),
        refresh_token=create_refresh_token(user_id),
    )


@router.get("/me/", response_model=UserRead, include_in_schema=False)
@router.get("/me", response_model=UserRead)
async def me(current_user: User = Depends(get_current_user)):
    return current_user


@router.put("/me/", response_model=UserRead, include_in_schema=False)
@router.put("/me", response_model=UserRead)
async def update_me(
    update_in: UserUpdate,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    update_data = update_in.model_dump(exclude_unset=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="未提供需要更新的字段")

    if "username" in update_data and update_data["username"] is not None:
        username = update_data["username"].strip()
        if not username:
            raise HTTPException(status_code=400, detail="用户名不能为空")
        exists_stmt = select(User).where(
            User.username == username,
            User.id != current_user.id,
        )
        exists = (await db.exec(exists_stmt)).first()
        if exists:
            raise HTTPException(status_code=400, detail="用户名已存在")
        current_user.username = username

    if "email" in update_data:
        email = update_data["email"]
        email = email.strip() if email else None
        if email:
            email_exists_stmt = select(User).where(
                User.email == email,
                User.id != current_user.id,
            )
            email_exists = (await db.exec(email_exists_stmt)).first()
            if email_exists:
                raise HTTPException(status_code=400, detail="邮箱已被使用")
        current_user.email = email

    if "password" in update_data and update_data["password"] is not None:
        password = update_data["password"]
        if len(password) < 6:
            raise HTTPException(status_code=400, detail="密码长度至少 6 位")
        current_user.hashed_password = hash_password(password)

    db.add(current_user)
    await db.commit()
    await db.refresh(current_user)
    return current_user


@router.post("/me/avatar/", response_model=UserRead, include_in_schema=False)
@router.post("/me/avatar", response_model=UserRead)
async def upload_avatar(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    if not file.content_type or file.content_type not in ALLOWED_AVATAR_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="仅支持 JPG/PNG/WEBP/GIF 图片")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空")
    if len(content) > MAX_AVATAR_SIZE:
        raise HTTPException(status_code=413, detail="头像文件不能超过 2MB")

    suffix = ALLOWED_AVATAR_CONTENT_TYPES[file.content_type]
    user_dir = AVATAR_ROOT / str(current_user.id)
    user_dir.mkdir(parents=True, exist_ok=True)

    if current_user.avatar and current_user.avatar.startswith("/api/uploads/avatars/"):
        old_rel_path = current_user.avatar.replace("/api/uploads/", "", 1)
        old_path = BACKEND_ROOT / "uploads" / old_rel_path
        if old_path.exists() and old_path.is_file():
            old_path.unlink(missing_ok=True)

    filename = f"{uuid.uuid4().hex}{suffix}"
    target_path = user_dir / filename
    target_path.write_bytes(content)

    current_user.avatar = f"/api/uploads/avatars/{current_user.id}/{filename}"
    db.add(current_user)
    await db.commit()
    await db.refresh(current_user)
    return current_user


@router.delete("/me/avatar/", response_model=UserRead, include_in_schema=False)
@router.delete("/me/avatar", response_model=UserRead)
async def delete_avatar(
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    if current_user.avatar and current_user.avatar.startswith("/api/uploads/avatars/"):
        old_rel_path = current_user.avatar.replace("/api/uploads/", "", 1)
        old_path = BACKEND_ROOT / "uploads" / old_rel_path
        if old_path.exists() and old_path.is_file():
            old_path.unlink(missing_ok=True)

    current_user.avatar = None
    db.add(current_user)
    await db.commit()
    await db.refresh(current_user)
    return current_user


@router.delete("/me/", response_model=CommonMessage, include_in_schema=False)
@router.delete("/me", response_model=CommonMessage)
async def delete_me(
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    uid = str(current_user.id)

    await db.execute(
        text(
            "DELETE FROM messages "
            "WHERE session_id IN ("
            "SELECT id FROM sessions WHERE user_id = CAST(:uid AS uuid)"
            ")"
        ),
        {"uid": uid},
    )
    await db.execute(
        text("DELETE FROM sessions WHERE user_id = CAST(:uid AS uuid)"),
        {"uid": uid},
    )
    await db.execute(
        text("DELETE FROM categories WHERE user_id = CAST(:uid AS uuid)"),
        {"uid": uid},
    )
    await db.execute(
        text("DELETE FROM documents WHERE user_id = CAST(:uid AS uuid)"),
        {"uid": uid},
    )
    await db.execute(
        text("DELETE FROM diary_entries WHERE user_id = CAST(:uid AS uuid)"),
        {"uid": uid},
    )
    await db.execute(
        text("DELETE FROM users WHERE id = CAST(:uid AS uuid)"),
        {"uid": uid},
    )
    await db.commit()

    avatar_dir = AVATAR_ROOT / uid
    if avatar_dir.exists() and avatar_dir.is_dir():
        shutil.rmtree(avatar_dir, ignore_errors=True)

    return CommonMessage(message="账号已删除")
