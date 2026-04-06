from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.api.deps import get_current_user
from app.database import get_session
from app.models import User
from app.schemas import (
    AuthResponse,
    RefreshTokenRequest,
    TokenPair,
    UserCreate,
    UserLogin,
    UserRead,
)
from app.services.auth_service import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)

router = APIRouter()


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
