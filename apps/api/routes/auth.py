from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.schemas import Token
from apps.api.security import CurrentUser, create_access_token, verify_password
from db.models import User
from db.session import get_session

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/token", response_model=Token)
async def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> Token:
    result = await session.execute(select(User).where(User.username == form.username))
    user = result.scalar_one_or_none()
    if user is None or not user.is_active or not verify_password(form.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return Token(access_token=create_access_token(user.username, user.role))


@router.get("/me")
async def me(user: CurrentUser) -> dict:
    return {"username": user.username, "role": user.role}
