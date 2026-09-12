from datetime import UTC, datetime, timedelta
from typing import Annotated

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from db.models import User
from db.session import get_session

settings = get_settings()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# Higher roles inherit every capability of the roles beneath them.
ROLE_HIERARCHY = {"viewer": 0, "list_editor": 1, "gate_operator": 2, "admin": 3}


def hash_password(password: str) -> str:
    # bcrypt hard-caps input at 72 bytes and errors above it rather than truncating.
    return bcrypt.hashpw(password.encode()[:72], bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode()[:72], hashed.encode())


def create_access_token(username: str, role: str) -> str:
    expire = datetime.now(UTC) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": username, "role": role, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict:
    """Validate a JWT and return its claims.

    Used by endpoints that cannot carry an Authorization header -- an <img>
    tag pointing at the MJPEG stream must pass its token in the query string.
    """
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> User:
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        username = payload.get("sub")
    except JWTError:
        raise credentials_error from None
    if username is None:
        raise credentials_error

    result = await session.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()
    if user is None or not user.is_active:
        raise credentials_error
    return user


def require_role(minimum: str):
    async def dependency(user: Annotated[User, Depends(get_current_user)]) -> User:
        if ROLE_HIERARCHY.get(user.role, -1) < ROLE_HIERARCHY[minimum]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires role '{minimum}' or higher",
            )
        return user

    return dependency


CurrentUser = Annotated[User, Depends(get_current_user)]
Viewer = Annotated[User, Depends(require_role("viewer"))]
ListEditor = Annotated[User, Depends(require_role("list_editor"))]
GateOperator = Annotated[User, Depends(require_role("gate_operator"))]
