"""Create an API user. Usage: python -m scripts.create_user <username> <password> [role]"""
import asyncio
import sys

from sqlalchemy import select

from apps.api.security import ROLE_HIERARCHY, hash_password
from db.models import User
from db.session import SessionLocal


async def create(username: str, password: str, role: str) -> None:
    if role not in ROLE_HIERARCHY:
        raise SystemExit(f"role must be one of: {', '.join(ROLE_HIERARCHY)}")

    async with SessionLocal() as session:
        existing = await session.execute(select(User).where(User.username == username))
        if existing.scalar_one_or_none() is not None:
            raise SystemExit(f"user {username!r} already exists")

        session.add(
            User(username=username, hashed_password=hash_password(password), role=role)
        )
        await session.commit()
    print(f"created user {username!r} with role {role!r}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    asyncio.run(create(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "viewer"))
