import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from apps.api.main import app
from apps.api.security import hash_password
from db.models import Base, User
from db.session import get_session

# SQLite exercises routing, auth and RBAC without a live Postgres. Postgres-only
# features (partitioning, trigram index) are covered by the migration SQL instead.
TEST_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def client():
    engine = create_async_engine(TEST_URL)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        for table in ("cameras", "users", "vehicles", "provinces", "audit_log"):
            await conn.run_sync(Base.metadata.tables[table].create)

    async with sessionmaker() as session:
        session.add(User(username="viewer", hashed_password=hash_password("pw"), role="viewer"))
        session.add(User(username="editor", hashed_password=hash_password("pw"), role="list_editor"))
        await session.commit()

    async def override():
        async with sessionmaker() as session:
            yield session

    app.dependency_overrides[get_session] = override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()
    await engine.dispose()


async def token(client, username: str) -> str:
    resp = await client.post("/auth/token", data={"username": username, "password": "pw"})
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def auth(tok: str) -> dict:
    return {"Authorization": f"Bearer {tok}"}


@pytest.mark.asyncio
async def test_health_is_public(client):
    assert (await client.get("/health")).status_code == 200


@pytest.mark.asyncio
async def test_endpoints_require_auth(client):
    assert (await client.get("/vehicles")).status_code == 401
    assert (await client.get("/cameras")).status_code == 401


@pytest.mark.asyncio
async def test_login_rejects_bad_password(client):
    resp = await client.post("/auth/token", data={"username": "viewer", "password": "wrong"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_viewer_cannot_create_vehicle(client):
    tok = await token(client, "viewer")
    resp = await client.post("/vehicles", json={"plate_text": "2D-0888"}, headers=auth(tok))
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_editor_can_create_and_list_vehicle(client):
    tok = await token(client, "editor")
    created = await client.post(
        "/vehicles",
        json={"plate_text": "2D-0888", "owner_name": "Sok", "list_type": "whitelist"},
        headers=auth(tok),
    )
    assert created.status_code == 201, created.text
    assert created.json()["plate_text"] == "2D-0888"

    listed = await client.get("/vehicles", headers=auth(tok))
    assert listed.status_code == 200
    assert [v["plate_text"] for v in listed.json()] == ["2D-0888"]


@pytest.mark.asyncio
async def test_list_edit_is_audited(client):
    tok = await token(client, "editor")
    created = await client.post(
        "/vehicles", json={"plate_text": "4B-2222"}, headers=auth(tok)
    )
    vehicle_id = created.json()["id"]

    await client.patch(
        f"/vehicles/{vehicle_id}",
        json={"list_type": "blacklist", "reason": "reported stolen"},
        headers=auth(tok),
    )

    from sqlalchemy import select

    from db.models import AuditLog

    async for session in app.dependency_overrides[get_session]():
        rows = (await session.execute(select(AuditLog).order_by(AuditLog.id))).scalars().all()
        break

    assert [r.action for r in rows] == ["create", "update"]
    update = rows[1]
    assert update.actor == "editor"
    assert update.before["list_type"] == "neutral"
    assert update.after["list_type"] == "blacklist"
    assert update.reason == "reported stolen"


@pytest.mark.asyncio
async def test_duplicate_plate_conflicts(client):
    tok = await token(client, "editor")
    payload = {"plate_text": "3A-1111"}
    assert (await client.post("/vehicles", json=payload, headers=auth(tok))).status_code == 201
    assert (await client.post("/vehicles", json=payload, headers=auth(tok))).status_code == 409


@pytest.mark.asyncio
async def test_camera_name_cannot_be_blank(client):
    """A nameless camera renders as an unidentifiable row in the UI."""
    tok = await token(client, "editor")
    for name in ("", "   ", "\t\n"):
        resp = await client.post(
            "/cameras", json={"name": name, "rtsp_url": "rtsp://host/s"}, headers=auth(tok)
        )
        assert resp.status_code == 422, f"{name!r} was accepted"


@pytest.mark.asyncio
async def test_camera_rtsp_url_cannot_be_blank(client):
    tok = await token(client, "editor")
    resp = await client.post(
        "/cameras", json={"name": "gate-1", "rtsp_url": "  "}, headers=auth(tok)
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_camera_fields_are_trimmed(client):
    tok = await token(client, "editor")
    resp = await client.post(
        "/cameras",
        json={"name": "  gate-1  ", "rtsp_url": "  rtsp://host/s  "},
        headers=auth(tok),
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["name"] == "gate-1"
    assert body["rtsp_url"] == "rtsp://host/s"


@pytest.mark.asyncio
async def test_camera_update_rejects_blank_name(client):
    tok = await token(client, "editor")
    created = await client.post(
        "/cameras", json={"name": "gate-1", "rtsp_url": "rtsp://host/s"}, headers=auth(tok)
    )
    camera_id = created.json()["id"]
    resp = await client.patch(f"/cameras/{camera_id}", json={"name": "  "}, headers=auth(tok))
    assert resp.status_code == 422
