"""Container-read API: auth, server-side validation, listing, and lookup.

The checksum and known flags are computed by the server on insert, never taken
from the client -- a test posts a bad check digit and a made-up number and
checks the server disagrees with it.
"""

import uuid
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from apps.api.main import app
from apps.api.routes import containers as containers_routes
from apps.api.security import hash_password
from apps.inference_worker.container import KnownContainers
from db.models import Base, Camera, User
from db.session import get_session

TEST_URL = "sqlite+aiosqlite:///:memory:"
KNOWN = KnownContainers(["MRSU2818883", "CMAU7224270", "TCLU5437389"])


@pytest_asyncio.fixture
async def client(monkeypatch):
    engine = create_async_engine(TEST_URL)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        for table in ("cameras", "users", "provinces", "plate_reads", "container_reads"):
            await conn.run_sync(Base.metadata.tables[table].create)

    camera_id = uuid.uuid4()
    async with sessionmaker() as session:
        session.add(User(username="viewer", hashed_password=hash_password("pw"), role="viewer"))
        session.add(User(username="editor", hashed_password=hash_password("pw"), role="list_editor"))
        session.add(Camera(id=camera_id, name="gate", rtsp_url="rtsp://x"))
        await session.commit()

    async def override():
        async with sessionmaker() as session:
            yield session

    # A small known list instead of the 171k PAS file, and no disk dependency.
    monkeypatch.setattr(containers_routes, "known_containers", lambda: KNOWN)
    app.dependency_overrides[get_session] = override
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        c.camera_id = str(camera_id)
        yield c
    app.dependency_overrides.clear()
    await engine.dispose()


async def token(client, username):
    resp = await client.post("/auth/token", data={"username": username, "password": "pw"})
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def body(client, number, **extra):
    return {"camera_id": client.camera_id, "container_number": number, "confidence": 0.9,
            "frame_ts": datetime.now(UTC).isoformat(), "model_version": "test", **extra}


@pytest.mark.asyncio
async def test_container_endpoints_require_auth(client):
    assert (await client.get("/container-reads")).status_code == 401
    assert (await client.get("/containers/MRSU2818883")).status_code == 401


@pytest.mark.asyncio
async def test_viewer_cannot_insert(client):
    h = await token(client, "viewer")
    assert (await client.post("/container-reads", json=body(client, "MRSU2818883"), headers=h)).status_code == 403


@pytest.mark.asyncio
async def test_insert_computes_checksum_and_known_server_side(client):
    h = await token(client, "editor")
    good = (await client.post("/container-reads", json=body(client, "mrsu2818883"), headers=h)).json()
    assert good["container_number"] == "MRSU2818883"        # normalised
    assert good["owner_code"] == "MRSU"
    assert good["checksum_ok"] is True
    assert good["is_known"] is True

    bad = (await client.post("/container-reads", json=body(client, "MRSU2818880"), headers=h)).json()
    assert bad["checksum_ok"] is False
    assert bad["is_known"] is False

    unknown = (await client.post("/container-reads", json=body(client, "ZZZU1234565"), headers=h)).json()
    assert unknown["is_known"] is False


@pytest.mark.asyncio
async def test_list_filters_by_number(client):
    h = await token(client, "editor")
    await client.post("/container-reads", json=body(client, "MRSU2818883"), headers=h)
    await client.post("/container-reads", json=body(client, "CMAU7224270"), headers=h)

    rows = (await client.get("/container-reads", params={"container_number": "cmau"}, headers=h)).json()
    assert [r["container_number"] for r in rows] == ["CMAU7224270"]
    assert len((await client.get("/container-reads", headers=h)).json()) == 2


@pytest.mark.asyncio
async def test_lookup_reports_format_checksum_known_and_history(client):
    h = await token(client, "editor")
    await client.post("/container-reads", json=body(client, "TCLU5437389"), headers=h)
    await client.post("/container-reads", json=body(client, "TCLU5437389"), headers=h)

    seen = (await client.get("/containers/tclu 543738 9", headers=h)).json()
    assert seen["number"] == "TCLU5437389"
    assert seen["well_formed"] and seen["checksum_ok"] and seen["is_known"]
    assert seen["read_count"] == 2 and len(seen["recent_reads"]) == 2

    never = (await client.get("/containers/CMAU7224270", headers=h)).json()
    assert never["is_known"] is True and never["read_count"] == 0 and never["last_seen"] is None

    junk = (await client.get("/containers/HELLO", headers=h)).json()
    assert junk["well_formed"] is False and junk["checksum_ok"] is False and junk["owner_code"] is None
