from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from apps.api.routes import auth, cameras, live, plate_reads, vehicles

app = FastAPI(title="Cambodia ANPR API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(cameras.router)
app.include_router(plate_reads.router)
app.include_router(vehicles.router)
app.include_router(live.router)
app.mount("/metrics", make_asgi_app())


@app.get("/health", tags=["health"])
async def health() -> dict:
    return {"status": "ok"}
