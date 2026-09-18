import os

from fastapi import FastAPI

from app.routers import chat, health, whatif

app = FastAPI(title="Darukaa Biodiversity AI")

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(whatif.router)


if __name__ == "__main__":
    # Container entrypoint (see Dockerfile CMD). Binds to all interfaces and
    # reads the PORT env var Render/Railway inject, defaulting to 8000 for a
    # plain `docker run` or local `python -m app.main`.
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
