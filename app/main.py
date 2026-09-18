from fastapi import FastAPI

from app.routers import chat, health

app = FastAPI(title="Darukaa Biodiversity AI")

app.include_router(health.router)
app.include_router(chat.router)
