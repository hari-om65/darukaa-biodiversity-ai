from fastapi import FastAPI

from app.routers import chat, health, whatif

app = FastAPI(title="Darukaa Biodiversity AI")

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(whatif.router)
