from fastapi import FastAPI

from app.routers import health

app = FastAPI(title="Darukaa Biodiversity AI")

app.include_router(health.router)
