import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.router import router


app = FastAPI(
    title="Face Embedding Model",
    description="Service extract face vectors"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "facenet512-face-embedding"}

app.include_router(router, prefix="/api/v1")