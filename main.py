from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agents.search import SearchClient
from app.api.routes import router as api_router
from app.core.config import get_settings
from app.core.llm import LLMClient
from app.db.sqlite import init_db
from app.graph.workflow import create_workflow


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    llm = LLMClient(settings)
    search_client = SearchClient(settings)
    workflow = create_workflow(llm, search_client)

    await init_db(settings.database_url)

    app.state.settings = settings
    app.state.workflow = workflow
    yield


app = FastAPI(
    title="Multi Agent Research System",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> dict:
    return {
        "name": "Multi Agent Research System",
        "status": "ok",
        "ui": "Run React UI on http://127.0.0.1:5173",
        "endpoints": ["/api/health", "/api/research/stream"],
    }

app.include_router(api_router, prefix="/api")
