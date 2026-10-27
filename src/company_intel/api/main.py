"""FastAPI service. Run: uv run uvicorn company_intel.api.main:app --reload"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from company_intel import __version__

app = FastAPI(title="Company Intel RAG", version=__version__)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    citations: list[dict]
    retrievers: list[str]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    from company_intel.agent.graph import build_graph

    result = build_graph().invoke({"question": req.question})
    return AskResponse(
        answer=result.get("answer", ""),
        citations=result.get("citations", []),
        retrievers=result.get("retrievers", []),
    )
