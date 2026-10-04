"""HTTP API: `uvicorn server.api:app --reload`."""

import asyncio
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from lawhack.config import Settings
from lawhack.llm import answer_client, baseline_client
from lawhack.pipeline import analyse, load
from lawhack.schema import Answer, CheckedClaim, Registry
from lawhack.solution import Solution
from lawhack.system_one import TypeSafeSystemOne
from lawhack.system_two import ask, ask_baseline
from lawhack.verify import verify_text

load_dotenv()
app = FastAPI(title="LawHack")
_store: dict[str, tuple[str, Registry, Solution]] = {}


class Loaded(BaseModel):
    document_id: str
    solution: Solution
    segments: int
    seconds: float
    input_tokens: int


class AskRequest(BaseModel):
    document_id: str
    question: str
    baseline: bool = False


class AskResponse(BaseModel):
    answer: Answer
    text: str
    baseline: str | None = None
    baseline_check: list[CheckedClaim] | None = None


class VerifyRequest(BaseModel):
    document_id: str
    text: str


def _get(document_id: str) -> tuple[str, Registry, Solution]:
    if document_id not in _store:
        raise HTTPException(404, "Arrêt inconnu : appelez d'abord /load")
    return _store[document_id]


@app.post("/load", response_model=Loaded)
def load_decision(file: UploadFile) -> Loaded:
    suffix = Path(file.filename or "arret.pdf").suffix or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(file.file.read())
        tmp.flush()
        doc = load(tmp.name)
    registry, solution = analyse(doc)
    _store[doc.id] = (doc.text, registry, solution)
    return Loaded(document_id=doc.id, solution=solution, segments=len(registry.entries),
                  seconds=registry.seconds, input_tokens=registry.input_tokens)


@app.get("/registry/{document_id}", response_model=Registry)
def get_registry(document_id: str) -> Registry:
    return _get(document_id)[1]


@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest) -> AskResponse:
    text, registry, solution = _get(req.document_id)
    settings = Settings.from_env()
    jev = TypeSafeSystemOne()
    answer = asyncio.run(ask(req.question, registry, jev, answer_client(settings), settings, solution.value))
    out = AskResponse(answer=answer, text=answer.render())
    if req.baseline:
        base = asyncio.run(ask_baseline(req.question, text, baseline_client(settings)))
        out.baseline = base.text
        out.baseline_check, _ = asyncio.run(verify_text(base.text, registry, jev, settings))
    return out


@app.post("/verify", response_model=list[CheckedClaim])
def verify(req: VerifyRequest) -> list[CheckedClaim]:
    _, registry, _ = _get(req.document_id)
    settings = Settings.from_env()
    claims, _ = asyncio.run(verify_text(req.text, registry, TypeSafeSystemOne(), settings))
    return claims
