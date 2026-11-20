from pathlib import Path
from uuid import uuid4
from fastapi import APIRouter, UploadFile, File, HTTPException, Header
from pydantic import BaseModel, Field
from app.core.config import get_settings
from app.rag.workflow import ask
from app.rag.vectorstore import add_documents 
from app.services.ingestion import load_file, chunk_documents, SUPPORTED
from app.services.audit import (
    write_audit,
    save_chat_message,
    get_chat_history,
)

router = APIRouter(prefix="/api")
settings = get_settings()

class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=3000)
    session_id: str | None = None


@router.get("/health")
def health():
    return {"status": "ok", "service": settings.app_name}


@router.post("/chat")
def chat(payload: ChatRequest):
    try:
        session_id = payload.session_id or str(uuid4())

        # Retrieve previous conversation
        history = get_chat_history(session_id, limit=20)

        # Generate AI response using conversation history
        result = ask(
            payload.question,
            history=history,
        )

        # Store current conversation
        save_chat_message(
            session_id, "user", payload.question
        )
        save_chat_message(
            session_id, "assistant", result["answer"]
        )

        write_audit(
            payload.question,
            result["source_used"],
            result.get("trace", []),
        )

        return {
            "session_id": session_id,
            "answer": result["answer"],
            "source_used": result["source_used"],
            "trace": result.get("trace", []),
            "citations": result.get("citations", []),
            "rewritten_query": result.get(
                "current_query", payload.question
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=str(exc)
        ) from exc


@router.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    x_admin_key: str = Header(default=""),
):
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")

    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in SUPPORTED:
        raise HTTPException(
            status_code=400,
            detail=f"Supported: {', '.join(sorted(SUPPORTED))}",
        )

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    dest = upload_dir / Path(file.filename).name
    dest.write_bytes(await file.read())

    docs = load_file(dest)
    chunks = chunk_documents(docs)
    ids = add_documents(chunks)

    return {
        "message": "Document indexed",
        "file": dest.name,
        "chunks": len(chunks),
        "ids_created": len(ids),
    }