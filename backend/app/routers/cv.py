import io

import docx
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pypdf import PdfReader
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import CVMaster
from app.schemas import CVIn, CVOut
from app.services.vector_matcher import embed_text, embedding_to_json

router = APIRouter(prefix="/api/cv", tags=["cv"])


def _extract_pdf_text(content: bytes) -> str:
    reader = PdfReader(io.BytesIO(content))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_docx_text(content: bytes) -> str:
    document = docx.Document(io.BytesIO(content))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


@router.get("", response_model=CVOut | None)
def get_cv(db: Session = Depends(get_db)):
    return db.execute(select(CVMaster).order_by(CVMaster.id.desc())).scalars().first()


@router.post("", response_model=CVOut)
def upsert_cv(payload: CVIn, db: Session = Depends(get_db)):
    if not payload.raw_text.strip():
        raise HTTPException(status_code=400, detail="raw_text must not be empty")

    vector = embed_text(payload.raw_text)
    cv = db.execute(select(CVMaster).order_by(CVMaster.id.desc())).scalars().first()

    if cv:
        cv.raw_text = payload.raw_text
        cv.embedding = embedding_to_json(vector)
    else:
        cv = CVMaster(raw_text=payload.raw_text, embedding=embedding_to_json(vector))
        db.add(cv)

    db.commit()
    db.refresh(cv)
    return cv


@router.post("/upload", response_model=CVOut)
async def upload_cv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    filename = file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in {"pdf", "docx"}:
        raise HTTPException(status_code=400, detail="Only .pdf and .docx files are supported.")

    content = await file.read()

    try:
        if ext == "pdf":
            raw_text = _extract_pdf_text(content)
        else:
            raw_text = _extract_docx_text(content)
    except Exception as exc:  # noqa: BLE001 - surface a clear JSON error instead of a 500
        raise HTTPException(status_code=400, detail=f"Could not parse {filename}: {exc}") from exc

    raw_text = raw_text.strip()
    if not raw_text:
        raise HTTPException(
            status_code=400, detail=f"No extractable text found in {filename}."
        )

    vector = embed_text(raw_text)
    cv = db.execute(select(CVMaster).order_by(CVMaster.id.desc())).scalars().first()

    if cv:
        cv.raw_text = raw_text
        cv.embedding = embedding_to_json(vector)
    else:
        cv = CVMaster(raw_text=raw_text, embedding=embedding_to_json(vector))
        db.add(cv)

    db.commit()
    db.refresh(cv)
    return cv
