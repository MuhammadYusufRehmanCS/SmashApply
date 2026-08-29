import json

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import MasterCV
from app.schemas import MasterCVOut
from app.services.cv_layout import parse_master_cv

router = APIRouter(prefix="/api/cv", tags=["cv"])


def _to_out(cv: MasterCV) -> MasterCVOut:
    return MasterCVOut(
        id=cv.id,
        filename=cv.filename,
        raw_text=cv.raw_text,
        sections=json.loads(cv.sections_json),
        layout=json.loads(cv.layout_json),
        created_at=cv.created_at,
        updated_at=cv.updated_at,
    )


@router.get("", response_model=MasterCVOut | None)
def get_master_cv(db: Session = Depends(get_db)):
    cv = db.execute(select(MasterCV).order_by(MasterCV.id.desc())).scalars().first()
    return _to_out(cv) if cv else None


@router.post("/upload", response_model=MasterCVOut)
async def upload_master_cv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    filename = file.filename or "master_cv.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported for layout parsing.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail=f"{filename} is empty.")

    try:
        parsed = parse_master_cv(content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surface a clear JSON error instead of a 500
        raise HTTPException(status_code=400, detail=f"Could not parse {filename}: {exc}") from exc

    cv = db.execute(select(MasterCV).order_by(MasterCV.id.desc())).scalars().first()
    if cv is None:
        cv = MasterCV()
        db.add(cv)

    cv.filename = filename
    cv.raw_text = parsed["raw_text"]
    cv.sections_json = json.dumps(parsed["sections"])
    cv.layout_json = json.dumps(parsed["layout"])

    db.commit()
    db.refresh(cv)
    return _to_out(cv)
