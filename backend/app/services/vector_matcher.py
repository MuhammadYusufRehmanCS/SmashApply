"""Local embedding + cosine similarity matching via sentence-transformers.

Model is loaded lazily and cached as a module-level singleton so the (relatively
slow) first load only happens once per process, not once per request.
"""
import json
from functools import lru_cache

import numpy as np

from app.config import get_settings


@lru_cache
def _get_model():
    # Imported lazily so the app can start even before the model dependency
    # has downloaded its weights on first run.
    from sentence_transformers import SentenceTransformer

    settings = get_settings()
    return SentenceTransformer(settings.embedding_model)


def embed_text(text: str) -> list[float]:
    model = _get_model()
    vector = model.encode(text or "", normalize_embeddings=True)
    return vector.tolist()


def embedding_to_json(vector: list[float]) -> str:
    return json.dumps(vector)


def embedding_from_json(raw: str) -> np.ndarray:
    return np.array(json.loads(raw), dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
    return float(np.dot(a, b) / denom)


def match_score(job_description: str, cv_embedding_json: str) -> float:
    """Returns a 0-100 match score between a job description and the master CV."""
    job_vector = np.array(embed_text(job_description), dtype=np.float32)
    cv_vector = embedding_from_json(cv_embedding_json)
    similarity = cosine_similarity(job_vector, cv_vector)
    # Cosine similarity on normalized embeddings is in [-1, 1]; clamp to [0, 1] then scale.
    score = max(0.0, min(1.0, (similarity + 1) / 2))
    return round(score * 100, 1)
