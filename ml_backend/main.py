"""
PeerGrid ML API
---------------
FastAPI endpoint for the Flutter Predict College page.

Start:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

API:
    POST /predict
    GET  /health
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from predict import recommend


app = FastAPI(
    title="PeerGrid KEAM Recommendation API",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    rank: int = Field(gt=0, description="Student KEAM rank")
    category: str = Field(min_length=1)
    course: str = Field(min_length=1)
    year: int | None = Field(default=None, ge=2022)
    phase: str | None = None
    top_n: int = Field(default=10, ge=1, le=50)


@app.get("/")
def root():
    return {
        "message": "PeerGrid KEAM Recommendation API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        result = recommend(
            rank=request.rank,
            category=request.category,
            course=request.course,
            year=request.year,
            phase=request.phase,
            top_n=request.top_n,
        )
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        )
