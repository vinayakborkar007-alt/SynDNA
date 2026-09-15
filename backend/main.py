from fastapi import FastAPI
from pydantic import BaseModel
from parser import parse_fasta

app = FastAPI(
    title="SynDNA Guard API",
    description="Backend API for SynDNA",
    version="1.0.0"
)


class PredictionRequest(BaseModel):
    sequence: str
    host: str = "unknown"


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "SynDNA Guard backend is running"
    }


@app.post("/predict")
def predict(request: PredictionRequest):

    try:
        sequence = parse_fasta(request.sequence)

        return {
            "success": True,
            "sequence_length": len(sequence),
            "anomaly_score": 0.78,
            "classification": "High",
            "suspicious_regions": [],
            "feature_importance": {
                "kmer": 0.42,
                "cai": 0.31,
                "gc_skew": 0.18
            }
        }

    except ValueError as e:
        return {
            "success": False,
            "error": str(e)
        }