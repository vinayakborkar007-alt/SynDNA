from fastapi import FastAPI
from schemas import PredictionRequest, PredictionResponse
from parser import parse_fasta
from pipeline import run_prediction


app = FastAPI(
    title="SynDNA Guard API",
    description="Backend API for SynDNA",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "SynDNA Guard backend is running"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    try:
        # Parse and validate the FASTA/DNA input
        sequence = parse_fasta(request.sequence)

        # Run prediction pipeline
        result = run_prediction(sequence, request.host)

        # Return prediction response
        return {
            "success": True,
            "sequence_length": len(sequence),
            **result
        }

    except ValueError as e:
        return {
            "success": False,
            "error": str(e)
        }