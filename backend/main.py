from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schemas import PredictionRequest, PredictionResponse
from parser import parse_fasta
from pipeline import run_prediction

app = FastAPI(
    title="SynDNA Guard API",
    description="Backend API for SynDNA",
    version="1.0.0"
)

# Enable CORS for local frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

        result = run_prediction(
            sequence=sequence,
            host=request.host
        )

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

    except Exception as e:
        return {
            "success": False,
            "error": f"Prediction failed: {str(e)}"
        }