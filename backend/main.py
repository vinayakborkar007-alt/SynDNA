from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from .pipeline import analyze, predict_sequence, map_suspicious_regions
except ImportError:
    from pipeline import analyze, predict_sequence, map_suspicious_regions

BASE = Path(__file__).resolve().parent
REPORT_DIR = BASE / "reports"
REPORT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="SynDNA FastAPI", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500", "http://localhost:5501", "http://127.0.0.1:5501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/reports", StaticFiles(directory=REPORT_DIR), name="reports")

class AnalyzeRequest(BaseModel):
    sequence: str = Field(..., min_length=1)
    host: str = "ecoli"

@app.get("/")
def root():
    return {"status": "online", "model": "XGBoost", "training_rows": 40, "natural_rows": 20, "synthetic_rows": 20}

@app.get("/health")
def health():
    return {"status": "ok", "model": "XGBoost", "training_rows": 40}

@app.post("/analyze")
def analyze_endpoint(req: AnalyzeRequest):
    try:
        return analyze(req.sequence, req.host)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/generate-reports")
def generate_reports(req: AnalyzeRequest):
    try:
        prediction = predict_sequence(req.sequence)
        region = map_suspicious_regions(req.sequence)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]

        row = {
            "sequence_length": prediction["sequence_length"],
            "host": req.host,
            **prediction["features"],
            "classification": prediction["classification"],
            "anomaly_score": prediction["anomaly_score"],
            "anomaly_score_pct": prediction["anomaly_score_pct"],
            "flagged_start": region.get("start"),
            "flagged_end": region.get("end"),
            "flagged_region_score": region.get("score"),
        }
        csv_name = f"syndna_features_{run_id}.csv"
        pd.DataFrame([row]).to_csv(REPORT_DIR / csv_name, index=False)

        graph_name = f"syndna_region_map_{run_id}.png"
        windows = region.get("windows", [])
        plt.figure(figsize=(12, 4.5))
        if windows:
            plt.plot([w["start"] for w in windows], [w["score"] * 100 for w in windows], marker="o", markersize=3)
        plt.axhline(region.get("threshold", 0.65) * 100, linestyle="--", linewidth=1)
        plt.ylim(0, 100)
        plt.xlim(1, prediction["sequence_length"])
        plt.xlabel("Sequence position (bp)")
        plt.ylabel("XGBoost synthetic probability (%)")
        plt.title("SynDNA XGBoost Suspicious-Region Mapping")
        plt.grid(alpha=0.25)
        if region.get("start") is not None:
            plt.axvspan(region["start"], region["end"], alpha=0.18)
        plt.tight_layout()
        plt.savefig(REPORT_DIR / graph_name, dpi=160)
        plt.close()

        return {
            "success": True,
            "csv_download_url": f"/reports/{csv_name}",
            "plot_download_url": f"/reports/{graph_name}",
            "classification": prediction["classification"],
            "anomaly_score": prediction["anomaly_score"],
            "flagged_region": region,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
