# SynDNA — XGBoost 40-row integration

This patch is designed for the current SynDNA architecture where the frontend runs separately on `localhost:5500` and FastAPI runs on `localhost:8000`.

## Replace/add

Replace these existing backend files:

- `backend/features.py`
- `backend/model.py`
- `backend/pipeline.py`
- `backend/main.py`
- `backend/requirements.txt`

Add:

- `backend/artifacts/xgboost_model_40.pkl`
- `backend/artifacts/final_feature_columns.json`

Replace `frontend/index.html` with the included patched version. The patched JavaScript uses `flagged_region.start/end` returned by FastAPI to position the red bar instead of the fixed CSS position.

Leave `backend/dna_pipeline.py` unchanged unless your current `main.py` explicitly depends on functions from it.

## Model

The included XGBoost classifier is trained on exactly 40 rows from `combined_features.csv`: 20 natural (`0`) and 20 synthetic/lab-made (`1`). It uses 17 sequence-derived feature columns. The same extractor is used for full-sequence prediction and overlapping 500-bp windows, so the suspicious-region map is produced from XGBoost window probabilities rather than a hard-coded region.

## Run

From the repository root:

```powershell
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

Keep the frontend on:

```text
http://localhost:5500
```

## API

`POST /analyze`

Returns the model classification, anomaly score, extracted features, and `flagged_region` coordinates.

`POST /generate-reports`

Returns the feature CSV and the XGBoost window-score graph used by the UI.
