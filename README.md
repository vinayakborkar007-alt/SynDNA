# SynDNA — Local Setup (Frontend + FastAPI Backend, hosted separately)

This package contains two independently-runnable services:

```
syndna-app/
├── backend/     FastAPI service — feature extraction + XGBoost prediction
│   ├── main.py                  FastAPI app (/analyze, /generate-reports, /download/...)
│   ├── pipeline.py              Glue: sequence -> features -> model
│   ├── features.py              Builds the 377 model features from a raw sequence
│   ├── model.py                 Loads syndna_final_model.pkl, runs XGBoost prediction
│   ├── dna_pipeline.py          Bioinformatics feature/report/plot pipeline (used by /generate-reports)
│   ├── parser.py                FASTA parsing helper
│   ├── schemas.py                Pydantic response schema
│   ├── final_feature_columns.csv Exact 377-feature column order the model expects
│   ├── syndna_final_model.pkl    Trained XGBoost model
│   └── requirements.txt
└── frontend/
    ├── index.html               The dashboard UI (unchanged — already calls http://localhost:8000)
    └── server.py                Tiny static file server so the frontend runs on its own port
```

Both services run on `localhost` only. The frontend is plain static HTML/JS — it
talks to the backend purely via `fetch()` calls to `http://localhost:8000`, so
the two can be started, stopped, and hosted completely independently of each
other (e.g. on different ports, different machines on the same network, or
later deployed to two separate hosts without touching the code — you'd only
change the `http://localhost:8000` strings in `index.html`).

## 1. Run the backend (port 8000)

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Verify it's up: open http://localhost:8000/ — you should see
`{"status":"SynDNA FastAPI Backend is running successfully!"}`

Interactive API docs: http://localhost:8000/docs

## 2. Run the frontend (port 5500), in a separate terminal

```bash
cd frontend
python3 server.py          # serves on http://localhost:5500
```

Then open **http://localhost:5500** in your browser.

(Any static server works here — `npx serve .` or the VS Code "Live Server"
extension would do the same job. `server.py` is included so no extra install
is required.)

## 3. Use it

1. Paste a FASTA sequence (or raw ATGC) into the dashboard and click **Analyze**.
2. The frontend calls `POST /analyze` on the backend, which runs the full
   pipeline: `features.py` extracts the 377 model features from your
   sequence → `model.py` loads `syndna_final_model.pkl` and runs the real
   XGBoost prediction → the verdict, anomaly score, and top feature
   importances are returned and rendered.
3. It then calls `POST /generate-reports`, which runs the separate
   bioinformatics pipeline in `dna_pipeline.py` (GC content, ORFs, repeats,
   restriction-site scars, linear DNA map + GC-track plot) and returns a
   downloadable CSV and PNG, fetched via `/download/csv/{id}` and
   `/download/plot/{id}/{filename}`.

I ran all of this in a sandbox before handing it back to you — `/analyze`,
`/generate-reports`, and the CSV/plot download endpoints all returned correct
real output from the actual trained model and pipeline (not placeholders).

## Notes / things worth knowing

- **CORS** is already open (`allow_origins=["*"]`) in `main.py`, so the
  frontend can be on any port/host and still reach the backend.
- **Demo override in `main.py`**: there's a hardcoded rule in
  `process_prediction()` that forces a fixed 94% "Synthetic / High Risk"
  result whenever a sequence contains the `GAATTCGGATCCAAGCTTGCGGCCGC`
  motif or 3+ `GAATTC` (EcoRI) sites — it bypasses the real model for those
  specific inputs. I left it in place since it looked intentional (a
  guaranteed high-risk demo case), but flagging it in case you want every
  result to come purely from the trained model — just delete that `if`
  block in `main.py` if so.
- The backend loads the model once at startup (`model.py`), so predictions
  after the first request are fast.
- `dna_pipeline.py` uses `matplotlib` with the non-interactive `Agg`
  backend, so plot generation works headless (no display needed).
- I removed the stray `" - Copy.py"` / `"- Copy"` duplicate files from the
  original repo's `backend/` folder in this package — they weren't imported
  by anything and were just clutter.
