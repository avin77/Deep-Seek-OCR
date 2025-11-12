# Local DeepSeek Invoice OCR

A local-first workflow for converting scanned bills/invoices into structured text with the DeepSeek-OCR model. The stack consists of:

1. **DeepSeek FastAPI container** - loads the 6.6 GB checkpoint and exposes `/ocr`.
2. **Streamlit reviewer UI** - uploads PDFs/images, previews them, and displays the model output.
3. (Optional) legacy **vLLM + FastAPI + Streamlit** trio for people who want to run every service directly on the host.

All processing happens on your machine; documents never leave your GPU box.

## Current Functionality

- Upload PDFs or images from the UI; PDFs are rasterized and only the first page is sent for now.
- DeepSeek FastAPI endpoint returns markdown-like text extracted from the document.
- UI shows both the cleaned text and the raw JSON payload for downstream automations.
- Helper scripts exist for spinning up independent vLLM/FastAPI/Streamlit processes (see `scripts/run_local.sh`).
- Backend source (`backend/app.py`) mirrors the container code so the API can also be run natively for debugging.

## Architecture

### Docker inference stack (recommended)
```
+----------------+    +-----------------------------+    +-------------------------+
| Streamlit UI   | -> | FastAPI in Docker (uvicorn) | -> | DeepSeek-OCR model dir  |
+----------------+    | GPU bindings + HF cache     |    | mounted from host       |
                      +-----------------------------+    +-------------------------+
                                   ^
               localhost:7861 (/ocr)+
```
- **Volumes**: `C:\models\DeepSeek-OCR` (weights), `C:\models\ocr-app` (FastAPI code), `C:\models\hf-cache` (HF cache).
- **Image**: `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` with `accelerate`, `transformers`, etc.

### Manual "three server" stack
```
Streamlit (8501) -> FastAPI (9000) -> vLLM (8000) -> local DeepSeek weights
```
Use `scripts/run_local.sh {vllm|api|ui|all}` to launch each component with overidable env vars (`MODEL_PATH`, `VLLM_PORT`, `FASTAPI_PORT`, `STREAMLIT_PORT`).

## Setup - Docker stack

1. Download DeepSeek-OCR weights to `C:\models\DeepSeek-OCR` (e.g., via `huggingface-cli download deepseek-ai/DeepSeek-OCR --local-dir C:\models\DeepSeek-OCR`).
2. Put the FastAPI app in `C:\models\ocr-app\app.py` (this repo's `backend/app.py`).
3. Start the container:
   ```powershell
   docker rm -f deepseek-ocr 2>$null
   docker run -d --name deepseek-ocr --gpus all `
     -p 7861:7860 `
     -v "C:\models\DeepSeek-OCR:/models/DeepSeek-OCR" `
     -v "C:\models\ocr-app:/opt/app" `
     -v "C:\models\hf-cache:/root/.cache/huggingface" `
     pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime `
     bash -lc "pip install -q --no-cache-dir 'accelerate>=0.26.0' 'transformers==4.47.1' 'tokenizers==0.21.0' addict matplotlib einops easydict timm fastapi uvicorn pillow python-multipart && uvicorn app:app --host 0.0.0.0 --port 7860 --app-dir /opt/app"
   ```
4. Watch `docker logs -f deepseek-ocr` for **Application startup complete**.
5. Smoke test: `curl -X POST -F "file=@C:\path\to\sample.png" http://localhost:7861/ocr`.

## Setup - three-server stack (optional)

```bash
# Terminal 1 - vLLM serving DeepSeek on port 8000
MODEL_PATH=deepseek-ai/DeepSeek-OCR scripts/run_local.sh vllm

# Terminal 2 - FastAPI on port 9000
env FASTAPI_PORT=9000 scripts/run_local.sh api

# Terminal 3 - Streamlit UI on port 8501
env STREAMLIT_PORT=8501 scripts/run_local.sh ui
```
Run `scripts/run_local.sh all` to start everything sequentially (Ctrl+C stops all child processes).

## Streamlit usage

```powershell
cd C:\Coding\OCR
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run ui\app.py
```
- Visit `http://localhost:8501`.
- Leave the sidebar base URL as `http://localhost:7861` (unless you changed ports).
- Upload a PDF/Image -> click **Extract text with DeepSeek-OCR** -> review the text + raw JSON.

## Repository layout

| Path | Purpose |
|------|---------|
| `backend/app.py` | FastAPI service used inside the Docker container (loads model, exposes `/ocr`). |
| `ui/app.py` | Streamlit reviewer UI hooked to the FastAPI endpoint. |
| `scripts/run_local.sh` | Convenience launcher for the legacy vLLM/FastAPI/Streamlit stack. |
| `app/`, `faiss_service/`, `utils/`, `tests/` | Scaffolding for future structured OCR + FAISS workflows. |
| `requirements.txt` | Streamlit + helper dependencies. |

-## Future roadmap
-
- **Multi-page PDF support** - send every page (batching) instead of just page 1.
- **Structured JSON extraction** - revive the FAISS/indexing pipeline for line items and semantic search.
- **Inline editing + export** - allow fixing OCR output in the UI and exporting to CSV/JSON.
- **Authentication & API keys** - optional auth between UI and API for multi-user setups.
- **Automated tests** - flesh out the placeholders under `tests/` once the extended pipeline ships.
- **Human-in-the-loop review** - capture reviewer edits/approvals in the UI so model outputs can be corrected, audited, and used for continual fine-tuning.
- **RAG enrichment** - attach a retrieval layer (FAISS plus invoice history) so outputs reference prior documents or vendor-specific context.
- **Observability dashboard** - add Grafana/Streamlit dashboards for throughput, latency, GPU utilization, and error rates.
- **Feedback dataset builder** - persist human edits alongside raw model output to seed future supervised fine-tunes.

## Core metrics to track

- **Latency**: end-to-end time from upload to OCR text (p50/p90) plus GPU utilization for capacity planning.
- **Accuracy**: field-level correctness (invoice numbers, totals, line items) using sampled human labels.
- **Adoption**: number of documents processed per day and the percentage handled without manual edits.
- **Cost**: GPU hours vs. throughput, amortized per invoice.
- **Quality debt**: backlog of human overrides awaiting model retraining (drives human-in-the-loop staffing).

These metrics will feed the planned dashboard to keep the solution grounded in measurable value and highlight when workflow or model updates are needed.
- **Human-in-the-loop review** - capture reviewer edits/approvals in the UI so model outputs can be corrected, audited, and used for continual fine-tuning.
- **RAG enrichment** - attach a retrieval layer (FAISS + invoice history) so summaries can reference prior documents or vendor-specific context.
- **Observability dashboard** - Grafana/Streamlit dashboard to surface throughput, latency, GPU utilization, and error rates across the stack.
- **Business metrics** - compute per-invoice accuracy (field-level DIFOT), reviewer time saved, and confidence scores to judge solution feasibility.
- **Feedback dataset builder** - persist human edits plus raw model output to a training set for future supervised fine-tuning.

-## Core metrics to track
-
- **Latency**: end-to-end time from upload to OCR text (p50/p90) plus GPU utilization for capacity planning.
- **Accuracy**: field-level correctness (invoice number, totals, line items) using sampled human labels.
- **Adoption**: number of documents processed per day, % handled without manual edits.
- **Cost**: GPU hours vs. throughput, amortized per invoice.
- **Quality debt**: backlog of human overrides awaiting model retraining (helps plan HITT efforts).

-These metrics will feed the planned dashboard to keep the solution grounded in measurable value and highlight when model updates or workflow tweaks are needed.

Contributions welcome-open an issue or PR with proposed improvements.
