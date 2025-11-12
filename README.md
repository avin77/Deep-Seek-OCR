# Local DeepSeek Invoice OCR

This repo hosts the Streamlit reviewer UI, helper scripts, and utilities for a local document OCR workflow powered by the DeepSeek-OCR model. The backend runs inside a GPU-enabled Docker container that exposes a FastAPI `/ocr` endpoint, and the frontend uploads PDFs/images to that service and renders the extracted text.

## Components

| Component | Location | Notes |
|-----------|----------|-------|
| DeepSeek FastAPI service | external (docker) | Container launches `uvicorn app:app` inside `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`, mounting the model + frontend volumes. |
| Streamlit UI | `ui/app.py` | Uploads PDFs/images, previews them, calls `/ocr`, and shows the returned Markdown/text. |
| Supporting code | `app/`, `utils/`, `scripts/`, `tests/` | Scaffolding for future endpoints/utilities/tests (not required for the basic workflow). |

## Prerequisites

- Windows 11 / WSL / Linux with Docker, NVIDIA Container Toolkit, and a CUDA-capable GPU.
- DeepSeek-OCR weights downloaded locally (e.g. `C:\models\DeepSeek-OCR`).
- Python 3.10+ and `pip`.
- Poppler binaries on PATH for `pdf2image` previews (`choco install poppler` on Windows).

## 1. Start the DeepSeek FastAPI container

The container mounts:

- `C:\models\DeepSeek-OCR` – DeepSeek checkpoint.
- `C:\models\ocr-app` – FastAPI app (`app.py` already loads the model and exposes `/ocr`).
- `C:\models\hf-cache` – Hugging Face cache (optional but recommended).

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

Watch `docker logs -f deepseek-ocr` until you see `Application startup complete`. The API now serves `http://localhost:7861/ocr`.

Smoke test:

```powershell
curl -X POST -F "file=@C:\path\to\sample.png" http://localhost:7861/ocr
```

## 2. Install the Streamlit UI

```powershell
cd C:\Coding\OCR
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

> Restart your terminal after installing Poppler so `pdf2image` can find `pdftoppm`.

## 3. Run the frontend

```powershell
cd C:\Coding\OCR
.\.venv\Scripts\activate
streamlit run ui\app.py
```

Open `http://localhost:8501`. The sidebar contains the FastAPI base URL (defaults to `http://localhost:7861`).

### Using the UI

1. Upload a PDF or image (PNG/JPG/WebP/TIFF) of the bill.
2. PDFs are previewed, and page 1 is converted to PNG before hitting `/ocr`. Images are sent as-is.
3. Click **Extract text with DeepSeek-OCR**.
4. The extracted Markdown/text appears in a textarea, and the raw JSON response is displayed below for debugging/export.

### Troubleshooting

- **Streamlit cannot connect** – ensure the Docker container is running and the sidebar URL matches the backend (port 7861 by default).
- **PDF preview errors** – confirm Poppler is installed and restart your shell.
- **Empty/failed OCR** – check `docker logs deepseek-ocr` for stack traces. The FastAPI app surfaces exceptions in the HTTP response body.
- **Slow first request** – initial model warm-up may take ~90 seconds after container start. Wait for the curl smoke test to succeed before using Streamlit.

## 4. (Optional) Tests

The repo includes scaffolding for utilities/tests. Once additional logic is implemented:

```powershell
.\.venv\Scripts\activate
pytest
```

## File overview

- `ui/app.py` – Streamlit UI wired to `/ocr`.
- `requirements.txt` – runtime dependencies (Streamlit, pdf2image, httpx, etc.).
- `scripts/`, `app/`, `utils/`, `tests/` – placeholders for expanding beyond the current UI.

All processing remains local: documents never leave your machine.