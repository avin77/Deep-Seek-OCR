from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io, os, torch, tempfile, logging
from transformers import AutoModel, AutoTokenizer

MODEL_DIR = "/models/DeepSeek-OCR"  # you already bind this into the container
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda" and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
    DTYPE = torch.bfloat16
elif DEVICE == "cuda":
    DTYPE = torch.float16
else:
    DTYPE = torch.float32

# Load per the model card while keeping peak memory down
tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    use_safetensors=True,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True
).eval()

if DEVICE == "cuda":
    model = model.to("cuda")
app = FastAPI()
logger = logging.getLogger("uvicorn.error")

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    # Save upload to a temp file path because model.infer expects a file
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.save(tmp.name)
        image_path = tmp.name
    img.close()

    # Prompt style per model card
    prompt = "<image>\n<|grounding|>Convert the document to markdown."
    # Model requires an output directory even if we do not persist files
    try:
        with tempfile.TemporaryDirectory() as output_dir:
            res = model.infer(
                tok,
                prompt=prompt,
                image_file=image_path,
                output_path=output_dir,
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=False,
                test_compress=False,
                eval_mode=True
            )
    except Exception as exc:
        logger.exception("DeepSeek OCR inference failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            os.unlink(image_path)
        except OSError:
            pass
    # res is a string in eval mode or dict otherwise
    if isinstance(res, str):
        text = res.strip()
    elif isinstance(res, dict):
        text = res.get("text") or res.get("result") or ""
    else:
        logger.error("DeepSeek OCR returned %s instead of supported types", type(res))
        raise HTTPException(status_code=500, detail="Model produced no structured output.")

    return JSONResponse({"text": text})
