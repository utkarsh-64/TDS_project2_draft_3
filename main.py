# main.py (fixed)
import os
import uuid
import json
import logging
import asyncio
from typing import Optional, Dict, Any

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask

import aiofiles
from pathlib import Path

# ---- your modules (unchanged interface) ----
# They should keep the same function signatures used here.
from gemini import parse_question_with_llm, answer_with_data
from task_engine import run_python_code

# ---------------- App & CORS ----------------
app = FastAPI(title="Data Analyst Agent (Gemini-powered)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Config ----------------
PORT = int(os.getenv("PORT", "10000"))  # Render sets PORT
BASE_DIR = Path(__file__).parent.resolve()

# Use /tmp on Render (ephemeral but writable), local "uploads" otherwise
DEFAULT_UPLOAD_DIR = "/tmp/uploads" if os.getenv("RENDER") else str(BASE_DIR / "uploads")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", DEFAULT_UPLOAD_DIR))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

FRONTEND_FILE = BASE_DIR / "frontend.html"

# ---------------- Utilities ----------------

def new_request_dir() -> Path:
    req_id = str(uuid.uuid4())
    req_dir = UPLOAD_DIR / req_id
    req_dir.mkdir(parents=True, exist_ok=True)
    return req_dir


def setup_logger(req_dir: Path) -> logging.Logger:
    log_path = req_dir / "app.log"
    logger = logging.getLogger(f"req-{req_dir.name}")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers on reload
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    fh = logging.FileHandler(str(log_path))
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


async def save_upload(file: UploadFile, dest: Path) -> str:
    async with aiofiles.open(dest, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            await f.write(chunk)
    return str(dest)


def last_n_words(s, n=100):
    s = str(s or "")
    return " ".join(s.split()[-n:])


# ---------------- Routes ----------------
@app.get("/", response_class=HTMLResponse)
async def root():
    if FRONTEND_FILE.exists():
        return HTMLResponse(FRONTEND_FILE.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<h2>Data Analyst Agent API</h2><p>POST JSON or multipart to <code>/api</code></p>",
        status_code=200,
    )


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/favicon.ico")
async def favicon():
    return PlainTextResponse("", status_code=204)


# ------------- Core endpoint --------------
# Accept BOTH JSON and multipart in ONE route
@app.post("/api")
@app.post("/api/")
async def api(
    request: Request,
    # Multipart fields (only used when Content-Type is multipart/form-data)
    prompt_text: Optional[str] = Form(None),
    url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    prompt_file: Optional[UploadFile] = File(None),
):
    """
    Flexible ingestion:
    - application/json:
        { "prompt": "...", "url": "...", "files": [{"name":"...", "b64":"..."}] }
    - multipart/form-data: fields: prompt_text, url, file, image, prompt_file
    """
    req_dir = new_request_dir()
    logger = setup_logger(req_dir)
    logger.info("Request dir: %s", req_dir)

    # ---------- Parse input based on content-type ----------
    content_type = request.headers.get("content-type", "")
    prompt = None
    saved_files: Dict[str, str] = {}  # field_name -> absolute_path

    try:
        if "multipart/form-data" in content_type:
            # Form fields come from function params; stream is not consumed twice
            if prompt_file and not prompt_text:
                dest = req_dir / prompt_file.filename
                await save_upload(prompt_file, dest)
                # Treat the content of this uploaded file as the prompt
                async with aiofiles.open(dest, "r", encoding="utf-8", errors="ignore") as f:
                    prompt = await f.read()
                saved_files["prompt_file"] = str(dest)

            if prompt_text:
                prompt = prompt_text

            if url:
                saved_files["url"] = url

            if file and file.filename:
                dest = req_dir / file.filename
                await save_upload(file, dest)
                saved_files["file"] = str(dest)

            if image and image.filename:
                dest = req_dir / image.filename
                await save_upload(image, dest)
                saved_files["image"] = str(dest)

        else:
            # JSON (or raw text fallback)
            raw = await request.body()  # read ONCE
            if not raw:
                prompt = ""
            else:
                # try json first
                try:
                    payload = json.loads(raw.decode("utf-8", errors="ignore"))
                    prompt = payload.get("prompt", "")
                    url = payload.get("url") or url
                    # Optional: base64 files array: [{"name":"...","b64":"..."}]
                    files = payload.get("files") or []
                    for i, fobj in enumerate(files):
                        name = fobj.get("name") or f"file_{i}"
                        b64 = fobj.get("b64")
                        if not b64:
                            continue
                        import base64

                        dest = req_dir / name
                        async with aiofiles.open(dest, "wb") as f:
                            await f.write(base64.b64decode(b64))
                        saved_files[name] = str(dest)
                    if url:
                        saved_files["url"] = url
                except json.JSONDecodeError:
                    # treat as raw text prompt
                    prompt = raw.decode("utf-8", errors="ignore")
    except Exception as e:
        logger.exception("Error parsing request: %s", e)
        return JSONResponse({"error": f"Failed to parse input: {e}"}, status_code=400)

    prompt = (prompt or "").strip()
    logger.info("Prompt (tail): %s", last_n_words(prompt))

    # Validate prompt before calling Gemini
    if not prompt and not saved_files:
        logger.error("No prompt or files provided to process")
        return JSONResponse({"error": "No prompt or files provided in the request."}, status_code=400)

    # ---------- LLM Planning (Gemini) ----------
    llm_response_file_path = req_dir / "llm_response.txt"

    async def call_parse_with_retries(attempts: int = 3, backoff: float = 1.0):
        last_exc = None
        for i in range(attempts):
            try:
                logger.info("Calling parse_question_with_llm (attempt %d)", i + 1)
                plan = await parse_question_with_llm(
                    question_text=prompt,
                    uploaded_files=saved_files,
                    folder=str(req_dir),
                    session_id=req_dir.name,
                )
                return plan
            except Exception as e:
                last_exc = e
                logger.warning("parse_question_with_llm failed (attempt %d): %s", i + 1, last_n_words(e))
                # quick backoff
                await asyncio.sleep(backoff * (i + 1))
        raise last_exc

    try:
        plan = await call_parse_with_retries()
        if not isinstance(plan, dict):
            raise ValueError("parse_question_with_llm did not return dict")
        async with aiofiles.open(llm_response_file_path, "a", encoding="utf-8") as f:
            await f.write(json.dumps({"step": "plan", "plan": plan}, indent=2))
    except Exception as e:
        logger.exception("Gemini planning failed: %s", e)
        # If the underlying error is from the Gemini client and mentions internal server error,
        # provide a helpful message asking the caller to retry later.
        msg = str(e)
        if "InternalServerError" in msg or "internal error" in msg.lower():
            return JSONResponse({"error": "Gemini planning failed: 500 internal error from model. Please retry later."}, status_code=502)
        return JSONResponse({"error": f"Gemini planning failed: {e}"}, status_code=500)

    # ---------- Execute code for data prep ----------
    try:
        exec_result = await run_python_code(plan.get("code", ""), plan.get("libraries", []), folder=str(req_dir))
    except Exception as e:
        logger.exception("Executor failed (planning stage): %s", e)
        return JSONResponse({"error": f"Executor failed in planning stage: {e}"}, status_code=500)

    if exec_result.get("code") != 1:
        # Include partial output in response for debugging
        logger.error("Planning code execution failed: %s", last_n_words(exec_result.get("output")))
        return JSONResponse({"error": "Planning code execution failed", "detail": str(exec_result.get("output"))}, status_code=500)

    # ---------- Ask Gemini to generate final analysis code ----------
    async def call_answer_with_retries(attempts: int = 3, backoff: float = 1.0):
        last_exc = None
        for i in range(attempts):
            try:
                logger.info("Calling answer_with_data (attempt %d)", i + 1)
                analysis = await answer_with_data(
                    question_text=plan.get("questions", prompt),
                    folder=str(req_dir),
                    session_id=req_dir.name,
                )
                return analysis
            except Exception as e:
                last_exc = e
                logger.warning("answer_with_data failed (attempt %d): %s", i + 1, last_n_words(e))
                await asyncio.sleep(backoff * (i + 1))
        raise last_exc

    try:
        analysis = await call_answer_with_retries()
        if not isinstance(analysis, dict):
            raise ValueError("answer_with_data did not return dict")
    except Exception as e:
        logger.exception("Gemini analysis failed: %s", e)
        msg = str(e)
        if "InternalServerError" in msg or "internal error" in msg.lower():
            return JSONResponse({"error": "Gemini analysis failed: 500 internal error from model. Please retry later."}, status_code=502)
        return JSONResponse({"error": f"Gemini analysis failed: {e}"}, status_code=500)

    # ---------- Execute final analysis code ----------
    try:
        final_exec = await run_python_code(analysis.get("code", ""), analysis.get("libraries", []), folder=str(req_dir))
    except Exception as e:
        logger.exception("Executor failed (analysis stage): %s", e)
        return JSONResponse({"error": f"Executor failed in analysis stage: {e}"}, status_code=500)

    if final_exec.get("code") != 1:
        logger.error("Final code execution failed: %s", last_n_words(final_exec.get("output")))
        return JSONResponse({"error": "Final code execution failed", "detail": str(final_exec.get("output"))}, status_code=500)

    # ---------- Return result.json if present ----------
    result_path = req_dir / "result.json"

    async def cleanup(path: Path):
        # Optional: clean per-request folder later if you want (comment out to keep logs)
        # import shutil
        # shutil.rmtree(path, ignore_errors=True)
        pass

    if result_path.exists():
        try:
            async with aiofiles.open(result_path, "r", encoding="utf-8") as f:
                content = await f.read()
            data = json.loads(content)
            return JSONResponse(content=data, background=BackgroundTask(cleanup, req_dir))
        except Exception as e:
            # Return raw content if parse fails
            logger.exception("Failed to parse result.json: %s", e)
            raw = await aiofiles.open(result_path, "r", encoding="utf-8")
            raw_content = await raw.read()
            await raw.close()
            return JSONResponse(
                {"message": f"Error parsing result.json: {e}", "raw_result": raw_content},
                background=BackgroundTask(cleanup, req_dir),
            )

    # If the code didn’t create result.json, return generic success
    return JSONResponse(
        {"message": "Completed but result.json missing", "note": "Check logs in request folder."},
        background=BackgroundTask(cleanup, req_dir),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=bool(os.getenv("DEV", "")))
