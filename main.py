# ==============================================================================
# Imports
# ==============================================================================
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import aiofiles
import json
import logging
import difflib
import sqlite3
import csv

# Local application imports
from task_engine import run_python_code
from gemini import parse_question_with_llm, answer_with_data

# ==============================================================================
# FastAPI App Initialization
# ==============================================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ==============================================================================
# Helper Functions
# ==============================================================================

def setup_logging(request_id, request_folder):
    """Configures a dedicated logger for a single request."""
    log_path = os.path.join(request_folder, "app.log")
    logger = logging.getLogger(request_id)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Log to a file specific to the request
    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Also log to the console for real-time monitoring on Render
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    return logger

def get_db_schema(db_path):
    """Connects to a SQLite DB and extracts its schema."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        schema_info = {}
        for table_tuple in tables:
            table_name = table_tuple[0]
            cursor.execute(f"PRAGMA table_info('{table_name}');")
            columns = cursor.fetchall()
            schema_info[table_name] = [f"{col[1]} ({col[2]})" for col in columns]
        conn.close()
        return schema_info
    except Exception as e:
        print(f"Error reading schema from {db_path}: {e}")
        return {"error": f"Could not read schema from {db_path}: {e}"}

def get_csv_headers(file_path):
    """Reads the first row of a CSV file to get header columns."""
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f: # Use utf-8-sig to handle BOM
            reader = csv.reader(f)
            headers = next(reader)
            return headers
    except Exception as e:
        print(f"Error reading CSV headers from {file_path}: {e}")
        return {"error": f"Could not read headers from {file_path}: {e}"}

def last_n_words(s, n=100):
    """Utility to get the tail of a long string for concise logging."""
    s = str(s)
    words = s.split()
    return ' '.join(words[-n:])

# ==============================================================================
# API Endpoints
# ==============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serves the static HTML frontend."""
    with open("frontend.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.post("/api")
async def analyze(request: Request):
    """
    Main endpoint to handle file uploads, AI code generation, and execution.
    This function follows a multi-step pipeline:
    1. Setup: Create a unique directory and logger for the request.
    2. File Processing: Save uploaded files and extract schemas/headers.
    3. Scraping Code Generation: Ask the LLM to generate data scraping code.
    4. Scraping Code Execution: Run the generated code to get the data.
    5. Analysis Code Generation: Ask the LLM to generate data analysis code.
    6. Analysis Code Execution: Run the analysis code to get the final result.
    7. Return Result: Send back the contents of the final result.json.
    """
    # --- Stage 1: Setup ---
    request_id = str(uuid.uuid4())
    request_folder = os.path.join(UPLOAD_DIR, request_id)
    os.makedirs(request_folder, exist_ok=True)
    logger = setup_logging(request_id, request_folder)
    logger.info("Pipeline started. Request ID: %s", request_id)

    # --- Stage 2: File Processing ---
    try:
        form = await request.form()
        saved_files = {}
        question_text = None

        for field_name, value in form.items():
            if hasattr(value, "filename") and value.filename:
                file_path = os.path.join(request_folder, value.filename)
                async with aiofiles.open(file_path, "wb") as f:
                    await f.write(await value.read())
                saved_files[field_name] = file_path
                if "question" in field_name.lower():
                    question_text = (await aiofiles.open(file_path, "r")).read()

        if not question_text:
            logger.warning("No 'question.txt' found. Searching for a fallback.")
            # Fallback logic if a specific question file isn't found
            # (This part can be adapted based on specific needs)
            if saved_files:
                 # A simple fallback to the first file if no question file is identified
                first_file_path = next(iter(saved_files.values()))
                async with aiofiles.open(first_file_path, "r") as f:
                    question_text = await f.read()

        if not question_text:
            logger.error("Critical error: No question text could be determined from uploaded files.")
            raise HTTPException(status_code=400, detail="No question file was provided.")

        logger.info("Files saved: %s", saved_files.keys())
        
        db_schemas = {name: get_db_schema(path) for name, path in saved_files.items() if path.endswith(".db")}
        csv_headers = {name: get_csv_headers(path) for name, path in saved_files.items() if path.endswith(".csv")}
        logger.info("Extracted DB schemas: %s", db_schemas.keys())
        logger.info("Extracted CSV headers: %s", csv_headers.keys())

    except Exception as e:
        logger.error("Error during file processing: %s", e, exc_info=True)
        return JSONResponse(status_code=500, content={"message": "Failed during file processing.", "error": str(e)})

    # --- Stage 3 & 4: Scraping Code Generation and Execution ---
    logger.info("--- Entering Scraping Stage ---")
    max_attempts = 3
    execution_result = {"code": 0, "output": "Scraping stage not initiated."}

    for attempt in range(max_attempts):
        logger.info("Scraping attempt %d of %d", attempt + 1, max_attempts)
        try:
            # Generate code
            retry_message = last_n_words(execution_result["output"]) if attempt > 0 else None
            llm_response = await parse_question_with_llm(
                question_text=question_text,
                uploaded_files=saved_files,
                db_schemas=db_schemas,
                csv_headers=csv_headers,
                folder=request_folder,
                session_id=request_id,
                retry_message=retry_message
            )
            
            # Execute code
            execution_result = await run_python_code(llm_response.get("code", ""), llm_response.get("libraries", []), folder=request_folder)
            
            if execution_result["code"] == 1:
                logger.info("Scraping code executed successfully.")
                break  # Success, exit loop
            else:
                logger.warning("Scraping code execution failed. Output: %s", execution_result["output"])

        except Exception as e:
            logger.error("Exception during scraping stage: %s", e, exc_info=True)
            execution_result = {"code": 0, "output": str(e)}
    
    if execution_result["code"] != 1:
        logger.error("Scraping stage failed after all attempts.")
        return JSONResponse(status_code=500, content={"message": "Failed to generate and execute data scraping code.", "details": execution_result["output"]})

    # --- Stage 5 & 6: Analysis Code Generation and Execution ---
    logger.info("--- Entering Analysis Stage ---")
    final_result = {"code": 0, "output": "Analysis stage not initiated."}

    for attempt in range(max_attempts):
        logger.info("Analysis attempt %d of %d", attempt + 1, max_attempts)
        try:
            # Generate code
            retry_message = last_n_words(final_result["output"]) if attempt > 0 else None
            analysis_response = await answer_with_data(
                question_text=llm_response.get("questions"),
                folder=request_folder,
                session_id=request_id,
                retry_message=retry_message
            )

            # Execute code
            final_result = await run_python_code(analysis_response.get("code", ""), analysis_response.get("libraries", []), folder=request_folder)

            if final_result["code"] == 1:
                logger.info("Analysis code executed successfully.")
                break # Success, exit loop
            else:
                logger.warning("Analysis code execution failed. Output: %s", final_result["output"])

        except Exception as e:
            logger.error("Exception during analysis stage: %s", e, exc_info=True)
            final_result = {"code": 0, "output": str(e)}

    if final_result["code"] != 1:
        logger.error("Analysis stage failed after all attempts.")
        return JSONResponse(status_code=500, content={"message": "Failed to generate and execute final analysis code.", "details": final_result["output"]})

    # --- Stage 7: Return Result ---
    logger.info("--- Entering Final Result Stage ---")
    result_path = os.path.join(request_folder, "result.json")
    
    if not os.path.exists(result_path) or os.path.getsize(result_path) == 0:
        logger.error("Execution succeeded, but result.json is missing or empty.")
        return JSONResponse(status_code=500, content={"message": "Code executed but the result.json file was not created or is empty."})

    try:
        with open(result_path, "r") as f:
            data = json.load(f)
        logger.info("Success! Sending result back.")
        return JSONResponse(content=data)
    except Exception as e:
        logger.error("Failed to read or parse final result.json: %s", e, exc_info=True)
        return JSONResponse(status_code=500, content={"message": f"Error reading result.json: {e}"})
