from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import aiofiles
import json
import logging
from fastapi.responses import HTMLResponse
import difflib
import sqlite3
import csv # <-- Import csv

from task_engine import run_python_code
from gemini import parse_question_with_llm, answer_with_data

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- HELPER FUNCTION TO GET DB SCHEMA ---
def get_db_schema(db_path):
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

# --- NEW HELPER FUNCTION TO GET CSV HEADERS ---
def get_csv_headers(file_path):
    """Reads the first row of a CSV file to get header columns."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)
            return headers
    except Exception as e:
        print(f"Error reading CSV headers from {file_path}: {e}")
        return {"error": f"Could not read headers from {file_path}: {e}"}


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("frontend.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def last_n_words(s, n=100):
    s = str(s)
    words = s.split()
    return ' '.join(words[-n:])

def is_csv_empty(csv_path):
    return not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0



@app.post("/api")
async def analyze(request: Request):
    request_id = str(uuid.uuid4())
    request_folder = os.path.join(UPLOAD_DIR, request_id)
    os.makedirs(request_folder, exist_ok=True)

    llm_response_file_path = os.path.join(request_folder, "llm_response.txt")
    
    log_path = os.path.join(request_folder, "app.log")
    logger = logging.getLogger(request_id)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info("Step-1: Folder created: %s", request_folder)

    form = await request.form()
    question_text = None
    saved_files = {}

    for field_name, value in form.items():
        if hasattr(value, "filename") and value.filename:
            file_path = os.path.join(request_folder, value.filename)
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(await value.read())
            saved_files[field_name] = file_path

            if field_name == "question.txt":
                async with aiofiles.open(file_path, "r") as f:
                    question_text = await f.read()
        else:
            saved_files[field_name] = value

    if question_text is None and saved_files:
        target_name = "question.txt"
        file_names = list(saved_files.keys())
        closest_matches = difflib.get_close_matches(target_name, file_names, n=1, cutoff=0.6)
        if closest_matches:
            selected_file_key = closest_matches[0]
        else:
            selected_file_key = next(iter(saved_files.keys()))
        selected_file_path = saved_files[selected_file_key]
        async with aiofiles.open(selected_file_path, "r") as f:
            question_text = await f.read()

    logger.info("Step-2: File sent %s", saved_files)

    # --- ADDITION: Extract schema and headers ---
    db_schemas = {}
    csv_headers = {}
    for file_name, file_path in saved_files.items():
        if file_path.endswith(".db"):
            schema = get_db_schema(file_path)
            db_schemas[file_name] = schema
        elif file_path.endswith(".csv"):
            headers = get_csv_headers(file_path)
            csv_headers[file_name] = headers
    # --- END ADDITION ---

    max_attempts = 3
    attempt = 0
    response = None
    error_occured = 0
    
    while attempt < max_attempts:
        logger.info("Step-3: Getting scrap code and metadata from llm. Tries count = %d", attempt)
        try:
            if error_occured == 0:
                response = await parse_question_with_llm(
                    question_text=question_text,
                    uploaded_files=saved_files,
                    db_schemas=db_schemas,
                    csv_headers=csv_headers, # <-- Pass headers
                    folder=request_folder,
                    session_id=request_id
                )
            else:
                response = await parse_question_with_llm(retry_message=retry_message, folder=request_folder, session_id=request_id)
            if isinstance(response, dict):
                with open(llm_response_file_path, "a") as f:
                    result = response
                    result["comment"] = f"Step-3: Tries count = %d {attempt}"
                    json.dump(result, f, indent=4)
                break
        except Exception as e:
            error_occured = 1
            retry_message = last_n_words(str(e), 100) + str("Provide a valid JSON response")
            logger.error("Step-3: Error in parsing the result. %s", retry_message)
        attempt += 1

    if not isinstance(response, dict):
        logger.error("Error: Could not get valid response from LLM after retries.")
        return JSONResponse({"message": "Error: Could not get valid response from LLM after retries."})

    logger.info("Step-3: Response from scrapping: %s", last_n_words(response))

    execution_result = await run_python_code(response["code"], response["libraries"], folder=request_folder)
   
    logger.info("Step-4: Execution result: %s", last_n_words(execution_result["output"]))

    count = 0
    while execution_result["code"] == 0 and count < 3:
        logger.error("Step-4: Error occured while scrapping. Tries count = %d", count)
        retry_message = last_n_words(str(execution_result["output"]), 100)
        try:
            response = await parse_question_with_llm(
                retry_message=retry_message, 
                session_id=request_id, 
                folder=request_folder, 
                db_schemas=db_schemas, 
                csv_headers=csv_headers # <-- Pass headers on retry
            )
            with open(llm_response_file_path, "a") as f:
                    result = response
                    result["comment"] = f"Step-4: Error retry, Tries count = %d, {count}"
                    json.dump(result, f, indent=4)
        except Exception as e:
            logger.error("Step-4: error reading json. %s", last_n_words(e))

        logger.info("Step-3: Response from scrapping: %s", last_n_words(response))
        execution_result = await run_python_code(response["code"], response["libraries"], folder=request_folder)
        logger.info("Step-4: Execution result: %s", last_n_words(execution_result["output"]))
        
        count += 1

    if execution_result["code"] != 1:
        logger.error("error occured while scrapping.")
        return JSONResponse({"message": "error occured while scrapping."})

    max_attempts = 3
    attempt = 0
    gpt_ans = None
    response_questions = response["questions"]
    error_occured = 0

    while attempt < max_attempts:
        logger.info("Step-5: Getting execution code from llm. Tries count = %d", attempt)
        try:
            if error_occured == 0:
                gpt_ans = await answer_with_data(question_text=response_questions, folder=request_folder, session_id=request_id)
            else:
                gpt_ans = await answer_with_data(retry_message=retry_message, folder=request_folder, session_id=request_id)
            
            if isinstance(gpt_ans, dict):
                logger.info("Step-5: Response from llm: %s", last_n_words(gpt_ans.get("code", "")))
                break
        except Exception as e:
            error_occured = 1
            logger.error("Step-5: Error: %s", e)
            retry_message = last_n_words(str(e), 100)
        attempt += 1
    
    if not isinstance(gpt_ans, dict):
        logger.error("Error: Could not get valid response from answer_with_data after retries.")
        return JSONResponse({"message": "Error: Could not get valid response from answer_with_data after retries."})
    
    final_result = {"code": 0} # Default to failure
    count = 0
    while final_result["code"] == 0 and count < 3:
        try:
            logger.info("Step-6: Executing final code. Tries count = %d", count)
            final_result = await run_python_code(gpt_ans["code"], gpt_ans.get("libraries", []), folder=request_folder)
            logger.info("Step-6: Executing final code result: %s", last_n_words(final_result["output"]))
            if final_result["code"] == 1:
                break
        except Exception as e:
            logger.error("Step-6: Exception during execution: %s", last_n_words(e))
            final_result = {"code": 0, "output": str(e)}

        logger.error("Step-6: Error occured while executing code. Tries count = %d", count)
        retry_message = last_n_words(str(final_result.get("output", "")), 100)
        
        try:
            gpt_ans = await answer_with_data(retry_message=retry_message, session_id=request_id, folder=request_folder)
            logger.info("Step-5 (Retry): Response from llm: %s", last_n_words(gpt_ans.get("code", "")))
        except Exception as e:
            logger.error("Step-5 (Retry): Json parsing error. %s", last_n_words(e))
        
        count += 1

    result_path = os.path.join(request_folder, "result.json")
    with open(result_path, "r") as f:
        try:
            data = json.load(f)
            logger.info("Step-7: send result back")
            return JSONResponse(content=data)
        except Exception as e:
            logger.error("Step-7: Error sending result: %s", last_n_words(e))
            f.seek(0)
            raw_content = f.read()
            return JSONResponse({"message": f"Error processing result.json: {e}", "raw_result": raw_content})
