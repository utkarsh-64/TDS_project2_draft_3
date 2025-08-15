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
import sqlite3 # <-- Import sqlite3

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

# --- NEW HELPER FUNCTION TO GET DB SCHEMA ---
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
            # Format: { 'table_name': ['col1 (type)', 'col2 (type)'] }
            schema_info[table_name] = [f"{col[1]} ({col[2]})" for col in columns]
        conn.close()
        return schema_info
    except Exception as e:
        # Log the error, but don't crash the app
        print(f"Error reading schema from {db_path}: {e}")
        return {"error": f"Could not read schema from {db_path}: {e}"}

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("frontend.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Helper funtion to show last 25 words of string s
def last_n_words(s, n=100):
    s = str(s)
    words = s.split()
    return ' '.join(words[-n:])

def is_csv_empty(csv_path):
    return not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0



@app.post("/api")
async def analyze(request: Request):
    # Create a unique folder for this request
    request_id = str(uuid.uuid4())
    request_folder = os.path.join(UPLOAD_DIR, request_id)
    os.makedirs(request_folder, exist_ok=True)

    # Setting up file for llm response
    llm_response_file_path = os.path.join(request_folder, "llm_response.txt")
    
    # Setup logging for this request
    log_path = os.path.join(request_folder, "app.log")
    logger = logging.getLogger(request_id)
    logger.setLevel(logging.INFO)
    # Remove previous handlers if any (avoid duplicate logs)
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # Also log to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info("Step-1: Folder created: %s", request_folder)

    form = await request.form()
    question_text = None
    saved_files = {}

    # Save all uploaded files to the request folder
    for field_name, value in form.items():
        if hasattr(value, "filename") and value.filename:  # It's a file
            file_path = os.path.join(request_folder, value.filename)
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(await value.read())
            saved_files[field_name] = file_path

            # If it's questions.txt, read its content
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

    # --- ADDITION: Extract schema from any .db files ---
    db_schemas = {}
    for file_name, file_path in saved_files.items():
        if file_path.endswith(".db"):
            schema = get_db_schema(file_path)
            db_schemas[file_name] = schema
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
                    db_schemas=db_schemas,  # <-- Pass the schema to the LLM
                    folder=request_folder,
                    session_id=request_id
                )
            else:
                response = await parse_question_with_llm(retry_message=retry_message, folder=request_folder, session_id=request_id)
            if isinstance(response, dict):
                with open(llm_response_file_path, "a") as f:
                    result = response
                    result["comment"] = f"Step-3: Getting scrap code and metadata from llm. Tries count = %d {attempt}"
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
   
    logger.info("Step-4: Execution result of the scrape code: %s", last_n_words(execution_result["output"]))

    count = 0
    while execution_result["code"] == 0 and count < 3:
        logger.error("Step-4: Error occured while scrapping. Tries count = %d", count)
        retry_message = last_n_words(str(execution_result["output"]), 100)
        try:
            # Pass schema info on retry as well
            response = await parse_question_with_llm(retry_message=retry_message, session_id=request_id, folder=request_folder, db_schemas=db_schemas)
            with open(llm_response_file_path, "a") as f:
                    result = response
                    result["comment"] = f"Step-4: Error occured while scrapping. Tries count = %d, {count}"
                    json.dump(result, f, indent=4)
        except Exception as e:
            logger.error("Step-4: error occured while reading json. %s", last_n_words(e))

        logger.info("Step-3: Response from scrapping: %s", last_n_words(response))
        execution_result = await run_python_code(response["code"], response["libraries"], folder=request_folder)
        logger.info("Step-4: Execution result of the scrape code: %s", last_n_words(execution_result["output"]))
        
        csv_path = os.path.join(request_folder, "data.csv")
        csv_retry_count = 0
        max_csv_retries = 3
        while os.path.exists(csv_path) and is_csv_empty(csv_path) and csv_retry_count < max_csv_retries:
            logger.warning("data.csv is present but empty. Retrying code generation and  execution. Attempt %d", csv_retry_count + 1)
            response = await parse_question_with_llm(retry_message=str("There is no content present in scraped data files."), folder=request_folder, session_id=request_id, db_schemas=db_schemas)
            with open(llm_response_file_path, "a") as f:
                    result = response
                    result["comment"] = f"data.csv is present but empty. Retrying code generation and  execution. Attempt %d, {csv_retry_count + 1}"
                    json.dump(result, f, indent=4)
            execution_result = await run_python_code(response["code"], response["libraries"], folder=request_folder)
            csv_retry_count += 1
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
                logger.info("Step-5: Response from llm: %s", last_n_words(gpt_ans["code"]))
                if isinstance(gpt_ans, dict):
                    break
            else:
                gpt_ans = await answer_with_data(retry_message=retry_message, folder=request_folder, session_id=request_id)
        except Exception as e:
            error_occured = 1
            logger.error("Step-5: Error: %s", e)
            retry_message = last_n_words(str(e), 100)
        attempt += 1
    
    if not isinstance(gpt_ans, dict):
        logger.error("Error: Could not get valid response from answer_with_data after retries.")
        return JSONResponse({"message": "Error: Could not get valid response from answer_with_data after retries."})
    
    try:
        logger.info("Step-6: Executing final code. Tries count = 0")
        final_result = await run_python_code(gpt_ans["code"], gpt_ans["libraries"], folder=request_folder)
        logger.info("Step-6: Executing final code result: %s", last_n_words(final_result["output"]))
    except Exception as e:
        logger.error("Step-6: Trying after it caught under except block-wrong json format. Tries count = 1 %s", last_n_words(e))
        gpt_ans = await answer_with_data(retry_message=str("Please follow the json structure"), folder=request_folder, session_id=request_id)
        logger.info("Step-5: Response from llm: %s", last_n_words(gpt_ans["code"]))
        final_result = await run_python_code(gpt_ans["code"], gpt_ans["libraries"], folder=request_folder)
        logger.info("Step-6: Executing final code result: %s", last_n_words(final_result["output"]))
        
    count = 0
    json_str = 1
    while final_result["code"] == 0 and count < 3:
        logger.error("Step-6: Error occured while executing code. Tries count = %d", count+2)
        retry_message = last_n_words(str(final_result["output"]), 100)
        if json_str == 0:
            retry_message = "follow the structure {'code': '', 'libraries': ''}"
        logger.info("Step-5: Getting execution code from llm. Tries count = %d", attempt+1)
        try:
            gpt_ans = await answer_with_data(retry_message=retry_message,session_id=request_id, folder=request_folder)
            logger.info("Step-5: Response from llm: %s", last_n_words(gpt_ans["code"]))
        except Exception as e:
            logger.error("Step-5: Json parsing error. %s", last_n_words(e))
            json_str = 0
        try:
            final_result = await run_python_code(gpt_ans["code"], gpt_ans["libraries"], folder=request_folder)
            logger.info("Step-6: Executing final code result: %s", last_n_words(final_result["output"]))
            json_str = 1
        except Exception as e:
            logger.error("Exception occurred: %s", e)
            count -= 1
        count += 1

    result_path = os.path.join(request_folder, "result.json")
    with open(result_path, "r") as f:
        try:
            data = json.load(f)
            logger.info("Step-7: send result back")
            return JSONResponse(content=data)
        except Exception as e:
            logger.error("Step-7: Error occur while sending result: %s", last_n_words(e))
            f.seek(0)
            raw_content = f.read()
            return JSONResponse({"message": f"Error occured while processing result.json: {e}", "raw_result": raw_content})
