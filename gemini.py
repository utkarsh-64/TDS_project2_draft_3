import os
import json
import google.generativeai as genai

api_key = os.getenv("GENAI_API_KEY")

if not api_key:
    raise ValueError("GENAI_API_KEY environment variable is not set.")

genai.configure(api_key=api_key)

MODEL_NAME = "gemini-1.5-flash-latest" 

generation_config = genai.types.GenerationConfig(
    response_mime_type="application/json"
)

parse_chat_sessions = {}
answer_chat_sessions = {}

async def get_chat_session(sessions_dict, session_id, system_prompt, model_name=MODEL_NAME):
    if session_id not in sessions_dict:
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            system_instruction=system_prompt
        )        
        chat = model.start_chat(history=[])
        sessions_dict[session_id] = chat    
    return sessions_dict[session_id]

# ------------------------
# PARSE QUESTION FUNCTION
# ------------------------
async def parse_question_with_llm(question_text=None, uploaded_files=None, db_schemas=None, csv_headers=None, session_id="default_parse", retry_message=None, folder="uploads"):
    SYSTEM_PROMPT = f"""
You are a Python code generation assistant. Your task is to generate a JSON object containing Python code to scrape data, a list of required libraries, and the user's questions.

RULES:
- If database schemas or CSV headers are provided, you MUST use them to write correct SQL queries and pandas code. Do not invent table, column, or header names. Pay close attention to capitalization.
- The generated code must save data to the '{folder}' directory.
- The code must also generate a '{folder}/metadata.txt' file.
- Do NOT include 'sqlite3', 'base64', or 'csv' in the libraries list, as they are built-in Python libraries.
- Respond ONLY with a valid JSON object matching this schema: {{"code": "...", "libraries": [...], "questions": [...]}}
- Do NOT include explanations or any text outside the JSON response.
"""

    chat = await get_chat_session(parse_chat_sessions, session_id, SYSTEM_PROMPT)

    if retry_message:
        prompt = f"The previous code failed with this error: <error>{retry_message}</error>. Please generate a corrected JSON response. Pay close attention to the provided database schemas and CSV headers."
    else:
        db_schema_prompt_part = ""
        if db_schemas:
            db_schema_prompt_part = f'''
Database Schemas:
{json.dumps(db_schemas, indent=2)}
'''
        csv_headers_prompt_part = ""
        if csv_headers:
            csv_headers_prompt_part = f'''
CSV Headers:
{json.dumps(csv_headers, indent=2)}
'''
        prompt = f"""
User Question:
{question_text}

Uploaded Files:
{uploaded_files}
{db_schema_prompt_part}
{csv_headers_prompt_part}
Generate the JSON response as instructed.
"""

    file_path = os.path.join(folder, "metadata.txt")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("")
    
    response = chat.send_message(prompt)
    return json.loads(response.text)

# ------------------------
# ANSWER WITH DATA FUNCTION
# ------------------------
async def answer_with_data(question_text=None, session_id="default_answer", retry_message=None, folder="uploads"):
    metadata_path = os.path.join(folder, "metadata.txt")
    with open(metadata_path, "r") as file:
        metadata = file.read()

    SYSTEM_PROMPT = f"""
You are a Python code generation assistant. Your task is to generate a JSON object containing Python code to analyze data and a list of required libraries.

CRITICAL RULES:
1.  **Strictly Adhere to Output Format**: The user's question will describe a required JSON output format. Your generated Python code MUST produce a `result.json` file that EXACTLY matches this structure. All specified keys must be present.
2.  **Handle Missing Values**: If a value for a required key cannot be calculated, your code must include the key in the JSON with a default value (e.g., 0 for numbers, "" for strings, [] for lists). DO NOT omit keys.
3.  **Handle Charting Errors**: If code to generate a base64 chart fails for any reason, it MUST catch the exception and use an empty string "" as the value for that chart's key in the final JSON. This will prevent invalid base64 errors.
4.  **No Built-in Libraries**: Do NOT include 'sqlite3', 'base64', or 'csv' in the `libraries` list.
5.  **JSON Output Only**: Respond ONLY with a valid JSON object matching this schema: {{"code": "...", "libraries": [...]}}. Do not include any explanations.
"""

    chat = await get_chat_session(answer_chat_sessions, session_id, SYSTEM_PROMPT)

    if retry_message:
        prompt = f"The previous code failed with this error: <error>{retry_message}</error>. Please generate a corrected JSON response, paying close attention to the critical rules."
    else:
        prompt = f"""
User Questions & Required JSON Format:
{question_text}

Data Metadata:
{metadata}

Generate the JSON response as instructed. Ensure the Python code you generate strictly follows all the rules, especially regarding the final JSON output structure and error handling for charts.
"""

    file_path = os.path.join(folder, "result.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("")

    response = chat.send_message(prompt)
    return json.loads(response.text)
