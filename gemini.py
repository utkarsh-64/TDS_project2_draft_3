import os
import json
import google.generativeai as genai

# Get the API key from environment variable
api_key = os.getenv("GENAI_API_KEY")

if not api_key:
    raise ValueError("GENAI_API_KEY environment variable is not set.")

genai.configure(api_key=api_key)

# CORRECTED MODEL NAME
MODEL_NAME = "gemini-1.5-flash-latest" 

# Give response in JSON format
generation_config = genai.types.GenerationConfig(
    response_mime_type="application/json"
)

# Store chat sessions for both parsing and answering
parse_chat_sessions = {}
answer_chat_sessions = {}

# Get or create a persistent chat session for a given session_id.
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
async def parse_question_with_llm(question_text=None, uploaded_files=None, db_schemas=None, session_id="default_parse", retry_message=None, folder="uploads"):
    """
    Parse question with persistent chat session.
    - If retry_message is provided, sends only that to continue conversation.
    """
    SYSTEM_PROMPT = f"""
You are a Python code generation assistant. Your task is to generate a JSON object containing Python code to scrape data, a list of required libraries, and the user's questions.

RULES:
- If a database schema is provided, you MUST use it to write correct SQL queries. Do not invent table or column names.
- The generated code must save data to the '{folder}' directory.
- The code must also generate a '{folder}/metadata.txt' file containing dataframe info, column names, and the first few rows.
- If the user provides an ANSWER_FORMAT, copy it verbatim into the metadata file. Otherwise, use "ANSWER_FORMAT: JSON".
- Do NOT include 'sqlite3' in the libraries list, as it is a built-in Python library.
- Respond ONLY with a valid JSON object matching this schema: {{"code": "...", "libraries": [...], "questions": [...]}}
- Do NOT include explanations or any text outside the JSON response.
"""

    chat = await get_chat_session(parse_chat_sessions, session_id, SYSTEM_PROMPT)

    if retry_message:
        prompt = f"The previous code failed with this error: <error>{retry_message}</error>. Please generate a corrected JSON response. Pay close attention to the provided database schema if the error is SQL-related."
    else:
        db_schema_prompt_part = ""
        if db_schemas:
            db_schema_prompt_part = f'''
Database Schemas:
{json.dumps(db_schemas, indent=2)}
'''
        prompt = f"""
User Question:
{question_text}

Uploaded Files:
{uploaded_files}
{db_schema_prompt_part}
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
    """
    Answer analytical question with persistent chat session.
    - If retry_message is provided, sends only that to continue conversation.
    """
    metadata_path = os.path.join(folder, "metadata.txt")
    with open(metadata_path, "r") as file:
        metadata = file.read()

    SYSTEM_PROMPT = f"""
You are a Python code generation assistant. Your task is to generate a JSON object containing Python code to analyze data and a list of required libraries.

RULES:
- The generated code must answer the user's question based on the provided metadata.
- The code MUST save the final answer as a JSON file to '{folder}/result.json'.
- The code must adhere to the 'ANSWER_FORMAT' specified in the metadata.
- If visualizations are created, they must be saved as base64-encoded PNGs within the result JSON.
- Do NOT include 'sqlite3' in the libraries list, as it is a built-in Python library.
- Respond ONLY with a valid JSON object matching this schema: {{"code": "...", "libraries": [...]}}
- Do NOT include explanations or any text outside the JSON response.
"""

    chat = await get_chat_session(answer_chat_sessions, session_id, SYSTEM_PROMPT)

    if retry_message:
        prompt = f"The previous code failed with this error: <error>{retry_message}</error>. Please generate a corrected JSON response."
    else:
        prompt = f"""
User Questions:
{question_text}

Data Metadata:
{metadata}

Generate the JSON response as instructed.
"""

    file_path = os.path.join(folder, "result.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("")

    response = chat.send_message(prompt)
    return json.loads(response.text)
