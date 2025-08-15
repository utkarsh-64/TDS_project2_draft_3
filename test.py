import google.generativeai as genai

# Store chat sessions for both parsing and answering
parse_chat_sessions = {}
answer_chat_sessions = {}

def get_chat_session(sessions_dict, session_id, system_prompt, model_name="gemini-1.5-pro"):
    """
    Get or create a persistent chat session for a given session_id.
    """
    if session_id not in sessions_dict:
        model = genai.GenerativeModel(model_name)
        chat = model.start_chat(history=[
            {"role": "system", "parts": system_prompt}
        ])
        sessions_dict[session_id] = chat    
    return sessions_dict[session_id]

# ------------------------
# PARSE QUESTION FUNCTION
# ------------------------
def parse_question_with_llm(question_text=None, metadata=None, session_id="default_parse", retry_message=None):
    """
    Parse question with persistent chat session.
    - If retry_message is provided, sends only that to continue conversation.
    """
    SYSTEM_PROMPT = """
    You are a data extraction and scraping code generator.
    Your job: Take a user question and metadata about available files or URLs,
    then return ONLY a JSON with:
      - "code": Python scraping code
      - "libraries": list of pip packages required (exclude built-ins)
    Do NOT add explanations or extra text.
    """

    chat = get_chat_session(parse_chat_sessions, session_id, SYSTEM_PROMPT)

    if retry_message:
        # Only send error/retry message
        prompt = f"Previous code failed with: {retry_message}. Please fix the code."
    else:
        prompt = f"""
        Question:
        {question_text}

        Metadata:
        {metadata}
        """

    response = chat.send_message(prompt)
    return response.text

# ------------------------
# ANSWER WITH DATA FUNCTION
# ------------------------
def answer_with_data(question_text=None, extracted_data=None, session_id="default_answer", retry_message=None):
    """
    Answer analytical question with persistent chat session.
    - If retry_message is provided, sends only that to continue conversation.
    """
    SYSTEM_PROMPT = """
    You are a data analysis assistant.
    Given the extracted data (tables, lists, text, images in base64),
    and the question, return ONLY in the specified answer format from the question.
    """

    chat = get_chat_session(answer_chat_sessions, session_id, SYSTEM_PROMPT)

    if retry_message:
        prompt = f"Previous answer failed or was incomplete: {retry_message}. Please correct it."
    else:
        prompt = f"""
        Question:
        {question_text}

        Extracted Data:
        {extracted_data}
        """

    response = chat.send_message(prompt)
    return response.text
