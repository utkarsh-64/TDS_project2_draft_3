import subprocess
import sys
import traceback
from typing import List
import datetime
import os
import io
import black  # For pretty-printing code


async def run_python_code(code: str, libraries: List[str], folder: str = "uploads") -> dict:
    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)

    # File where we’ll log execution results
    log_file_path = os.path.join(folder, "execution_result.txt")

    def log_to_file(content: str):
        """Append timestamped content to the log file."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"\n[{timestamp}]\n{content}\n{'-'*40}\n")

    def execute_code():
        exec_globals = {}
        exec(code, exec_globals)

    # Step 1: Install all required libraries first
    for lib in libraries:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
        except Exception as install_error:
            error_message = f"❌ Failed to install library '{lib}':\n{install_error}"
            log_to_file(error_message)
            return {"code": 0, "output": error_message}

    # Step 2: Execute the code after installation
    try:
        # Pretty-print the code before execution
        try:
            code_formatted = black.format_str(code, mode=black.Mode())
        except Exception:
            code_formatted = code  # Fallback if formatting fails

        # Save the pretty-printed code to the log before running
        log_to_file(f"📜 Executing Code:\n{code_formatted}")

        execute_code()
        success_message = "✅ Code executed successfully after installing libraries."
        log_to_file(success_message)
        return {"code": 1, "output": success_message}

    except Exception as e:
        error_details = f"❌ Error during code execution:\n{traceback.format_exc()}"
        log_to_file(error_details)
        return {"code": 0, "output": error_details}
