#!/bin/bash

# This command starts the Gunicorn production server.
# -w 4: Specifies 4 worker processes to handle requests.
# -k uvicorn.workers.UvicornWorker: Tells Gunicorn to use Uvicorn for running an ASGI app (FastAPI).
# main:app: Points to the 'app' instance in your 'main.py' file.
# -b 0.0.0.0:$PORT: Binds the server to the host and port provided by Render's environment.
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app -b 0.0.0.0:$PORT
