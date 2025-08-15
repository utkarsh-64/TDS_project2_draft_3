#!/bin/bash

# --preload: Loads application code before forking workers for better memory usage and stability.
# --timeout 300: Sets a 5-minute timeout for workers.
# -w 1: Uses a single worker to conserve memory on the free tier.
gunicorn --preload --timeout 300 -w 1 -k uvicorn.workers.UvicornWorker main:app -b 0.0.0.0:$PORT
