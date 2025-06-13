import multiprocessing
import os

# Get port from environment variable (Render sets this)
port = int(os.getenv("PORT", "10000"))

# Gunicorn configuration
bind = f"0.0.0.0:{port}"  # Use the port from environment variable
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 300  # Increased timeout for large file uploads
keepalive = 5
max_requests = 1000
max_requests_jitter = 50
limit_request_line = 0  # No limit on request line length
limit_request_fields = 0  # No limit on number of request fields
limit_request_field_size = 0  # No limit on size of request fields

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info" 