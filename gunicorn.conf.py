import multiprocessing

# Gunicorn configuration
bind = "0.0.0.0:10000"  # Render expects the app to listen on port 10000
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 120  # Increased timeout for long-running PDF operations
keepalive = 5
max_requests = 1000
max_requests_jitter = 50

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info" 