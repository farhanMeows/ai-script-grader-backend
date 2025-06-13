import multiprocessing
import os

# Calculate workers based on available memory
# For 512MB instance, use fewer workers
workers = 2  # Reduced from multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 300  # Increased timeout for large file uploads
keepalive = 5
max_requests = 100  # Reduced to recycle workers more frequently
max_requests_jitter = 20
limit_request_line = 0  # No limit on request line length
limit_request_fields = 0  # No limit on number of request fields
limit_request_field_size = 0  # No limit on size of request fields

# Memory optimization settings
max_worker_connections = 1000
worker_connections = 1000
worker_tmp_dir = "/dev/shm"  # Use RAM-based temporary directory
worker_class = "uvicorn.workers.UvicornWorker"

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Memory management
preload_app = True  # Preload application to share memory between workers
max_worker_lifetime = 3600  # Restart workers after 1 hour to prevent memory leaks 