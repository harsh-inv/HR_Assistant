import multiprocessing

# Worker configuration
workers = 4  # Increased from 2 for better concurrency
worker_class = 'sync'
timeout = 600  # Increased from 300 to 10 minutes
graceful_timeout = 120
keepalive = 5

# Connection settings
bind = "0.0.0.0:10000"
backlog = 2048

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Process naming
proc_name = 'hr_assistant'

# Worker lifecycle
max_requests = 1000
max_requests_jitter = 50

# Preload app
preload_app = False  # Prevents memory issues during loading
