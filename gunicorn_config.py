import multiprocessing
# Gunicorn config
bind = "0.0.0.0:10000"
workers = 1  # CHANGE THIS - Force single worker for now
timeout = 300
keepalive = 120
worker_class = "sync"
max_requests = 1000
max_requests_jitter = 50
