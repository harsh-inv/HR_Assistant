import multiprocessing

# Gunicorn config
bind = "0.0.0.0:10000"
workers = multiprocessing.cpu_count() * 2 + 1
timeout = 300  # 5 minutes (increase from default 30 seconds)
keepalive = 120
worker_class = "sync"
max_requests = 1000
max_requests_jitter = 50
