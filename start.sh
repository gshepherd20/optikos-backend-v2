#!/bin/bash
echo "Starting Flask app on port: $PORT"
exec gunicorn --bind 0.0.0.0:$PORT --log-level debug --access-logfile - --error-logfile - app:app
