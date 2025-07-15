FROM python:3.11-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglx-mesa0 \
    libegl1-mesa \
    libgbm1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Use direct shell command for PORT expansion
CMD ["bash", "-c", "echo 'Starting Flask app on port:' $PORT && exec gunicorn --bind 0.0.0.0:$PORT --log-level debug --access-logfile - --error-logfile - app:app"]
