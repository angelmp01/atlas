# Dockerfile for ATLAS API
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
# Note: If using psycopg2-binary, no gcc needed. If using psycopg2, uncomment build dependencies below.
RUN pip install --no-cache-dir -r requirements.txt

# Uncomment if you need to compile C extensions (e.g., using psycopg2 instead of psycopg2-binary):
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     gcc \
#     libpq-dev \
#     && pip install --no-cache-dir -r requirements.txt \
#     && apt-get purge -y --auto-remove gcc libpq-dev \
#     && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/', timeout=5)"

# Run uvicorn
CMD ["uvicorn", "atlas_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
