# Dockerfile for ATLAS Web
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8001/', timeout=5)"

# Run uvicorn
CMD ["uvicorn", "atlas_web.main:app", "--host", "0.0.0.0", "--port", "8001"]
