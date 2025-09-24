# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p Config Data

# Copy Python files individually to ensure all are included
COPY api.py .
COPY card_filter.py .
COPY card_name_trie.py .
COPY database_connection.py .
COPY embedding_processor.py .
COPY gpt_interface.py .
COPY queryExpansion.py .
COPY query_parameters.py .
COPY query_processing.py .
COPY text_processor.py .
COPY __init__.py .

# Copy configuration files
COPY Config/X509-cert-3753233507821277243.pem Config/
COPY Config/values.yaml Config/

# Copy essential data files first
COPY Data/cardTemplate.json Data/
COPY Data/sampleDoc.json Data/
COPY card_name_to_uuid.json .

# Copy large data file last to optimize build caching
COPY Data/AllPrintings.json Data/

# Set environment variables
ENV FLASK_APP=api.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["flask", "run", "--host=0.0.0.0"]

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1
