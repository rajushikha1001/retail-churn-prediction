# Base image
FROM python:3.9-slim

# Set work directory
WORKDIR /app

# Copy dependencies first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/
COPY data/ ./data/

# Default command (run tests)
CMD ["pytest", "--maxfail=1", "--disable-warnings", "-q"]
