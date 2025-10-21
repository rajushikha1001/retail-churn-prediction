# Use lightweight image for serving
FROM python:3.9-slim

WORKDIR /app

# Copy only the necessary files for inference
COPY requirements.txt .
COPY deploy/app.py .
COPY src/model.py ./src/
COPY model/churn_model.pkl ./model/

# Install only the runtime dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the FastAPI port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
