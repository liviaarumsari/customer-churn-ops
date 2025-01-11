# Use a lightweight Python image
FROM python:3.9-slim

# Set environment variables for MinIO and MLflow
ENV MLFLOW_S3_ENDPOINT_URL=http://minio-service.default.svc.cluster.local:8000
ENV AWS_ACCESS_KEY_ID=minioadmin
ENV AWS_SECRET_ACCESS_KEY=minioadmin

# Set the working directory in the container
WORKDIR /app

# Copy the required files into the container
COPY main.py .
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the main script
CMD ["python", "main.py"]
