#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Start Minikube with insecure registry
echo "Starting Minikube with insecure registry..."
minikube start

# Deploy MinIO
echo "Deploying MinIO..."
kubectl apply -f minio/minio-deployment.yaml

# Build Spark Containers
echo "Building Docker images for Spark jobs..."

# Build spark-preprocessor image
echo "Building spark-preprocessor image..."
cd spark/preprocessor
docker build -t spark-preprocessor:latest -f Dockerfile .
cd ../..

# Build spark-psi-worker image
echo "Building spark-psi-worker image..."
cd spark/psi-worker
docker build -t spark-psi-worker:latest -f Dockerfile .
cd ../..

# Load Docker images into Minikube
echo "Loading Docker images into Minikube..."
minikube image load spark-preprocessor:latest
minikube image load spark-psi-worker:latest

# Deploy Spark jobs
echo "Deploying Spark jobs..."
kubectl apply -f spark/preprocessor/spark-preprocessor-job.yaml
kubectl apply -f spark/psi-worker/spark-psi-worker-job.yaml

# Deploy MLflow Tracking Server
echo "Deploying MLflow Tracking Server..."
kubectl apply -f mlflow/mlflow-deployment.yaml
kubectl apply -f mlflow/mlflow-service.yaml

# Deploy MLflow mock job
echo "Deploying MLflow mock job..."
kubectl apply -f mlflow/mlflow-job.yaml

# Verify Deployment
echo "Checking MinIO deployment status..."
kubectl rollout status deployment/minio

echo "Checking Spark job statuses..."
kubectl get jobs

echo "All tasks completed successfully!"
