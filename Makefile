.PHONY: all start_minikube deploy_minio build_spark_images load_images deploy_spark deploy_airflow verify

all: start_minikube deploy_minio build_spark_images load_images deploy_spark deploy_airflow verify

start_minikube:
	@echo "Starting Minikube with insecure registry..."
	minikube start

deploy_minio:
	@echo "Deploying MinIO..."
	kubectl apply -f minio/minio-deployment.yaml

build_spark_images:
	@echo "Building Docker images for Spark jobs..."
	@echo "Building spark-preprocessor image..."
	cd spark/preprocessor && docker build -t spark-preprocessor:latest -f Dockerfile . && cd ../..

load_images:
	@echo "Loading Docker images into Minikube..."
	minikube image load docker.io/library/spark-preprocessor:latest

deploy_spark:
	@echo "Deploying Spark jobs..."
	kubectl apply -f spark/preprocessor/spark-preprocessor-job.yaml

deploy_airflow:
	@echo "Deploying Airflow..."
	@echo "Building Airflow Docker image..."
	cd airflow && docker build -t airflow:latest . && cd ..
	minikube image load docker.io/library/airflow:latest
	kubectl apply -f airflow/airflow-deployment.yaml

verify:
	@echo "Checking MinIO deployment status..."
	kubectl rollout status deployment/minio
	@echo "Checking Airflow deployment status..."
	kubectl rollout status deployment/airflow
	@echo "Checking Spark job statuses..."
	kubectl get jobs
	@echo "All tasks completed successfully!"

# NOTE: DO NOT RUN THIS
ssh-into-airflow:
	kubectl exec -it airflow-64f54b9979-fjg7g -- /bin/bash

airflow-create-user:
	airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

change-to-minikube-ctx:
	minikube docker-env --shell powershell | Invoke-Expression