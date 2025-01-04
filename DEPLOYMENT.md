# Minikube

minikube start

minikube docker-env --shell powershell | Invoke-Expression

# Build Stage

## Build spark preprocessor image

cd spark/preprocessor

docker build -t spark-preprocessor:latest -f Dockerfile .

cd ../..

## Build mlflow app image

cd mlflow/app

docker build -t mlflow-job:latest .

cd ../..

## Build airflow image

cd airflow

docker build -t airflow:latest .

cd ..

# Deploy Stage

## Deploy Minio

kubectl apply -f minio/minio-deployment.yaml

kubectl apply -f minio/minio-service.yaml

## Deploy MLFlow

kubectl apply -f mlflow/mlflow-deployment.yaml

kubectl apply -f mlflow/mlflow-service.yaml

## Deploy Airflow

kubectl apply -f airflow/airflow-deployment.yaml

# Expose

## Expose Minio Console

kubectl port-forward service/minio-service 8001:8001

## Expose Airflow Console

kubectl port-forward service/airflow-service 8080:8080
