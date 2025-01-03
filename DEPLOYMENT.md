minikube start

# Minio:

## Setup & Run pod

kubectl apply -f minio/minio-deployment.yaml

kubectl apply -f minio/minio-service.yaml

kubectl port-forward service/minio-service 8001:8001

# Spark:

## Setup & Run pod

cd spark/preprocessor

docker build -t spark-preprocessor:latest -f Dockerfile .

cd ../..

minikube image load spark-preprocessor:latest

minikube image load spark-psi-worker:latest

kubectl apply -f spark/preprocessor/spark-preprocessor-job.yaml

kubectl apply -f spark/psi-worker/spark-psi-worker-job.yaml

# MLFlow

## Setup

cd mlflow/app

docker build -t mlflow-job:latest .

cd ../..

minikube image load mlflow-job:latest

## Run pod

kubectl apply -f mlflow/mlflow-deployment.yaml

kubectl apply -f mlflow/mlflow-service.yaml

kubectl apply -f mlflow/mlflow-job.yaml

# Airflow:

## Setup & Run pod

cd airflow && docker build -t airflow:latest . && cd ..

minikube image load airflow:latest

kubectl apply -f airflow/airflow-deployment.yaml

kubectl port-forward service/airflow 8080:8080

## Restart pod

kubectl delete pod -l app=airflow

## To Grant Cluster-Admin permission to `default` service account (to allow Airflow to trigger Spark Job)

kubectl create clusterrolebinding default-admin \
 --clusterrole=cluster-admin \
 --serviceaccount=default:default


# Web UI
1. Minio: http://localhost:8001 (minioadmin/minioadmin)
1. Airflow: http://localhost:8080 (admin/admin). DAG id: minio_to_spark