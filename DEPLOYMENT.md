minikube start

kubectl apply -f k8s/ingress.yaml

minikube tunnel

Minio:
kubectl apply -f minio/minio-deployment.yaml
kubectl apply -f minio/minio-service.yaml

Spark:
Build Container:

cd spark/preprocessor
docker build -t spark-preprocessor:latest -f Dockerfile .
cd ../psi-worker
docker build -t spark-psi-worker:latest -f Dockerfile .
cd ../..

minikube image load spark-preprocessor:latest
minikube image load spark-psi-worker:latest

Deploy:
kubectl apply -f spark/preprocessor/spark-preprocessor-job.yaml
kubectl apply -f spark/psi-worker/spark-psi-worker-job.yaml

MLFlow

cd mlflow/app
docker build -t mlflow-job:latest .
cd ../..

minikube image load mlflow-job:latest

kubectl apply -f mlflow/mlflow-deployment.yaml
kubectl apply -f mlflow/mlflow-service.yaml
kubectl apply -f mlflow/mlflow-job.yaml
