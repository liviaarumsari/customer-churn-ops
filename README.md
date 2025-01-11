# How to Run the Project

## Prerequisites

- [Minikube Installed](https://minikube.sigs.k8s.io/docs/start/)
- [Docker Installed](https://www.docker.com/products/docker-desktop/)

## 1. **GitHub Setup**

### Generate Personal Access Token

1. Go to `Settings` > `Developer Settings` > `Personal access tokens`.
2. Click `Generate new token`.
3. Save the token securely as it will be used for authentication.

## 2. **Minikube Setup**

### Start Minikube

```bash
minikube start
```

### Configure Docker (Windows) to Use Minikube's Environment

```powershell
minikube docker-env --shell powershell | Invoke-Expression
```

### Create Secret for GitHub Container Registry

Replace `<github-username>` and `<github-personal-access-token>` with your GitHub credentials.

```bash
kubectl create secret docker-registry ghcr-secret `
  --docker-server=ghcr.io `
  --docker-username=<github-username> `
  --docker-password=<github-personal-access-token>
```

## 3. **Build Stage**

### Build Spark Preprocessor Image

```bash
cd spark/preprocessor
docker build -t spark-preprocessor:latest -f Dockerfile .
cd ../..
```

### Build MLFlow App Image

```bash
cd mlflow/app
docker build -t mlflow-job:latest .
cd ../..
```

### Build Airflow Image

```bash
cd airflow
docker build -t airflow:latest .
cd ..
```

## 4. **Deploy Stage**

### Deploy Minio

```bash
kubectl apply -f minio/minio-deployment.yaml
kubectl apply -f minio/minio-service.yaml
```

### Deploy MLFlow

```bash
kubectl apply -f mlflow/mlflow-deployment.yaml
kubectl apply -f mlflow/mlflow-service.yaml
```

### Deploy Airflow

```bash
kubectl apply -f airflow/airflow-deployment.yaml
```

## 5. **Expose Services**

### Expose Minio Console

```bash
kubectl port-forward service/minio-service 8001:8001
```

### Expose MLFlow Console

```bash
kubectl port-forward service/mlflow-service 8002:8002
```

### Expose Airflow Console

```bash
kubectl port-forward service/airflow-service 8080:8080
```

## Notes

- Ensure that Minikube is running and configured correctly before starting the deployment.
- Use the specified ports to access the web consoles:
  - **Minio Console**: [http://localhost:8001](http://localhost:8001)
  - **MLFlow Console**: [http://localhost:8002](http://localhost:8002)
  - **Airflow Console**: [http://localhost:8080](http://localhost:8080)
- If you encounter any issues, check the logs using:
  ```bash
  kubectl logs <pod-name>
  ```
