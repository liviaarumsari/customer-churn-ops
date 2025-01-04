import mlflow

MLFLOW_TRACKING_URL = 'http://mlflow-service.default.svc.cluster.local:8002'
MINIO_URL = "http://minio-service.default.svc.cluster.local:8000"

def main():
    # Set MLflow tracking URI
    # mlflow.set_tracking_uri(MLFLOW_TRACKING_URL)

    # # Start a new MLflow run
    # with mlflow.start_run():
    #     mlflow.log_param("learning_rate", 0.01)
    #     mlflow.log_metric("accuracy", 0.95)

    print("Mock pipeline completed!")

if __name__ == "__main__":
    main()
