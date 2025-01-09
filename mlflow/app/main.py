import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from io import BytesIO
import pandas as pd
from minio import Minio

# Minio configuration
MLFLOW_TRACKING_URL = 'http://mlflow-service.default.svc.cluster.local:8002'
MINIO_URL = "http://minio-service.default.svc.cluster.local:8000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
TRAIN_BUCKET = "train"
TRAIN_FILE = "train.parquet"

# Initialize MinIO client
minio_client = Minio(
    MINIO_URL.replace("http://", ""),
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)

def fetch_data_from_minio(bucket_name, file_name):
    """
    Fetch data from MinIO and return as a Pandas DataFrame.
    """
    try:
        # Get object from MinIO
        response = minio_client.get_object(bucket_name, file_name)

        # Load data into Pandas DataFrame
        data = pd.read_parquet(BytesIO(response.read()))
        print(f"Successfully fetched data from {bucket_name}/{file_name}")
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        raise


def main():
    # MLflow setup
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URL)
    mlflow.set_experiment("Customer_Churn_Model")

    # Fetch training data from MinIO
    data = fetch_data_from_minio(TRAIN_BUCKET, TRAIN_FILE)

    # Separate features and target
    X = data.drop(columns=["Churn", "CustomerID"], errors="ignore")
    y = data["Churn"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    knn = KNeighborsClassifier()

    # Use StratifiedKFold for better validation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(knn, param_grid, cv=skf, scoring='accuracy', n_jobs=2)

    with mlflow.start_run() as run:
        # Log parameters being tuned
        mlflow.log_param("param_grid", param_grid)

        # Fit the GridSearchCV
        grid_search.fit(X_train, y_train)

        # Best parameters and model
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        # Log best parameters
        mlflow.log_params(best_params)

        # Evaluate the model on the test set
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", float(report.get("1", {}).get("precision", 0)))
        mlflow.log_metric("recall", float(report.get("1", {}).get("recall", 0)))

        # input_example = np.array([X_train[0]])
        # Log the model
        # mlflow.sklearn.log_model(knn, "knn_model", input_example=input_example)

        print(f"Run ID: {run.info.run_id}")
        print("Model logged successfully!")

    # End phase: Clear summary
    print("\nPipeline Execution Complete")
    print(f"Experiment URL: {MLFLOW_TRACKING_URL}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")


if __name__ == "__main__":
    main()
