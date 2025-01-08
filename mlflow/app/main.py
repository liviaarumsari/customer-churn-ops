import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
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
        data = pd.read_csv(BytesIO(response.data))
        print(f"Successfully fetched data from {bucket_name}/{file_name}")
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        raise


TRAIN_FILE = "train_data.csv"


def main():
    # MLflow setup
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URL)
    mlflow.set_experiment("Customer_Churn_Model")

    # Fetch training data from MinIO
    data = fetch_data_from_minio(TRAIN_BUCKET, TRAIN_FILE)

    # Preprocess the data
    X = data.drop("Churn", axis=1)
    y = LabelEncoder().fit_transform(data["Churn"])

    # Separate customerID
    X = X.drop(columns=["customerID"], errors="ignore")

    # Detect and encode all non-numeric columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        X[col] = le.fit_transform(X[col].astype(str))

    # Scale the features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train the model
    knn = KNeighborsClassifier(n_neighbors=5)

    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_param("model_type", "KNN")
        mlflow.log_param("n_neighbors", 5)

        # Fit the model
        knn.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", float(report.get("1", {}).get("precision")))
        mlflow.log_metric("recall", float(report.get("1", {}).get("recall")))

        # input_example = np.array([X_train[0]])
        # Log the model
        # mlflow.sklearn.log_model(knn, "knn_model", input_example=input_example)

        predictions = pd.DataFrame({
            "prediction": y_pred
        })

        print(predictions.head())
        print(f"Run ID: {run.info.run_id}")
        print("Model logged successfully!")

    # End phase: Clear summary
    print("\nPipeline Execution Complete")
    print(f"Experiment URL: {MLFLOW_TRACKING_URL}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")


if __name__ == "__main__":
    main()
