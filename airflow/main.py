from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
from io import BytesIO
from minio import Minio
from minio.error import S3Error
from kubernetes import client, config
import logging
import time
import numpy as np
import pandas as pd

# MinIO client setup
MINIO_URL = "http://minio-service.default.svc.cluster.local:8000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
RAW_BUCKET = "raw"
PREPROCESSED_BUCKET = "preprocessed"
TRAIN_BUCKET = "train"

# Initialize MinIO client
minio_client = Minio(
    MINIO_URL.replace("http://", ""),
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)


# Check file size in MinIO
def check_minio_files():
    try:
        logging.info("Connecting to Minio.")
        objects = minio_client.list_objects(RAW_BUCKET, recursive=True)
        logging.info("Connected to Minio and listing objects.")
        for obj in objects:
            file_info = minio_client.stat_object(RAW_BUCKET, obj.object_name)
            logging.info(f"Found file: {obj.object_name}, Size: {file_info.size} bytes")
            if file_info.size > 1:
                logging.info(f"File {obj.object_name} exceeds threshold!")
                return obj.object_name
        logging.info("No files exceeded the threshold.")
    except Exception as e:
        logging.error(f"Error in check_minio_files: {str(e)}")
        raise
    return None


# Decide next step based on check_minio_files() return value
def decide_next_step(**context):
    file_name = context["task_instance"].xcom_pull(task_ids="check_minio_files")
    if file_name:
        return "trigger_spark_preprocessor"
    return "no_large_files"


# Trigger the Spark job as a Kubernetes Job
def trigger_spark_job():
    try:
        # Load in-cluster Kubernetes configuration
        config.load_incluster_config()

        # Define the Kubernetes Job specification
        job = client.V1Job(
            metadata=client.V1ObjectMeta(name="spark-preprocessor"),
            spec=client.V1JobSpec(
                backoff_limit=1,
                template=client.V1PodTemplateSpec(
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name="spark-preprocessor",
                                image="spark-preprocessor:latest",
                                image_pull_policy="Never",
                                resources=client.V1ResourceRequirements(
                                    limits={"memory": "4Gi", "cpu": "2"},
                                    requests={"memory": "2Gi", "cpu": "1"},
                                ),
                                env=[
                                    client.V1EnvVar(
                                        name="MINIO_ENDPOINT", value=MINIO_URL
                                    ),
                                    client.V1EnvVar(
                                        name="MINIO_ACCESS_KEY", value=MINIO_ACCESS_KEY
                                    ),
                                    client.V1EnvVar(
                                        name="MINIO_SECRET_KEY", value=MINIO_SECRET_KEY
                                    ),
                                    client.V1EnvVar(name="HTTPS", value="False"),
                                ],
                            )
                        ],
                        restart_policy="Never",
                    )
                ),
            ),
        )

        # Create the Job in the "default" namespace
        batch_v1 = client.BatchV1Api()

        # Check if the job already exists
        try:
            existing_job = batch_v1.read_namespaced_job(
                name="spark-preprocessor", namespace="default"
            )
            if existing_job:
                logging.info("Job already exists. Deleting existing job...")
                batch_v1.delete_namespaced_job(
                    name="spark-preprocessor",
                    namespace="default",
                    body=client.V1DeleteOptions(propagation_policy="Foreground"),
                )
                # Wait for the job to be deleted
                while True:
                    try:
                        batch_v1.read_namespaced_job(
                            name="spark-preprocessor", namespace="default"
                        )
                        time.sleep(1)  # Wait for deletion to complete
                    except client.exceptions.ApiException as e:
                        if e.status == 404:  # Not Found
                            break
                        else:
                            raise

        except client.exceptions.ApiException as e:
            if e.status == 404:  # Not Found
                logging.info("No existing job found. Proceeding to create a new job.")
            else:
                raise


        batch_v1.create_namespaced_job(namespace="default", body=job)
        logging.info("Spark Preprocessor job triggered successfully.")

        # Poll the job status
        while True:
            job_status = batch_v1.read_namespaced_job_status(
                name="spark-preprocessor", namespace="default"
            )
            if job_status.status.succeeded:
                print("Job completed successfully")
                break
            elif job_status.status.failed:
                raise Exception("Job failed")
            time.sleep(10)
        logging.info("Spark Preprocessor job finished.")
    except Exception as e:
        logging.error(f"Failed to trigger Spark Preprocessor job: {str(e)}")
        batch_v1.delete_namespaced_job(
            name="spark-preprocessor", namespace="default", body={}
        )
        raise


# Trigger the MLFlow Retraining job as a Kubernetes Job
def trigger_retraining_job():
    try:
        # Load in-cluster Kubernetes configuration
        config.load_incluster_config()

        # Define the Kubernetes Job specification
        job = client.V1Job(
            metadata=client.V1ObjectMeta(name="mlflow-job"),
            spec=client.V1JobSpec(
                template=client.V1PodTemplateSpec(
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name="mlflow-job",
                                image="mlflow-job:latest",
                                image_pull_policy="Never",
                            )
                        ],
                        restart_policy="Never",
                    )
                ),
            ),
        )

        # Create the Job in the "default" namespace
        batch_v1 = client.BatchV1Api()

        # Check if the job already exists
        try:
            existing_job = batch_v1.read_namespaced_job(
                name="mlflow-job", namespace="default"
            )
            if existing_job:
                logging.info("Job already exists. Deleting existing job...")
                batch_v1.delete_namespaced_job(
                    name="mlflow-job",
                    namespace="default",
                    body=client.V1DeleteOptions(propagation_policy="Foreground"),
                )
                # Wait for the job to be deleted
                while True:
                    try:
                        batch_v1.read_namespaced_job(
                            name="mlflow-job", namespace="default"
                        )
                        time.sleep(1)  # Wait for deletion to complete
                    except client.exceptions.ApiException as e:
                        if e.status == 404:  # Not Found
                            break
                        else:
                            raise

        except client.exceptions.ApiException as e:
            if e.status == 404:  # Not Found
                logging.info("No existing job found. Proceeding to create a new job.")
            else:
                raise

        batch_v1.create_namespaced_job(namespace="default", body=job)
        logging.info("MLFlow Retraining job triggered successfully.")

        # Poll the job status
        while True:
            job_status = batch_v1.read_namespaced_job_status(
                name="mlflow-job", namespace="default"
            )
            if job_status.status.succeeded:
                print("Job completed successfully")
                break
            elif job_status.status.failed:
                raise Exception("Job failed")
            time.sleep(10)
        logging.info("MLFlow Retraining job finished.")
    except Exception as e:
        logging.error(f"Failed to trigger MLFlow Retraining job: {str(e)}")
        batch_v1.delete_namespaced_job(name="mlflow-job", namespace="default", body={})
        raise


# CONSTANT
PREPROCESSED_FILE_PREFIX = "preprocessed.parquet/"
TRAIN_FILE = "train.parquet"
NUMERICAL_COLUMNS = ["Tenure", "MonthlyCharges", "TotalCharges"]
PSI_THRESHOLD = 0.2


def calculate_psi_score(**context):
    try:
        # Fetching files from MinIO
        objects = list(
            minio_client.list_objects(
                PREPROCESSED_BUCKET, prefix=PREPROCESSED_FILE_PREFIX, recursive=True
            )
        )

        if not objects:
            raise Exception("No objects found in MinIO under the specified prefix.")

        for obj in objects:
            print(f"Found object: {obj.object_name}")

        first_file = objects[1].object_name
        preprocessed_object = minio_client.get_object(PREPROCESSED_BUCKET, first_file)

        # Read files into pandas dataframes
        preprocessed_df = pd.read_parquet(BytesIO(preprocessed_object.read()))

        # Check if training data exists in the bucket
        try:
            train_object = minio_client.get_object(TRAIN_BUCKET, TRAIN_FILE)
            train_df = pd.read_parquet(BytesIO(train_object.read()))
        except S3Error as e:
            logging.warning(f"Training data not found: {TRAIN_FILE}. Triggering retraining.")
            context['task_instance'].xcom_push(key='psi_score', value=1.0)
            return

        # Exclude CustomerID and Churn columns
        features = [col for col in train_df.columns if col not in ['CustomerID', 'Churn']]

        def calculate_psi(expected_series, actual_series, bucket=10, is_numerical=False):
            """
            Calculate PSI for a single feature.
            """
            if not is_numerical:
                # Handle categorical features
                expected_perc = expected_series.value_counts(normalize=True)
                actual_perc = actual_series.value_counts(normalize=True)

                # Align on the same categories
                all_categories = set(expected_perc.index).union(set(actual_perc.index))
                expected_perc = expected_perc.reindex(all_categories, fill_value=0)
                actual_perc = actual_perc.reindex(all_categories, fill_value=0)
            else:
                # Handle numerical features by binning
                bins = np.linspace(expected_series.min(), expected_series.max(), bucket + 1)
                expected_perc = np.histogram(expected_series, bins=bins, density=True)[0]
                actual_perc = np.histogram(actual_series, bins=bins, density=True)[0]

            # Avoid division by zero
            expected_perc = np.where(expected_perc == 0, 0.0000000001, expected_perc)
            actual_perc = np.where(actual_perc == 0, 0.0000000001, actual_perc)

            # Calculate PSI
            psi = np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))
            return psi

        # Calculate PSI for each feature
        psi_details = {}
        for feature in features:
            is_numerical = feature in NUMERICAL_COLUMNS
            psi = calculate_psi(train_df[feature], preprocessed_df[feature], is_numerical)
            psi_details[feature] = psi

        # Total PSI
        total_psi = sum(psi_details.values())

        # Push results to XCom
        context['task_instance'].xcom_push(key='psi_score', value=total_psi)
        logging.info(f"Total PSI Score Calculated: {total_psi}")
        logging.info(f"Detailed PSI per feature: {psi_details}")

    except Exception as e:
        logging.error(f"Error in calculating PSI: {str(e)}")
        raise e


# Decide next step based on total PSI
def decide_based_on_total_psi(**context):
    total_psi = context["task_instance"].xcom_pull(
        task_ids="calculate_psi", key="psi_score"
    )
    if total_psi > PSI_THRESHOLD:
        logging.info(
            "PSI exceeds threshold. Replacing training data with preprocessed data."
        )

        # Check and delete old training data if it exists
        try:
            minio_client.stat_object(TRAIN_BUCKET, TRAIN_FILE)
            minio_client.remove_object(TRAIN_BUCKET, TRAIN_FILE)
            logging.info(f"Deleted old training data: {TRAIN_FILE}")
        except S3Error as e:
            if e.code == "NoSuchKey":
                logging.warning(f"Training file {TRAIN_FILE} does not exist in bucket {TRAIN_BUCKET}. Proceeding...")
            else:
                raise

        # Copy preprocessed data to training data bucket
        # Fetching files from MinIO
        objects = list(minio_client.list_objects(
            PREPROCESSED_BUCKET, prefix=PREPROCESSED_FILE_PREFIX, recursive=True
        ))

        for obj in objects:
            print(f"Found object: {obj.object_name}")

        first_file = objects[1].object_name
        preprocessed_data = minio_client.get_object(PREPROCESSED_BUCKET, first_file)

        data = preprocessed_data.data
        data_length = len(data)
        minio_client.put_object(
            TRAIN_BUCKET, TRAIN_FILE, BytesIO(data), length=data_length
        )
        logging.info(f"Replaced training data with preprocessed data: {TRAIN_FILE}")

        return "trigger_retraining_job"
    return "skip_retraining"


# Default args for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=1),
}
with DAG(
    dag_id="cust_churn_workflow",
    default_args=default_args,
    description="DAG to watch MinIO and trigger Spark preprocessing",
    schedule_interval="0 * * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:
    logging.info("Starting the Task.")

    # Task 1: Check MinIO for files
    check_files_task = PythonOperator(
        task_id="check_minio_files",
        python_callable=check_minio_files,
    )

    # Task 2: Decide next step
    decide_step = BranchPythonOperator(
        task_id="decide_next_step",
        provide_context=True,
        python_callable=decide_next_step,
    )

    # Task 3a: Trigger Spark Job
    spark_preprocessor_task = PythonOperator(
        task_id="trigger_spark_preprocessor",
        python_callable=trigger_spark_job,
    )

    # Task 3b: No large files found
    no_large_files = DummyOperator(task_id="no_large_files")

    # Task 4 : Calculate PSI
    calculate_psi_task = PythonOperator(
        task_id="calculate_psi",
        python_callable=calculate_psi_score,
        provide_context=True,
    )

    # Task 5: Decide Retraining Based on PSI
    decide_retraining_task = BranchPythonOperator(
        task_id="decide_based_on_total_psi",
        python_callable=decide_based_on_total_psi,
        provide_context=True,
    )

    # Task 6a: Trigger Retraining Job
    retraining_job = PythonOperator(
        task_id="trigger_retraining_job",
        python_callable=trigger_retraining_job,
    )

    # Task 6b: Skip retraining
    skip_retraining_task = DummyOperator(
        task_id="skip_retraining",
    )

    # Task dependencies graph
    check_files_task >> decide_step
    decide_step >> [spark_preprocessor_task, no_large_files]
    spark_preprocessor_task >> calculate_psi_task
    calculate_psi_task >> decide_retraining_task
    decide_retraining_task >> [retraining_job, skip_retraining_task]
