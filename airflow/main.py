from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
from io import BytesIO
from minio import Minio
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
            if file_info.size > 288000:  # TODO: Update threshold
                logging.info(f"File {obj.object_name} exceeds threshold!")
                return obj.object_name
        logging.info("No files exceeded the threshold.")
    except Exception as e:
        logging.error(f"Error in check_minio_files: {str(e)}")
        raise
    return None


# Decide next step based on check_minio_files() return value
def decide_next_step(**context):
    file_name = context['task_instance'].xcom_pull(task_ids='check_minio_files')
    if file_name:
        return 'trigger_spark_preprocessor'
    return 'no_large_files'


# Trigger the Spark job as a Kubernetes Job
def trigger_spark_job():
    try:
        # Load in-cluster Kubernetes configuration
        config.load_incluster_config()

        # Define the Kubernetes Job specification
        job = client.V1Job(
            metadata=client.V1ObjectMeta(name="spark-preprocessor"),
            spec=client.V1JobSpec(
                template=client.V1PodTemplateSpec(
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name="spark-preprocessor",
                                image="spark-preprocessor:latest",
                                image_pull_policy="Never",
                                env=[
                                    client.V1EnvVar(name="MINIO_ENDPOINT", value=MINIO_URL),
                                    client.V1EnvVar(name="MINIO_ACCESS_KEY", value=MINIO_ACCESS_KEY),
                                    client.V1EnvVar(name="MINIO_SECRET_KEY", value=MINIO_SECRET_KEY),
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
        batch_v1.delete_namespaced_job(name="spark-preprocessor", namespace="default", body={})
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
PREPROCESSED_FILE = "preprocessed.parquet"
TRAIN_FILE = "train.parquet"
MONITORED_FEATURES = ["MonthlyCharges", "TotalCharges"]
PSI_THRESHOLD = 0.2


def calculate_psi_score(**context):
    try:
        # Fetching files from MinIO
        preprocessed_object = minio_client.get_object(PREPROCESSED_BUCKET, PREPROCESSED_FILE)
        train_object = minio_client.get_object(TRAIN_BUCKET, TRAIN_FILE)

        # Read files into pandas dataframes
        preprocessed_df = pd.read_parquet(BytesIO(preprocessed_object.read()))
        train_df = pd.read_parquet(BytesIO(train_object.read()))

        # PSI Calculation Logic
        def calculate_psi(expected_df, actual_df, columns, bucket=10):
            """
            Calculate the total PSI for a dataset by summing PSI values for individual features.
            """
            total_psi = 0
            psi_details = {}

            for col in columns:
                expected = expected_df[col]
                actual = actual_df[col]

                # Calculate percentage distributions for observed and expected
                observed_perc = np.histogram(actual, bins=bucket, range=(expected.min(), expected.max()), density=True)[
                    0]
                expected_perc = \
                    np.histogram(expected, bins=bucket, range=(expected.min(), expected.max()), density=True)[0]

                # Avoid division by zero
                expected_perc = np.where(expected_perc == 0, 0.0000000001, expected_perc)
                observed_perc = np.where(observed_perc == 0, 0.0000000001, observed_perc)

                # Calculate PSI for this feature
                psi = np.sum((observed_perc - expected_perc) * np.log(observed_perc / expected_perc))
                psi_details[col] = psi
                total_psi += psi

            return total_psi, psi_details

        psi_score, psi_per_features = calculate_psi(train_df, preprocessed_df, MONITORED_FEATURES)

        context['task_instance'].xcom_push(key='psi_score', value=psi_score)
        logging.info(f"PSI Score Calculated: {psi_score}")
        logging.info(f"Detailed PSI: {psi_per_features}")

    except Exception as e:
        logging.error(f"Error in calculating PSI: {str(e)}")
        raise e


# Decide next step based on total PSI
def decide_based_on_total_psi(**context):
    total_psi = context['task_instance'].xcom_pull(task_ids='calculate_psi', key='psi_score')
    if total_psi > PSI_THRESHOLD:
        logging.info("PSI exceeds threshold. Replacing training data with preprocessed data.")

        # Delete old training data
        if minio_client.stat_object(TRAIN_BUCKET, TRAIN_FILE):
            minio_client.remove_object(TRAIN_BUCKET, TRAIN_FILE)
            logging.info(f"Deleted old training data: {TRAIN_FILE}")

        # Copy preprocessed data to training data bucket
        preprocessed_data = minio_client.get_object(PREPROCESSED_BUCKET, PREPROCESSED_FILE)
        data = preprocessed_data.data
        data_length = len(data)
        minio_client.put_object(TRAIN_BUCKET, TRAIN_FILE, BytesIO(data), length=data_length)
        logging.info(f"Replaced training data with preprocessed data: {TRAIN_FILE}")

        return 'trigger_retraining_job'
    return 'skip_retraining'


# Default args for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}
with DAG(
        dag_id="minio_to_spark",
        default_args=default_args,
        description="DAG to watch MinIO and trigger Spark preprocessing",
        schedule_interval=timedelta(minutes=3),
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
        task_id='decide_next_step',
        provide_context=True,
        python_callable=decide_next_step,
    )

    # Task 3a: Trigger Spark Job
    # spark_preprocessor_task = PythonOperator(
    #     task_id="trigger_spark_preprocessor",
    #     python_callable=trigger_spark_job,
    # )

    # Task 3b: No large files found
    no_large_files = DummyOperator(task_id='no_large_files')

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
    # check_files_task >> decide_step
    # decide_step >> [spark_preprocessor_task, no_large_files]
    # spark_preprocessor_task >> calculate_psi_task
    calculate_psi_task >> decide_retraining_task
    decide_retraining_task >> [retraining_job, skip_retraining_task]
