from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from minio import Minio, S3Error
import os, traceback

MINIO_URL = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
HTTPS = os.getenv("HTTPS") == "True"
MINIO_BUCKET = "preprocessed"
FILENAME = "preprocessed.parquet"

def main():
    # Initialize Spark session
    spark = SparkSession.builder.appName("Preprocessor").getOrCreate()

    # Log message to confirm the script is running
    print("Running Preprocessor Spark Job...")

    # ## Load schema
    # schema for csv
    # customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Churn
    schema = StructType(
        [
            StructField("customerID", StringType(), nullable=False),
            StructField("gender", StringType()),
            StructField("SeniorCitizen", IntegerType()),
            StructField("Partner", StringType()),
            StructField("Dependents", StringType()),
            StructField("tenure", IntegerType()),
            StructField("PhoneService", StringType()),
            StructField("MultipleLines", StringType()),
            StructField("InternetService", StringType()),
            StructField("OnlineSecurity", StringType()),
            StructField("OnlineBackup", StringType()),
            StructField("DeviceProtection", StringType()),
            StructField("TechSupport", StringType()),
            StructField("StreamingTV", StringType()),
            StructField("StreamingMovies", StringType()),
            StructField("Contract", StringType()),
            StructField("PaperlessBilling", StringType()),
            StructField("PaymentMethod", StringType()),
            StructField("MonthlyCharges", FloatType()),
            StructField("TotalCharges", FloatType()),
            StructField("Churn", StringType()),
        ]
    )

    # TODO: read dataset from minio

    # ## Description
    # - customerID: string
    # - gender: Male/Female
    # - SeniorCitizen: 0/1
    # - Partner: Yes/No
    # - Dependents: Yes/No
    # - tenure: integer
    # - PhoneService: Yes/No
    #     - MultipleLines: Yes/No/No phone service
    # - InternetService: string
    #     - OnlineSecurity: Yes/No/No internet service
    #     - OnlineBackup: Yes/No/No internet service
    #     - DeviceProtection: Yes/No/No internet service
    #     - TechSupport: Yes/No/No internet service
    #     - StreamingTV: Yes/No/No internet service
    #     - StreamingMovies: Yes/No/No internet service
    # - Contract: string
    # - PaperlessBilling: Yes/No
    # - PaymentMethod: string
    # - MonthlyCharges: float
    # - TotalCharges: float
    # - Churn: Yes/No
    #
    # read csv into schema
    df = (
        spark.read.option("header", True)
        .schema(schema)
        .csv("dataset.csv")
    )

    # df.printSchema()
    print("Data row count:")
    df.count()

    # =================
    # ## DATA CLEANING
    # - rename columns
    # - drop duplicates
    # - remove null
    # - check consistency for PhoneService or InternetService related columns
    # - check consistency for SeniorCitizen
    # - fill missing values for PhoneService or InternetService related columns
    # - delete invalid values (NaN or negatives)
    # - reformat for boolean columns

    print("Cleaning data...")
    print("Renaming columns...")
    # ### Rename columns
    df = df.withColumnsRenamed(
        {"customerID": "CustomerID", "gender": "Gender", "tenure": "Tenure"}
    )

    print("Dropping duplicate and null values...")
    # ### Drop duplicates and null values
    # drop duplicates
    df = df.dropDuplicates()

    # drop duplicates by CustomerID
    df = df.dropDuplicates(["CustomerID"])

    # drop na on non PhoneService or InternetService related columns
    df = df.dropna(
        subset=[
            "CustomerID",
            "Gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "Tenure",
            "PhoneService",
            "InternetService",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
            "MonthlyCharges",
            "TotalCharges",
            "Churn",
        ]
    )

    print("Checking data consistency...")
    # ### Check consistency for PhoneService
    # check consistency for PhoneService related columns
    invalid_phone = df.where("PhoneService == 'No' AND MultipleLines <> 'No phone service'")

    # subtract invalid PhoneService from df
    df = df.subtract(invalid_phone)

    # ### Check consistency for InternetService
    # check consistency for InternetService related columns
    invalid_internet = df.where(
        """InternetService = 'No' AND (
                            OnlineSecurity <> 'No internet service' OR
                            OnlineBackup <> 'No internet service' OR
                            DeviceProtection <> 'No internet service' OR
                            TechSupport <> 'No internet service' OR
                            StreamingTV <> 'No internet service' OR
                            StreamingMovies <> 'No internet service')"""
    )

    # subtract invalid InternetService from df
    df = df.subtract(invalid_internet)

    # ### Check consistency for SeniorCitizen
    invalid_senior = df.where("SeniorCitizen <> 0 AND SeniorCitizen <> 1")

    # subtract
    df = df.subtract(invalid_senior)

    print("Filling missing values...")
    # ### Fill missing values (PhoneService and InternetService)
    #
    # - if PhoneService or InternetService == 'Yes' and respective column is missing, drop
    # - if either == 'No', fill missing column with 'No phone service' or 'No internet service' respectively
    #

    # if PhoneService or InternetService == 'Yes' and respective column is missing, drop
    missing_phone = df.where("PhoneService == 'Yes' AND MultipleLines IS NULL")

    # subtract missing "Yes" on phones
    df = df.subtract(missing_phone)

    # if PhoneService or InternetService == 'Yes' and respective column is missing, drop
    missing_internet = df.where(
        """InternetService == 'Yes' AND (
        OnlineSecurity IS NULL OR
        OnlineBackup IS NULL OR
        DeviceProtection IS NULL OR
        TechSupport IS NULL OR
        StreamingTV IS NULL OR
        StreamingMovies IS NULL)"""
    )
    # subtract missing "Yes" on internet
    df = df.subtract(missing_internet)

    # if == 'No', fill missing column with 'No phone service' or 'No internet service' respectively

    # handle PhoneService
    df = df.withColumn(
        "MultipleLines",
        when(col("PhoneService") == "No", "No phone service").otherwise(
            col("MultipleLines")
        ),
    )

    # handle InternetService
    df = df.withColumns(
        {
            "OnlineSecurity": when(
                col("InternetService") == "No", "No internet service"
            ).otherwise(col("OnlineSecurity")),
            "OnlineBackup": when(
                col("InternetService") == "No", "No internet service"
            ).otherwise(col("OnlineBackup")),
            "DeviceProtection": when(
                col("InternetService") == "No", "No internet service"
            ).otherwise(col("DeviceProtection")),
            "TechSupport": when(
                col("InternetService") == "No", "No internet service"
            ).otherwise(col("TechSupport")),
            "StreamingTV": when(
                col("InternetService") == "No", "No internet service"
            ).otherwise(col("StreamingTV")),
            "StreamingMovies": when(
                col("InternetService") == "No", "No internet service"
            ).otherwise(col("StreamingMovies")),
        }
    )

    print("Removing invalid values...")
    # ### Delete invalid values
    invalid_float = df.where(
        "Tenure < 0 OR MonthlyCharges < 0 OR TotalCharges < 0 OR MonthlyCharges == 'NaN' OR TotalCharges == 'NaN'"
    )
    df = df.subtract(invalid_float)

    print("Reformat...")
    # ### Reformat values
    df = df.withColumns(
        {
            "SeniorCitizen": when(col("SeniorCitizen") == 1, True).otherwise(False),
            "Partner": when(col("Partner") == "Yes", True).otherwise(False),
            "Dependents": when(col("Dependents") == "Yes", True).otherwise(False),
            "PhoneService": when(col("PhoneService") == "Yes", True).otherwise(False),
            "PaperlessBilling": when(col("PaperlessBilling") == "Yes", True).otherwise(
                False
            ),
            "Churn": when(col("Churn") == "Yes", True).otherwise(False),
        }
    )
    print("Data cleaned.")
    df.count()

    # ===============

    # Export
    print("Writing Parquet File...")
    df.write.mode("overwrite").parquet(FILENAME)

    # Log message to confirm the job is complete
    print("Preprocessor Spark Job completed successfully.")

    # Stop the Spark session
    spark.stop()
    print("Spark Session Stopped.")

    # Upload file to Minio
    print("Starting Upload to MinIO...")
    minio_client = Minio(
        MINIO_URL, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=HTTPS
    )

    # Check Minio bucket
    print("Check MinIO Bucket Availability...")
    found = minio_client.bucket_exists(MINIO_BUCKET)
    if not found:
        minio_client.make_bucket(MINIO_BUCKET)
        print("Created bucket", MINIO_BUCKET)
    else:
        print("Bucket", MINIO_BUCKET, "already exists")

    # TODO: fix upload parquet folder to minio
    minio_client.fput_object(
        MINIO_BUCKET,
        FILENAME,
        FILENAME,
    )
    print(
        FILENAME,
        "successfully uploaded as object",
        FILENAME,
        "to bucket",
        MINIO_BUCKET,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(traceback.format_exc())
