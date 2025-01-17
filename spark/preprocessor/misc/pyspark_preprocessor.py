# %% [markdown]
# # Pyspark Preprocessor

# %%
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *

# new session
spark = SparkSession.builder.getOrCreate()

# %% [markdown]
# ## Load schema and data

# %%
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

# %%
# read csv into schema

df = (
    spark.read.option("header", True)
    .schema(schema)
    .csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
)

# df.show()

# %%
df.printSchema()

df.count()

# %% [markdown]
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

# %% [markdown]
# ## Cleaning
# - check for typos
# - rename columns
# - drop duplicates
# - remove null
# - check consistency for PhoneService or InternetService related columns
# - check consistency for SeniorCitizen
# - fill missing values for PhoneService or InternetService related columns
# - delete invalid values (NaN or negatives)
# - one-hot encoding for string valued columns
# - reformat for boolean columns

# %% [markdown]
# ### Rename columns

# %%
df = df.withColumnsRenamed(
    {"customerID": "CustomerID", "gender": "Gender", "tenure": "Tenure"}
)

df.show()

# %% [markdown]
# ### Drop duplicates and null values

# %%
# drop duplicates
df = df.dropDuplicates()

# drop duplicates by CustomerID
df = df.dropDuplicates(["CustomerID"])

# df.count()

# %%
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
# df.count()

# %% [markdown]
# ### Check consistency for PhoneService

# %%
# check consistency for PhoneService related columns
invalid_phone = df.where("PhoneService == 'No' AND MultipleLines <> 'No phone service'")

# subtract invalid PhoneService from df
df = df.subtract(invalid_phone)
# df.count()

# %% [markdown]
# ### Check consistency for InternetService

# %%
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
# df.count()

# %% [markdown]
# ### Check consistency for SeniorCitizen

# %%
invalid_senior = df.where("SeniorCitizen <> 0 AND SeniorCitizen <> 1")

# subtract
df = df.subtract(invalid_senior)
# df.count()

# %% [markdown]
# ### Fill missing values (PhoneService and InternetService)
#
# - if PhoneService or InternetService == 'Yes' and respective column is missing, drop
# - if either == 'No', fill missing column with 'No phone service' or 'No internet service' respectively
#

# %%
# if PhoneService or InternetService == 'Yes' and respective column is missing, drop
missing_phone = df.where("PhoneService == 'Yes' AND MultipleLines IS NULL")

# subtract missing "Yes" on phones
df = df.subtract(missing_phone)
# df.count()

# %%
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
# df.count()

# %%
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

# %% [markdown]
# ### Delete invalid values

# %%
invalid_float = df.where(
    "Tenure < 0 OR MonthlyCharges < 0 OR TotalCharges < 0 OR MonthlyCharges == 'NaN' OR TotalCharges == 'NaN'"
)
df = df.subtract(invalid_float)
# df.count()

# %% [markdown]
# ### One-hot encoding

# %%
from pyspark.ml.feature import OneHotEncoder, StringIndexer

# setup string indexer for df
si = StringIndexer(
    inputCols=[
        "Gender",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
    ],
    outputCols=[
        "GenderIndexed",
        "MultipleLinesIndexed",
        "InternetServiceIndexed",
        "OnlineSecurityIndexed",
        "OnlineBackupIndexed",
        "DeviceProtectionIndexed",
        "TechSupportIndexed",
        "StreamingTVIndexed",
        "StreamingMoviesIndexed",
        "ContractIndexed",
        "PaymentMethodIndexed",
    ],
)
si_model = si.fit(df)

# indexing categoricals on df
indexed_df = si_model.transform(df)
indexed_df.show()


# %%
# setup onehotencoder
ohe = OneHotEncoder(
    inputCols=[
        "GenderIndexed",
        "MultipleLinesIndexed",
        "InternetServiceIndexed",
        "OnlineSecurityIndexed",
        "OnlineBackupIndexed",
        "DeviceProtectionIndexed",
        "TechSupportIndexed",
        "StreamingTVIndexed",
        "StreamingMoviesIndexed",
        "ContractIndexed",
        "PaymentMethodIndexed",
    ],
    outputCols=[
        "GenderVector",
        "MultipleLinesVector",
        "InternetServiceVector",
        "OnlineSecurityVector",
        "OnlineBackupVector",
        "DeviceProtectionVector",
        "TechSupportVector",
        "StreamingTVVector",
        "StreamingMoviesVector",
        "ContractVector",
        "PaymentMethodVector",
    ],
)
ohe_model = ohe.fit(indexed_df)

# encode
encoded_df = ohe_model.transform(indexed_df)
encoded_df.show()

df = encoded_df


# %% [markdown]
# ### Reformat values

# %%
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
df.show()

# %% [markdown]
# ## Export

# %%
# export parquet
df.write.mode("overwrite").parquet("preprocessed.parquet")
