from pyspark.sql import SparkSession

def main():
    # Initialize Spark session
    spark = SparkSession.builder.appName("Preprocessor").getOrCreate()
    
    # Print a message to confirm the script is running
    print("Running Preprocessor Spark Job...")
    
    # Stop the Spark session
    spark.stop()

if __name__ == "__main__":
    main()
