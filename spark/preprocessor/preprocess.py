from pyspark.sql import SparkSession

def main():
    # Initialize Spark session
    spark = SparkSession.builder.appName("Preprocessor").getOrCreate()
    
    # Log message to confirm the script is running
    print("Running Preprocessor Spark Job...")
    
    # Simple DataFrame operation for testing
    data = [("Alice", 34), ("Bob", 45), ("Cathy", 29)]
    columns = ["Name", "Age"]
    df = spark.createDataFrame(data, columns)
    df.show()  # Display the DataFrame in logs
    
    # Log message to confirm the job is complete
    print("Preprocessor Spark Job completed successfully.")
    
    # Stop the Spark session
    spark.stop()

if __name__ == "__main__":
    main()
