from pyspark.sql import SparkSession

def main():
    # Initialize Spark session
    spark = SparkSession.builder.appName("PSI Worker").getOrCreate()
    
    # Print a message to confirm the script is running
    print("Running PSI Calculation Spark Job...")
    
    # Stop the Spark session
    spark.stop()

if __name__ == "__main__":
    main()
