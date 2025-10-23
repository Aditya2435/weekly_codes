from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
import builtins

def load_data(spark: SparkSession, trips_path: str, drivers_path: str) -> (DataFrame, DataFrame):
    #define schema here for both dataframes 
    
    # trips_schema = StructType([
    #     StructField("trip_id", IntegerType(), True),
    #     StructField("driver_id", IntegerType(), True),
    #     StructField("distance_km", DoubleType(), True),
    #     StructField("fare_amount", DoubleType(), True),
    #     StructField("timestamp", LongType(), True)
    # ])

    # t_schema = (["trip_id","driver_id","distance_km",""])
    
    # drivers_schema = StructType([
    #     StructField("driver_id", IntegerType(), True),
    #     StructField("driver_name", StringType(), True),
    #     StructField("city", StringType(), True)
    # ])

    trips_df = spark.read.csv(trips_path,header=True, inferSchema=True)
    drivers_df = spark.read.csv(drivers_path, header=True, inferSchema=True)
    return trips_df, drivers_df

def clean_trips(trips_df: DataFrame) -> DataFrame:
    return trips_df.dropDuplicates().dropna()

def convert_unix_to_date(trips_df: DataFrame) -> DataFrame:
    trips_df = trips_df.withColumn("trip_datetime", from_unixtime("timestamp"))
    return trips_df

def calculate_avg_fare_per_km(trips_df: DataFrame) -> DataFrame:
    return trips_df.withColumn("fare_per_km",when(col("distance_km") != 0, round(col("fare_amount") / col("distance_km"), 2)).otherwise(lit(None)))

def join_with_driver(trips_df: DataFrame, drivers_df: DataFrame) -> DataFrame:
    return trips_df.join(drivers_df, on="driver_id", how="inner")

def top_n_earning_drivers(joined_df: DataFrame, n: int) -> DataFrame:
    return joined_df.groupBy("driver_id", "driver_name").agg(sum("fare_amount").alias("total_earning")).orderBy(desc("total_earning")).limit(n)
