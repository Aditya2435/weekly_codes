from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *

spark = SparkSession.builder.appName("pyspark_movie_analysis").getOrCreate()

def create_watch_history_df(spark, path):
    schema = StructType([
        StructField("user_id", StringType(), True),
        StructField("show_id", StringType(), True),
        StructField("genre", StringType(), True),
        StructField("watch_duration", IntegerType(), True),
        StructField("watch_timestamp", StringType(), True)
    ])
    df = spark.read.csv(path, header=True, schema=schema)
    df = df.withColumn("watch_date", to_date("watch_timestamp"))
    return df

def create_user_df(spark,path):
    schema = StructType(
        [StructField("user_id",StringType(),True),
         StructField("name",StringType(),True),
         StructField("age",IntegerType(),True),
         StructField("subscription_type",StringType(),True)]
    )
    df = spark.read.csv(path, schema=schema, header=True)
    return df

def join_user_watch_df(watch_df, user_df):
    return watch_df.join(user_df, on="user_id", how="inner")

def compute_avg_watch_duration(watch_df):
    watched = watch_df.groupBy("user_id").agg(avg("watch_duration").alias("avg_watch_duration"))   
    return watched.select("user_id","avg_watch_duration")

def categorize_watchers(watch_df):
    categorized_df = watch_df.withColumn(
        "watch_category",
        when(col("watch_duration") < 30, "Light")
        .when((col("watch_duration") >= 30) & (col("watch_duration") <= 60), "Moderate")
        .otherwise("Heavy")
    )
    return categorized_df

def get_unique_genres(watch_df):
    u_genre = watch_df.select("genre").distinct()
    return list(u_genre)

def filter_frequent_users(watch_df,min_watches):
    grouped = watch_df.groupBy("user_id").agg(count("*").alias("watch_count"))
    filtered = grouped.filter(col("watch_count")>=min_watches)
    return filtered

def most_watched_genre(watch_df):
    genre_counts = watch_df.groupBy("genre").agg(count("*"))
    top_genre_row = genre_counts.orderBy(col('genre').desc()).limit(1).collect()
    return top_genre_row[0]["genre"]
