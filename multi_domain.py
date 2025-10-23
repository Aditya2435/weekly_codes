from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.window import *
from pyspark.sql.types import *

# -------------------------------
# 1. Feedback Processing
# -------------------------------

def load_feedback_from_csv(spark: SparkSession, path: str) -> DataFrame:
    return spark.read.csv("file:///home/user/Downloads/pyspark-assessment-main/pyspark-multi-domain-assessment/data/feedback.csv", header=True, inferSchema=True)

def load_catalog_from_json(spark: SparkSession, path: str) -> DataFrame:
    # Load product catalog from multiline JSON file
    return spark.read.json("file:///home/user/Downloads/pyspark-assessment-main/pyspark-multi-domain-assessment/data/catalog.json")

def calculate_avg_feedback_score(feedback_df: DataFrame) -> DataFrame:
    # Group by product_id and compute average score
    result_df = (feedback_df.withColumn("score",col("score").cast("int")).groupBy("product_id").agg(avg("score").alias("avg_feedback_score")))
    return result_df
def top_products_by_feedback_score(feedback_df: DataFrame, top_n: int) -> DataFrame:
    # Compute average score and return top N products sorted by score descending
    result_df = feedback_df.withColumn("score", col("score").cast("double")).groupBy("product_id").agg(avg("score").alias("avg_score")).orderBy(desc("avg_score")).limit(top_n)
    return result_df

def convert_feedback_date_to_unix(feedback_df: DataFrame) -> DataFrame:
    # Convert feedback_date column to UNIX timestamp
    return feedback_df.withColumn("feedback_date",unix_timestamp(col("feedback_date")))

def flag_top_rated_items(feedback_df: DataFrame) -> DataFrame:
    # Add boolean flag column where score >= 4.5
    return feedback_df.withColumn("flag",when((col("score")>=4.5),True).otherwise(False))

def concat_feedback_summary(feedback_df: DataFrame) -> DataFrame:
    # Create a new summary column combining product_id and comment
    return feedback_df.withColumn("summary",concat_ws("product_id","comment"))

# -------------------------------
# 2. Refund Claims Processing
# -------------------------------

def load_refund_claims(spark: SparkSession, path: str) -> DataFrame:
    # Load refund claims data from CSV
    return spark.read.csv("file:///home/user/Downloads/pyspark-assessment-main/pyspark-multi-domain-assessment/data/refund_claims.csv",header=True, inferSchema=True)

def fill_missing_settlement_dates(claims_df: DataFrame) -> DataFrame:
    # Fill null settlement_date with default future date (e.g., 2099-12-31)
    return claims_df.fillna({"settlement_date":"2099-12-31"})

def region_with_highest_refunds(claims_df: DataFrame) -> DataFrame:
    # Aggregate total refund by region and return the region with the highest total
    return claims_df.groupBy("region").agg(sum("refund_amount").alias("highest_refund_amount")).orderBy(desc("highest_refund_amount")).limit(1)

def highest_refund_customer(claims_df: DataFrame) -> DataFrame:
    # Group by customer and return the one with the highest refund amount
    return claims_df.groupBy("customer_id").agg(count("refund_amount").alias("highest_refund_amount")).orderBy(desc("highest_refund_amount")).limit(1)


def filter_refunds_by_approval_range(claims_df: DataFrame, start: str, end: str) -> DataFrame:
    # Filter claims with approval_date between start and end dates
    return claims_df.withColumn("approval",datediff(col("approval_date"),col("settlement_date")))

def refund_summary_by_region(claims_df: DataFrame) -> DataFrame:
    # Group by region to count claims and sum refund amounts
    return claims_df.groupBy("region").agg(count("claim_id"),sum("refund_amount"))

def refund_days_difference(claims_df: DataFrame) -> DataFrame:
    # Calculate days between approval_date and settlement_date
    return claims_df.withColumn("days",datediff(col("approval_date"), col("settlement_date")))

# -------------------------------
# 3. Loyalty Program Analytics
# -------------------------------

def load_customer_enrollments(spark: SparkSession, path: str) -> DataFrame:
    # Load customer enrollment data from CSV
    return spark.read.csv("file:///home/user/Downloads/pyspark-assessment-main/pyspark-multi-domain-assessment/data/customers.csv",header=True,inferSchema=True)

def load_loyalty_programs(spark: SparkSession, path: str) -> DataFrame:
    # Load loyalty program master data from CSV
    return spark.read.csv("file:///home/user/Downloads/pyspark-assessment-main/pyspark-multi-domain-assessment/data/loyalty_programs.csv",header=True,inferSchema=True)

def join_customers_with_loyalty_programs(customers_df: DataFrame, programs_df: DataFrame) -> DataFrame:
    # Join customers with their respective loyalty program info
    return customers_df.join(programs_df, on="program_id", how="inner")

def calculate_loyalty_scores(enrollment_df: DataFrame) -> DataFrame:
    # Compute loyalty_score = points_earned / transactions
    return enrollment_df.withColumn("loyalty_score",col("points_earned") / col("transactions"))


def flag_inactive_loyalty_customers(enrollment_df: DataFrame) -> DataFrame:
    # Add flag if last_active_date is before 2023-01-01
    return enrollment_df.withColumn("flag",when(col("last_active_date") < lit("2023-01-01"),True).otherwise(False))

def rank_customers_by_loyalty_score(enrollment_df: DataFrame) -> DataFrame:
    # Rank customers using loyalty_score descending
    return enrollment_df.orderBy(desc("customer_id"))

def loyalty_score_ratio(enrollment_df: DataFrame) -> DataFrame:
    # Calculate loyalty_score as a ratio of max loyalty score
    enrollment_df = enrollment_df.withColumn("loyalty_score", col("points_earned")/col("transactions"))
    max_loyalty_score =  enrollment_df.agg(max("loyalty_score")).collect()[0][0]
    return enrollment_df.withColumn("loyalty_score_ration",col("loyalty_score")/max_loyalty_score)


# -------------------------------
# 4. Inventory Data Processing
# -------------------------------

def load_inventory_data(spark: SparkSession, path: str) -> DataFrame:
    # Load inventory stock data
    return spark.read.csv("file:///home/user/Downloads/pyspark-assessment-main/pyspark-multi-domain-assessment/data/inventory.csv",header=True,inferSchema=True)

def flag_low_stock_items(inventory_df: DataFrame) -> DataFrame:
    # Add a boolean column if quantity < 20
    return inventory_df.withColumn("flag", when((col("quantity")<20),True).otherwise(False))

def rank_zones_by_stock_volume(inventory_df: DataFrame) -> DataFrame:
    # Group by zone, sum quantity, sort descending
    return inventory_df.groupBy("zone").agg(sum("quantity").alias("total_quantity")).orderBy(desc("total_quantity"))

def recently_updated_critical_stock(inventory_df: DataFrame, days: int) -> DataFrame:
    # Filter items updated recently (e.g., after a threshold date)
    inventory_df = inventory_df.withColumn("last_updated", to_date(col("last_updated")))
    return inventory_df.filter(datediff(current_date(), col("last_updated")) <= days)
    

def average_stock_per_zone(inventory_df: DataFrame) -> DataFrame:
    # Compute average quantity per zone
    return inventory_df.groupBy("zone").agg(avg("quantity").alias("total_quantity"))

def inventory_summary_per_item(inventory_df: DataFrame) -> DataFrame:
    # Aggregate item-wise zone count and total quantity
    return inventory_df.groupBy("zone").agg(count("item_id"),sum("quantity"))

def flag_critical_inventory(inventory_df: DataFrame) -> DataFrame:
    # Add flag where quantity < 5
    return inventory_df.withColumn("flag",((col("quantity")<5)))

# -------------------------------
# 5. User Profile Analysis
# -------------------------------

def load_user_profiles(spark: SparkSession, path: str) -> DataFrame:
    # Load user profiles CSV with inferred schema
    return spark.read.csv("file:///home/user/Downloads/pyspark-assessment-main/pyspark-multi-domain-assessment/data/user_profiles.csv",header=True,inferSchema=True)

def popular_user_age_segments(profiles_df: DataFrame) -> DataFrame:
    # Count users by age group and sort descending
    return profiles_df.groupBy("age_group").agg(count("*")).orderBy(desc("age_group"))

def flag_incomplete_profiles(profiles_df: DataFrame) -> DataFrame:
    # Add flag for records missing email or city
    return profiles_df.withColumn("flag",((col("city").isNotNull())) & (col("email").isNotNull()))

def popular_user_roles_by_city(profiles_df: DataFrame) -> DataFrame:
    # Group by city and role, count combinations
    return profiles_df.groupBy("city","role").count()

def registration_date_to_unix(profiles_df: DataFrame) -> DataFrame:
    # Convert registration_date to UNIX timestamp
    return profiles_df.withColumn("registration_date", unix_timestamp("registration_date"))

def user_engagement_by_city(profiles_df: DataFrame) -> DataFrame:
    # Average engagement score grouped by city
    return profiles_df.groupBy("city").agg(avg("engagement_score").alias("avg_engagement_score"))

def sorted_user_roles(profiles_df: DataFrame) -> list:
    # Return sorted list of distinct user roles
    sorted_df = profiles_df.select("role").distinct().orderBy(asc("role"))
    return list(sorted_df)