# solution.py for RetailOrdersAnalytics - PySpark Assessment (Skeleton)
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *


# 1. Load CSV with header
def load_orders_data(spark: SparkSession, path: str) -> DataFrame:
    return spark.read.csv(path, header=True, inferSchema=True)

# 2. Load with schema
def load_orders_data_with_schema(spark: SparkSession, path: str) -> DataFrame:
    schema = StructType(
        [StructField('order_id',StringType(),True),
         StructField("customer_id",StringType(),True),
         StructField("order_date",StringType(),True),
         StructField("ship_date",StringType(),True),
         StructField("category",StringType(),True),
         StructField("city",StringType(),True),
         StructField("quantity",IntegerType(),True),
         StructField("unit_price",DoubleType(),True),
         StructField("total_price",DoubleType(),True),
         StructField("discount",DoubleType(),True),
         StructField("final_price",DoubleType(),True),
         StructField("payment_mode",StringType(),True),
         StructField("is_returned",StringType(),True)
         ]
    )
    return spark.read.csv(path, schema=schema)

# 3. Create DataFrame from list with 'sub_category' added
def create_sample_orders(spark: SparkSession) -> DataFrame:
    # Updated sample data including 'sub_category', 'status' and 'region'
    data = [
        ("O0001", "C001", "2023-01-02", "2023-01-05", "Electronics", "New York", 2, 120.5, 241.0, 0.1, 216.9, "Credit Card", "No", "Delivered", "Northeast", "Smartphone"),
        ("O0002", "C002", "2023-01-04", "2023-01-07", "Books", "Los Angeles", 1, 35.0, 35.0, 0.0, 35.0, "UPI", "Yes", "Returned", "West", "Fiction"),
        ("O0003", "C003", "2023-01-10", "2023-01-12", "Clothing", "Chicago", 3, 40.0, 120.0, 0.05, 114.0, "Debit Card", "No", "Delivered", "Midwest", "T-Shirts"),
        ("O0004", "C004", "2023-01-15", "2023-01-17", "Home Decor", "Houston", 5, 75.0, 375.0, 0.15, 318.75, "Net Banking", "No", "Delivered", "South", "Furniture"),
        ("O0005", "C005", "2023-01-20", "2023-01-25", "Groceries", "Phoenix", 10, 20.0, 200.0, 0.0, 200.0, "Cash", "Yes", "Cancelled", "Southwest", "Vegetables"),
        ("O0006", "C001", "2023-01-22", "2023-01-24", "Electronics", "New York", 1, 500.0, 500.0, 0.1, 450.0, "Credit Card", "No", "Delivered", "Northeast", "Laptop"),
        ("O0007", "C002", "2023-01-28", "2023-01-30", "Books", "Los Angeles", 2, 30.0, 60.0, 0.05, 57.0, "UPI", "No", "Delivered", "West", "Non-Fiction"),
        ("O0008", "C006", "2023-01-30", "2023-02-02", "Clothing", "Chicago", 4, 25.0, 100.0, 0.1, 90.0, "Debit Card", "No", "Returned", "Midwest", "Jeans"),
        ("O0009", "C007", "2023-02-01", "2023-02-04", "Home Decor", "Houston", 2, 150.0, 300.0, 0.2, 240.0, "Net Banking", "Yes", "Cancelled", "South", "Curtains"),
        ("O0010", "C008", "2023-02-03", "2023-02-06", "Groceries", "Phoenix", 6, 15.0, 90.0, 0.0, 90.0, "Cash", "No", "Delivered", "Southwest", "Fruits")
    ]
    
    # Define schema for the DataFrame with additional 'sub_category', 'status', and 'region' columns
    schema = StructType([
        StructField("order_id", StringType(), True),
        StructField("customer_id", StringType(), True),
        StructField("order_date", StringType(), True),
        StructField("ship_date", StringType(), True),
        StructField("category", StringType(), True),
        StructField("city", StringType(), True),
        StructField("quantity", IntegerType(), True),
        StructField("unit_price", DoubleType(), True),
        StructField("total_price", DoubleType(), True),
        StructField("discount", DoubleType(), True),
        StructField("final_price", DoubleType(), True),
        StructField("payment_mode", StringType(), True),
        StructField("is_returned", StringType(), True),
        StructField("status", StringType(), True),     # New column: status of the order
        StructField("region", StringType(), True),      # New column: region of the order
        StructField("sub_category", StringType(), True) # New column: sub_category of the product
    ])
    
    # Create the DataFrame using the sample data and schema
    orders_df = spark.createDataFrame(data, schema=schema)    
    return orders_df


# 4. Add computed column (total_price)
def add_total_price(df: DataFrame) -> DataFrame:
    df = df.withColumn("total_price", col("quantity") * col("unit_price"))
    return df.select("quantity","unit_price","total_price")

# 5. Filter by status = Delivered
def filter_delivered_orders(df: DataFrame) -> DataFrame:
    return df.filter(col("status")=="Delivered")

# 6. Drop unnecessary column
def drop_status_column(df: DataFrame) -> DataFrame:
    return df.drop(col("status"))

# 7. Rename a column
def rename_customer_id(df: DataFrame) -> DataFrame:
    return df.withColumnRenamed("customer_id", "rename_customer_id")

# 8. Cast quantity to double
def cast_quantity_to_double(df: DataFrame) -> DataFrame:
    return df.withColumn("quantity",col("quantity").cast("double"))

# 9. Get unique categories as list
def list_unique_categories(df: DataFrame) -> list:
    unique_cat = df.select("category").distinct()
    return list(unique_cat)

# 10. Total quantity per region
def total_quantity_by_region(df: DataFrame) -> DataFrame:
    return df.groupBy("region").agg(count("quantity").alias("total_quantity"))

# 11. Max unit price by category
def max_price_by_category(df: DataFrame) -> DataFrame:
    return df.groupBy("category").agg(max("unit_price").alias("max_unit_price")).orderBy(desc("max_unit_price")).limit(1)

# 12. Top 3 categories by revenue
def top_categories_by_revenue(df: DataFrame) -> DataFrame:
    df = df.withColumn("revenue",col("unit_price").cast("double")*col("quantity").cast("double"))
    top_df = df.groupBy("category").agg(sum("revenue").alias("total_revenue")).orderBy(col("total_revenue").desc()).limit(3)
    df.select("category", "unit_price", "quantity", "revenue").show(5)
    return top_df

# 13. Orders placed after specific date
def filter_orders_after_date(df: DataFrame, date_str: str) -> DataFrame:
    return df.filter(col("order_date")>lit("date_str"))

# 14. Year-wise revenue
def yearly_revenue(df: DataFrame) -> DataFrame:
    df = df.withColumn("year",year(to_date(col("order_date"), "yyyy-MM-dd")))
    df = df.withColumn("revenue", col("unit_price").cast("double") * col("quantity").cast("double"))
    return df.groupBy("year").agg(sum("revenue").alias("total_revenue"))

# 15. Drop duplicates
def remove_duplicate_orders(df: DataFrame) -> DataFrame:
    return df.dropDuplicates()

# 16. Orders per category per region
def category_region_count(df: DataFrame) -> DataFrame:
    return df.groupBy("category").agg(count("region").alias("region_count"))

# 17. Join with returns
def join_with_returns(df: DataFrame, returns_df: DataFrame) -> DataFrame:
    return df.join(returns_df, on="order_id", how="inner")

# 18. Filter orders with quantity > 10
def filter_large_orders(df: DataFrame) -> DataFrame:
    return df.filter(col("quantity")>10)

# 19. Replace null prices
def replace_null_prices(df: DataFrame) -> DataFrame:
    return df.fillna(0)

# 20. Calculate average unit price per category
def average_price_per_category(df: DataFrame) -> DataFrame:
    return df.groupBy("category").agg(avg("unit_price").alias("total_unit_price"))

# 21. Filter by multiple conditions
def filter_north_electronics(df: DataFrame) -> DataFrame:
    return df.filter((col("region") == "North") & (col("category") == "Electronics"))

# 22. Count delivered vs returned
def count_status_types(df: DataFrame) -> DataFrame:
    return df.groupBy("status").agg(count("*").alias("status_count"))

# 23. Find most frequent sub_category
def most_common_sub_category(df: DataFrame) -> str:
    return df.groupBy("sub_category").agg(count("*").alias("total_sub_category")).orderBy(desc("total_sub_category")).limit(1).collect()[0]["sub_category"]

# 24. Revenue per sub_category in tuple
def revenue_by_sub_category(df: DataFrame, target: str) -> tuple:
    df = df.withColumn("revenue", col("unit_price").cast("double") * col("quantity").cast("double"))
    revenue_df = df.groupBy("sub_category").agg(sum("revenue").alias("total_revenue"))
    return tuple(revenue_df)

# 25. Find null counts
def null_count(df: DataFrame) -> DataFrame:
    return df.fillna("0")

# 26. Get earliest order
def earliest_order(df: DataFrame) -> str:
    earliest = df.agg(min("order_date").alias("earliest_order")).collect()[0]["earliest_order"]
    return earliest

# 27. Remove orders before year
def remove_orders_before_year(df: DataFrame, year_threshold: int) -> DataFrame:
    return df.filter(year(to_date(col("order_date"), "yyyy-MM-dd")) >= year_threshold)

# 28. Total revenue by customer
def revenue_by_customer(df: DataFrame) -> DataFrame:
    df = df.withColumn("revenue", col("unit_price").cast("double") * col("quantity").cast("double"))
    return df.groupBy("customer_id").agg(sum("revenue").alias("total_revenue"))

# 29. Group by multiple columns with aggregation
def quantity_price_by_category_region(df: DataFrame) -> DataFrame:
    return df.groupBy("category","region").agg(sum("quantity").alias("total_quantity"),sum(col("quantity") * col("unit_price")).alias("total_price"))

# 30. Most valuable order
def highest_value_order(df: DataFrame) -> str:
    df = df.withColumn("revenue", col("unit_price").cast("double") * col("quantity").cast("double"))
    highest_order = df.groupBy("order_id").agg(sum("revenue").alias("total_revenue")).orderBy(col("total_revenue").desc()).limit(1).collect()[0]["order_id"]
    return highest_order