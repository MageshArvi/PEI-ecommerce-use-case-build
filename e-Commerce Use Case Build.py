# Databricks notebook source
# MAGIC %md
# MAGIC **Step 1: Write Tests That Fail**
# MAGIC
# MAGIC Here, we write initial failing tests to validate that the tables and views do not exist yet.

# COMMAND ----------

import unittest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, DateType

class DataPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder \
            .appName("e-Commerce Use Case Build") \
            .config("spark.jars.packages", "com.crealytics:spark-excel_2.12:0.13.4") \
            .getOrCreate()

        # Paths to datasets in ADLS Gen2 mounted directory
        cls.orders_path = "dbfs:/FileStore/Order.json"
        cls.customers_path = "dbfs:/FileStore/Customer_file.xlsx"
        cls.products_path = "dbfs:/FileStore/Product.csv"

        cls.spark.sql("DROP TABLE IF EXISTS raw_orders")
        cls.spark.sql("DROP TABLE IF EXISTS enriched_customers")
        cls.spark.sql("DROP TABLE IF EXISTS enriched_products")
        cls.spark.sql("DROP TABLE IF EXISTS enriched_orders")
        cls.spark.sql("DROP TABLE IF EXISTS profit_aggregate")

        # Define schema for orders.json
        cls.orders_schema = StructType([
            StructField("Row ID", IntegerType(), True),
            StructField("Order ID", StringType(), True),
            StructField("Order Date", DateType(), True),
            StructField("Ship Date", DateType(), True),
            StructField("Ship Mode", StringType(), True),
            StructField("Customer ID", StringType(), True),
            StructField("Product ID", StringType(), True),
            StructField("Quantity", IntegerType(), True),
            StructField("Price", DoubleType(), True),
            StructField("Discount", DoubleType(), True),
            StructField("Profit", DoubleType(), True)
        ])

    def test_raw_tables_creation(self):
        # Initial failing test for raw tables creation
        self.assertTrue(self.spark._jsparkSession.catalog().tableExists("raw_orders"))

    def test_enriched_tables_creation(self):
        # Initial failing test for enriched tables creation
        self.assertTrue(self.spark._jsparkSession.catalog().tableExists("enriched_customers"))
        self.assertTrue(self.spark._jsparkSession.catalog().tableExists("enriched_products"))

    def test_enriched_orders_table_creation(self):
        # Initial failing test for enriched orders table creation
        self.assertTrue(self.spark._jsparkSession.catalog().tableExists("enriched_orders"))

    def test_aggregate_table_creation(self):
        # Initial failing test for aggregate table creation
        self.assertTrue(self.spark._jsparkSession.catalog().tableExists("profit_aggregate"))

    def test_sql_queries(self):
        try:
            self.spark.sql("SELECT * FROM profit_aggregate")
            query_passed = True
        except Exception as e:
            query_passed = False
        self.assertTrue(query_passed)

# Create a suite and add tests
suite = unittest.TestSuite()
suite.addTest(DataPipelineTest('test_raw_tables_creation'))
suite.addTest(DataPipelineTest('test_enriched_tables_creation'))
suite.addTest(DataPipelineTest('test_enriched_orders_table_creation'))
suite.addTest(DataPipelineTest('test_aggregate_table_creation'))
suite.addTest(DataPipelineTest('test_sql_queries'))

# Run the tests
runner = unittest.TextTestRunner()
runner.run(suite)



# COMMAND ----------

# MAGIC %md
# MAGIC **Step 2: Refactor Code to Make Tests Pass**
# MAGIC
# MAGIC Now we implement the code to create the raw, enriched, and aggregate tables, ensuring the tests pass.

# COMMAND ----------

import unittest
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, DateType

class DataPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder \
            .appName("e-Commerce Use Case Build") \
            .config("spark.jars.packages", "com.crealytics:spark-excel_2.12:0.13.4") \
            .getOrCreate()

        # Paths to datasets in ADLS Gen2 mounted directory
        cls.orders_df = "/dbfs/FileStore/Order.json"
        cls.customers_path = "dbfs:/FileStore/Customer_file.xlsx"
        cls.products_path = "dbfs:/FileStore/Product.csv"

        # Define schema for orders.json
        cls.orders_schema = StructType([
            StructField("Row ID", IntegerType(), True),
            StructField("Order ID", StringType(), True),
            StructField("Order Date", DateType(), True),
            StructField("Ship Date", DateType(), True),
            StructField("Ship Mode", StringType(), True),
            StructField("Customer ID", StringType(), True),
            StructField("Product ID", StringType(), True),
            StructField("Quantity", IntegerType(), True),
            StructField("Price", DoubleType(), True),
            StructField("Discount", DoubleType(), True),
            StructField("Profit", DoubleType(), True)
        ])

    def test_raw_tables_creation(self):
        try:
            # Read datasets
            pandas_df = pd.read_json(self.orders_df)

            # Clean 'Price' column: Remove '%' character and convert to numeric
            pandas_df['Price'] = pandas_df['Price'].str.rstrip('%').astype(float)

            # Convert 'Order Date' and 'Ship Date' columns to datetime
            pandas_df['Order Date'] = pd.to_datetime(pandas_df['Order Date'], format='%d/%m/%Y', errors='coerce')
            pandas_df['Ship Date'] = pd.to_datetime(pandas_df['Ship Date'], format='%d/%m/%Y', errors='coerce')

            # Create Spark DataFrame
            spark_df = self.spark.createDataFrame(pandas_df, schema=self.orders_schema)

            # Display the DataFrame
            print("Orders DataFrame:")
            spark_df.show()

            customers_df = self.spark.read.format("com.crealytics.spark.excel").option("header", "true").option("inferSchema", "true").load(self.customers_path)
            print("Customers DataFrame:")
            customers_df.show()

            products_df = self.spark.read.csv(self.products_path, header=True, inferSchema=True)
            print("Products DataFrame:")
            products_df.show()

            # Create temporary views
            spark_df.createOrReplaceTempView("raw_orders")
            customers_df.createOrReplaceTempView("raw_customers")
            products_df.createOrReplaceTempView("raw_products")

            # Check if tables are created
            self.assertTrue(self.spark._jsparkSession.catalog().tableExists("raw_orders"))
            self.assertTrue(self.spark._jsparkSession.catalog().tableExists("raw_customers"))
            self.assertTrue(self.spark._jsparkSession.catalog().tableExists("raw_products"))
        except Exception as e:
            print(f"Error in test_raw_tables_creation: {e}")
            self.fail(e)

    def test_enriched_tables_creation(self):
        try:
            # Read raw tables
            customers_df = self.spark.table("raw_customers")
            products_df = self.spark.table("raw_products")

            # Clean and transform data
            customers_df = customers_df.withColumnRenamed("Customer ID", "CustomerID") \
                                       .withColumnRenamed("Customer Name", "CustomerName") \
                                       .withColumnRenamed("Country", "Country") \
                                       .withColumnRenamed("Segment", "Segment")
            
            products_df = products_df.withColumnRenamed("Product ID", "ProductID") \
                                     .withColumnRenamed("Category", "Category") \
                                     .withColumnRenamed("Sub-Category", "SubCategory")

            # Create enriched views
            enriched_customers_df = customers_df.select("CustomerID", "CustomerName", "Country", "Segment")
            enriched_customers_df.createOrReplaceTempView("enriched_customers")

            enriched_products_df = products_df.select("ProductID", "Category", "SubCategory")
            enriched_products_df.createOrReplaceTempView("enriched_products")

            # Check if tables are created
            self.assertTrue(self.spark._jsparkSession.catalog().tableExists("enriched_customers"))
            self.assertTrue(self.spark._jsparkSession.catalog().tableExists("enriched_products"))
        except Exception as e:
            print(f"Error in test_enriched_tables_creation: {e}")
            self.fail(e)

    def test_enriched_orders_table_creation(self):
        try:
            # Read raw and enriched tables
            raw_orders = self.spark.table("raw_orders")
            enriched_customers = self.spark.table("enriched_customers")
            enriched_products = self.spark.table("enriched_products")

            # Clean and transform data
            raw_orders = raw_orders.withColumnRenamed("Order ID", "OrderID") \
                                   .withColumnRenamed("Order Date", "OrderDate") \
                                   .withColumnRenamed("Customer ID", "CustomerID") \
                                   .withColumnRenamed("Product ID", "ProductID") \
                                   .withColumnRenamed("Profit", "Profit")

            # Create enriched orders view
            enriched_orders_df = raw_orders.join(enriched_customers, "CustomerID") \
                                           .join(enriched_products, "ProductID") \
                                           .withColumn("Profit", F.round("Profit", 2)) \
                                           .select("OrderID", "OrderDate", "CustomerID", "CustomerName", "Country", "ProductID", "Category", "SubCategory", "Profit")
            enriched_orders_df.createOrReplaceTempView("enriched_orders")

            # Check if table is created
            self.assertTrue(self.spark._jsparkSession.catalog().tableExists("enriched_orders"))

            # Show the enriched orders data
            print("Enriched Orders DataFrame:")
            enriched_orders_df.show()
        except Exception as e:
            print(f"Error in test_enriched_orders_table_creation: {e}")
            self.fail(e)

    def test_aggregate_table_creation(self):
        try:
            # Read enriched orders table
            enriched_orders_df = self.spark.table("enriched_orders")

            # Create aggregate view
            enriched_orders_df = enriched_orders_df.withColumn("Year", F.year("OrderDate"))
            aggregate_df = enriched_orders_df.groupBy("Year", "Category", "SubCategory", "CustomerName") \
                                             .agg(F.sum("Profit").alias("TotalProfit"))
            aggregate_df.createOrReplaceTempView("profit_aggregate")

            # Check if table is created
            self.assertTrue(self.spark._jsparkSession.catalog().tableExists("profit_aggregate"))

            # Show the aggregate data
            print("Aggregate DataFrame:")
            aggregate_df.show()
        except Exception as e:
            print(f"Error in test_aggregate_table_creation: {e}")
            self.fail(e)

    def test_sql_queries(self):
        try:
            # Run SQL queries
            profit_by_year = self.spark.sql("SELECT Year, SUM(TotalProfit) AS Profit FROM profit_aggregate GROUP BY Year ORDER BY Profit DESC")
            print("Profit by Year:")
            profit_by_year.show()

            profit_by_year_and_category = self.spark.sql("SELECT Year, Category, SUM(TotalProfit) AS Profit FROM profit_aggregate GROUP BY Year, Category ORDER BY Profit DESC")
            print("Profit by Year and Category:")
            profit_by_year_and_category.show()

            profit_by_customer = self.spark.sql("SELECT CustomerName, SUM(TotalProfit) AS Profit FROM profit_aggregate GROUP BY CustomerName ORDER BY Profit DESC")
            print("Profit by Customer:")
            profit_by_customer.show()

            profit_by_customer_and_year = self.spark.sql("SELECT CustomerName, Year, SUM(TotalProfit) AS Profit FROM profit_aggregate GROUP BY CustomerName, Year ORDER BY Profit DESC")
            print("Profit by Customer and Year:")
            profit_by_customer_and_year.show()

            # Check if results are not empty
            self.assertGreater(profit_by_year.count(), 0)
            self.assertGreater(profit_by_year_and_category.count(), 0)
            self.assertGreater(profit_by_customer.count(), 0)
            self.assertGreater(profit_by_customer_and_year.count(), 0)
        except Exception as e:
            print(f"Error in test_sql_queries: {e}")
            self.fail(e)

# Create a suite and add tests
suite = unittest.TestSuite()
suite.addTest(DataPipelineTest('test_raw_tables_creation'))
suite.addTest(DataPipelineTest('test_enriched_tables_creation'))
suite.addTest(DataPipelineTest('test_enriched_orders_table_creation'))
suite.addTest(DataPipelineTest('test_aggregate_table_creation'))
suite.addTest(DataPipelineTest('test_sql_queries'))

# Run the tests
runner = unittest.TextTestRunner()
runner.run(suite)


# COMMAND ----------

# MAGIC %md
# MAGIC **Step 3: Optimize and Clean Up Code (Refactor Phase)**
# MAGIC
# MAGIC In this step, we ensure the code is clean, efficient, and free of redundancies.

# COMMAND ----------

import unittest
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, DateType

class DataPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder \
            .appName("e-Commerce Use Case Build") \
            .config("spark.jars.packages", "com.crealytics:spark-excel_2.12:0.13.4") \
            .getOrCreate()

        # Paths to datasets in ADLS Gen2 mounted directory
        cls.orders_df = "/dbfs/FileStore/Order.json"
        cls.customers_path = "dbfs:/FileStore/Customer_file.xlsx"
        cls.products_path = "dbfs:/FileStore/Product.csv"

        # Define schema for orders.json
        cls.orders_schema = StructType([
            StructField("Row ID", IntegerType(), True),
            StructField("Order ID", StringType(), True),
            StructField("Order Date", DateType(), True),
            StructField("Ship Date", DateType(), True),
            StructField("Ship Mode", StringType(), True),
            StructField("Customer ID", StringType(), True),
            StructField("Product ID", StringType(), True),
            StructField("Quantity", IntegerType(), True),
            StructField("Price", DoubleType(), True),
            StructField("Discount", DoubleType(), True),
            StructField("Profit", DoubleType(), True)
        ])

    def create_temp_view(self, df, view_name):
        if df:
            df.createOrReplaceTempView(view_name)
            self.assertTrue(self.spark._jsparkSession.catalog().tableExists(view_name))
            print(f"{view_name} table created")

    def test_raw_tables_creation(self):
        try:
            # Read datasets
            pandas_df = pd.read_json(self.orders_df)

            # Clean 'Price' column: Remove '%' character and convert to numeric
            pandas_df['Price'] = pandas_df['Price'].str.rstrip('%').astype(float)

            # Convert 'Order Date' and 'Ship Date' columns to datetime
            pandas_df['Order Date'] = pd.to_datetime(pandas_df['Order Date'], format='%d/%m/%Y', errors='coerce')
            pandas_df['Ship Date'] = pd.to_datetime(pandas_df['Ship Date'], format='%d/%m/%Y', errors='coerce')

            # Create Spark DataFrame
            spark_df = self.spark.createDataFrame(pandas_df, schema=self.orders_schema)
            print("Orders DataFrame:")
            spark_df.show()

            customers_df = self.spark.read.format("com.crealytics.spark.excel").option("header", "true").option("inferSchema", "true").load(self.customers_path)
            print("Customers DataFrame:")
            customers_df.show()

            products_df = self.spark.read.csv(self.products_path, header=True, inferSchema=True)
            print("Products DataFrame:")
            products_df.show()

            # Create temporary views
            self.create_temp_view(spark_df, "raw_orders")
            self.create_temp_view(customers_df, "raw_customers")
            self.create_temp_view(products_df, "raw_products")
        except Exception as e:
            print(f"Error in test_raw_tables_creation: {e}")
            self.fail(e)

    def test_enriched_tables_creation(self):
        try:
            # Read raw tables
            customers_df = self.spark.table("raw_customers")
            products_df = self.spark.table("raw_products")

            # Clean and transform data
            customers_df = customers_df.withColumnRenamed("Customer ID", "CustomerID") \
                                       .withColumnRenamed("Customer Name", "CustomerName") \
                                       .withColumnRenamed("Country", "Country") \
                                       .withColumnRenamed("Segment", "Segment")
            
            products_df = products_df.withColumnRenamed("Product ID", "ProductID") \
                                     .withColumnRenamed("Category", "Category") \
                                     .withColumnRenamed("Sub-Category", "SubCategory")

            # Create enriched views
            enriched_customers_df = customers_df.select("CustomerID", "CustomerName", "Country", "Segment")
            self.create_temp_view(enriched_customers_df, "enriched_customers")

            enriched_products_df = products_df.select("ProductID", "Category", "SubCategory")
            self.create_temp_view(enriched_products_df, "enriched_products")
        except Exception as e:
            print(f"Error in test_enriched_tables_creation: {e}")
            self.fail(e)

    def test_enriched_orders_table_creation(self):
        try:
            # Read raw and enriched tables
            raw_orders = self.spark.table("raw_orders")
            enriched_customers = self.spark.table("enriched_customers")
            enriched_products = self.spark.table("enriched_products")

            # Clean and transform data
            raw_orders = raw_orders.withColumnRenamed("Order ID", "OrderID") \
                                   .withColumnRenamed("Order Date", "OrderDate") \
                                   .withColumnRenamed("Customer ID", "CustomerID") \
                                   .withColumnRenamed("Product ID", "ProductID") \
                                   .withColumnRenamed("Profit", "Profit")

            # Create enriched orders view
            enriched_orders_df = raw_orders.join(enriched_customers, "CustomerID") \
                                           .join(enriched_products, "ProductID") \
                                           .withColumn("Profit", F.round("Profit", 2)) \
                                           .select("OrderID", "OrderDate", "CustomerID", "CustomerName", "Country", "ProductID", "Category", "SubCategory", "Profit")
            self.create_temp_view(enriched_orders_df, "enriched_orders")

            # Show the enriched orders data
            print("Enriched Orders DataFrame:")
            enriched_orders_df.show()
        except Exception as e:
            print(f"Error in test_enriched_orders_table_creation: {e}")
            self.fail(e)

    def test_aggregate_table_creation(self):
        try:
            # Read enriched orders table
            enriched_orders_df = self.spark.table("enriched_orders")

            # Create aggregate view
            enriched_orders_df = enriched_orders_df.withColumn("Year", F.year("OrderDate"))
            aggregate_df = enriched_orders_df.groupBy("Year", "Category", "SubCategory", "CustomerName") \
                                             .agg(F.sum("Profit").alias("TotalProfit"))
            self.create_temp_view(aggregate_df, "profit_aggregate")

            # Show the aggregate data
            print("Aggregate DataFrame:")
            aggregate_df.show()
        except Exception as e:
            print(f"Error in test_aggregate_table_creation: {e}")
            self.fail(e)

    def test_sql_queries(self):
        try:
            # Run SQL queries
            profit_by_year = self.spark.sql("SELECT Year, SUM(TotalProfit) AS Profit FROM profit_aggregate GROUP BY Year ORDER BY Profit DESC")
            print("Profit by Year:")
            profit_by_year.show()

            profit_by_year_and_category = self.spark.sql("SELECT Year, Category, SUM(TotalProfit) AS Profit FROM profit_aggregate GROUP BY Year, Category ORDER BY Profit DESC")
            print("Profit by Year and Category:")
            profit_by_year_and_category.show()

            profit_by_customer = self.spark.sql("SELECT CustomerName, SUM(TotalProfit) AS Profit FROM profit_aggregate GROUP BY CustomerName ORDER BY Profit DESC")
            print("Profit by Customer:")
            profit_by_customer.show()

            profit_by_customer_and_year = self.spark.sql("SELECT CustomerName, Year, SUM(TotalProfit) AS Profit FROM profit_aggregate GROUP BY CustomerName, Year ORDER BY Profit DESC")
            print("Profit by Customer and Year:")
            profit_by_customer_and_year.show()

            # Check if results are not empty
            self.assertGreater(profit_by_year.count(), 0)
            self.assertGreater(profit_by_year_and_category.count(), 0)
            self.assertGreater(profit_by_customer.count(), 0)
            self.assertGreater(profit_by_customer_and_year.count(), 0)
        except Exception as e:
            print(f"Error in test_sql_queries: {e}")
            self.fail(e)

# Create a suite and add tests
suite = unittest.TestSuite()
suite.addTest(DataPipelineTest('test_raw_tables_creation'))
suite.addTest(DataPipelineTest('test_enriched_tables_creation'))
suite.addTest(DataPipelineTest('test_enriched_orders_table_creation'))
suite.addTest(DataPipelineTest('test_aggregate_table_creation'))
suite.addTest(DataPipelineTest('test_sql_queries'))

# Run the tests
runner = unittest.TextTestRunner()
runner.run(suite)

