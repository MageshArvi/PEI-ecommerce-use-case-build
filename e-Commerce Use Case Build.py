# Databricks notebook source
import unittest
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, DateType

class DataPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
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
        except Exception as e:
            print(f"Error in setUpClass: {e}")
            raise e

    @classmethod
    
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

            # Create temporary views if not null
            if spark_df:
                spark_df.createOrReplaceTempView("raw_orders")
                print(self.spark.sql("select * from raw_orders"))

            if customers_df:
                customers_df.createOrReplaceTempView("raw_customers")
                print(self.spark.sql("select * from raw_customers"))

            if products_df:
                products_df.createOrReplaceTempView("raw_products")
                print(self.spark.sql("select * from raw_products"))

            # Check if table is created
            if spark_df:
                self.assertTrue(self.spark._jsparkSession.catalog().tableExists("raw_orders"), "raw_orders table does not exist")
                print("raw_orders table created")
            if customers_df:
                self.assertTrue(self.spark._jsparkSession.catalog().tableExists("raw_customers"), "raw_customers table does not exist")
                print("raw_customers table created")
            if products_df:
                self.assertTrue(self.spark._jsparkSession.catalog().tableExists("raw_products"), "raw_products table does not exist")
                print("raw_products table created")
        except Exception as e:
            print(f"Error in test_raw_tables_creation: {e}")
            self.fail(e)
            
    def test_enriched_tables_creation(self):
        try:
            # Read raw tables
            customers_df = self.spark.table("raw_customers")
            products_df = self.spark.table("raw_products")

            # Clean and transform data
            if customers_df:
                customers_df = customers_df.withColumnRenamed("Customer ID", "CustomerID") \
                                           .withColumnRenamed("Customer Name", "CustomerName") \
                                           .withColumnRenamed("Country", "Country") \
                                           .withColumnRenamed("Segment", "Segment")
            
            if products_df:
                products_df = products_df.withColumnRenamed("Product ID", "ProductID") \
                                         .withColumnRenamed("Category", "Category") \
                                         .withColumnRenamed("Sub-Category", "SubCategory")

            # Create enriched views if not null
            if customers_df:
                enriched_customers_df = customers_df.select("CustomerID", "CustomerName", "Country", "Segment")
                enriched_customers_df.createOrReplaceTempView("enriched_customers")

            if products_df:
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
            if raw_orders and enriched_customers and enriched_products:
                raw_orders = raw_orders.withColumnRenamed("Order ID", "OrderID") \
                                       .withColumnRenamed("Order Date", "OrderDate") \
                                       .withColumnRenamed("Customer ID", "CustomerID") \
                                       .withColumnRenamed("Customer ID", "CustomerID") \
									   .withColumnRenamed("Order Date", "OrderDate") \
									   .withColumnRenamed("Customer ID", "CustomerID") \
									   .withColumnRenamed("Product ID", "ProductID") \
									   .withColumnRenamed("Profit", "Profit")

            # Create enriched orders view if raw_orders and enriched_customers and enriched_products are not null
            if raw_orders and enriched_customers and enriched_products:
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

            # Create aggregate view if enriched_orders_df is not null
            if enriched_orders_df:
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
            profit_by_year = self.spark.sql("SELECT Year, SUM(TotalProfit) AS Profit FROM profit_aggregate GROUP BY Year order by Profit desc")
            print("Profit by Year:")
            profit_by_year.show()

            profit_by_year_and_category = self.spark.sql("SELECT Year, Category, SUM(TotalProfit) AS Profit FROM profit_aggregate GROUP BY Year, Category order by Profit desc")
            print("Profit by Year and Category:")
            profit_by_year_and_category.show()

            profit_by_customer = self.spark.sql("SELECT CustomerName, SUM(TotalProfit) AS Profit FROM profit_aggregate GROUP BY CustomerName order by Profit desc")
            print("Profit by Customer:")
            profit_by_customer.show()

            profit_by_customer_and_year = self.spark.sql("SELECT CustomerName, Year, SUM(TotalProfit) AS Profit FROM profit_aggregate GROUP BY CustomerName, Year order by Profit desc")
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
print("Test for test_raw_tables_creation completed")
suite.addTest(DataPipelineTest('test_enriched_tables_creation'))
suite.addTest(DataPipelineTest('test_enriched_orders_table_creation'))
suite.addTest(DataPipelineTest('test_aggregate_table_creation'))
suite.addTest(DataPipelineTest('test_sql_queries'))

# Run the tests
try:
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    if result.wasSuccessful():
        print("All tests passed successfully.")
    else:
        print("Some tests failed.")
except Exception as e:
    print(f"An error occurred while running the tests: {e}")



# COMMAND ----------
