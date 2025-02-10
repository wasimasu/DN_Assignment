"""
Delaware North: Take-Home Assignment

Technical Prerequisites:
  - Python 3+
  - Spark: pip install delta-spark
  - Java Runtime: https://java.com/en/download/manual.jsp

Assignment Background:
    - You are a freelance analytics consultant who has partnered with the TTPD (Tiny Town Police Department)
      to analyze speeding tickets that have been given to the adult citizens of Tiny Town over the 2020-2023 period.
    - Inside the folder "ttpd_data" you will find a directory of data for Tiny Town. This dataset will need to be "ingested" for analysis.
    - The solutions must use the Dataframes API.
    - You will need to ingest this data into a PySpark environment and answer the following three questions for the TTPD.

Questions:
    1. Which police officer was handed the most speeding tickets?
        - Police officers are recorded as citizens. Find in the data what differentiates an officer from a non-officer.
    2. What 3 months (year + month) had the most speeding tickets? 
        - Bonus: What overall month-by-month or year-by-year trends, if any, do you see?
    3. Using the ticket fee table below, who are the top 10 people who have spent the most money paying speeding tickets overall?

Ticket Fee Table:
    - Ticket (base): $30
    - Ticket (base + school zone): $60
    - Ticket (base + construction work zone): $60
    - Ticket (base + school zone + construction work zone): $120
"""
import xml.etree.ElementTree as ET
import pandas as pd
import os
from delta import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, col, count, month, year, when, sum

def convert_xml_to_csv(input_folder, output_folder, file_count):
    os.makedirs(output_folder, exist_ok=True)  
    
    for i in range(file_count + 1): 
        
        if i <= 61:
            file_prefix = "20240503111609_automobiles_"
        else:
            file_prefix = "20240503111610_automobiles_"
        
        file_path = os.path.join(input_folder, f"{file_prefix}{i}.xml")
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        
        tree = ET.parse(file_path)
        root = tree.getroot()
        
       
        data = []
        for automobile in root.findall("automobile"):
            data.append({
                "person_id": automobile.find("person_id").text,
                "license_plate": automobile.find("license_plate").text,
                "vin": automobile.find("vin").text,
                "color": automobile.find("color").text,
                "year": automobile.find("year").text,
            })
        
        
        df = pd.DataFrame(data)
        
        
        csv_path = os.path.join(output_folder, f"automobiles_{i}.csv")
        df.to_csv(csv_path, index=False)



def get_spark_session() -> SparkSession:
    """Retrieves or creates an active Spark Session for Delta operations
    
    Returns:
        spark (SparkSession): the active Spark Session
    """
    builder = SparkSession \
        .builder \
        .appName('takehome') \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")\
        .config("spark.jars.packages", "com.databricks:spark-xml_2.12:0.14.0")

    return configure_spark_with_delta_pip(builder).getOrCreate()

def main():
    convert_xml_to_csv("ttpd_data", "converted_data", 81)

    spark: SparkSession = get_spark_session()

    #speeding tickets data frame
    df_json = spark.read.option("multiline", "true").json("ttpd_data/20240503111610_speeding_tickets_*.json")

    df_exploded = df_json.select(explode(col("speeding_tickets")).alias("ticket"))

    
    df_speeding_tickets = df_exploded.select(
        col("ticket.id").alias("id"),
        col("ticket.ticket_time").alias("ticket_time"),
        col("ticket.license_plate").alias("license_plate"),
        col("ticket.officer_id").alias("officer_id"),
        col("ticket.speed_limit").alias("speed_limit"),
        col("ticket.recorded_mph_over_limit").alias("recorded_mph_over_limit"),
        col("ticket.school_zone_ind").alias("school_zone_ind"),
        col("ticket.work_zone_ind").alias("work_zone_ind")
    )

    #people data frame
    df_people = spark.read.option("header", "true").option("delimiter", "|").csv("ttpd_data/20240503111609_people_*.csv")

    #automobile data frame
    df_automobile = spark.read.option("header", "true").option("delimiter", ",").csv("converted_data/automobiles_*.csv")

   

    #Q1. Which police officer was handed the most speeding tickets?
    
    df_police_officers = df_people.filter(col("profession") == "Police Officer").select(
        col("id").alias("officer_id"), col("first_name"), col("last_name")
    )

    
    df_joined = df_speeding_tickets.join(df_police_officers, "officer_id", "inner")

    
    df_result = df_joined.groupBy("officer_id", "first_name", "last_name").agg(
        count("officer_id").alias("ticket_count")
    )

    
    df_top_officer = df_result.orderBy(col("ticket_count").desc()).limit(1)

    
    df_top_officer.show(truncate = False)


    #Q2. What 3 months (year + month) had the most speeding tickets? 
    
    df_tickets_with_date = df_speeding_tickets.withColumn(
        "year", year(col("ticket_time"))
    ).withColumn(
        "month", month(col("ticket_time"))
    )

    
    df_monthly_tickets = df_tickets_with_date.groupBy("year", "month").agg(
        count("id").alias("ticket_count")
    )

    
    df_top_3_months = df_monthly_tickets.orderBy(col("ticket_count").desc()).limit(3)

    
    df_top_3_months.show(truncate = False)

    #3. Using the ticket fee table below, who are the top 10 people who have spent the most money paying speeding tickets overall?
    df_tickets_automobile = df_speeding_tickets.join(df_automobile, "license_plate", "inner")

    
    df_people = df_people.withColumnRenamed("id", "ppl_id")
    df_final = df_tickets_automobile.join(df_people, df_tickets_automobile.person_id == df_people.ppl_id)

    
    df_with_fees = df_final.withColumn(
        "ticket_fee",
        when(
            (col("school_zone_ind") == True) & (col("work_zone_ind") == True), 120
        ).when(
            (col("school_zone_ind") == True) | (col("work_zone_ind") == False), 60
        ).when(
            (col("school_zone_ind") == False) | (col("work_zone_ind") == False), 60
        ).otherwise(30)
    )

    
    df_total_fees = df_with_fees.groupBy("ppl_id", "first_name", "last_name").agg(
        sum("ticket_fee").alias("total_spent")
    )

    
    df_top_10_spenders = df_total_fees.orderBy(col("total_spent").desc()).limit(10)

    
    df_top_10_spenders.show(truncate = False)
    


if __name__ == '__main__':
    main()
