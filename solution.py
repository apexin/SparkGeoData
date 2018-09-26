# start to the Docker
# docker exec -it ws-data-spark_master_1 /bin/bash
# /usr/spark-2.3.1# pyspark

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, \
                              TimestampType, StringType, DoubleType
from pyspark.sql.window import Window
from pyspark.sql.functions import *  # noqa


def main():
    spark = SparkSession\
        .builder\
        .appName("NZSolution")\
        .getOrCreate()

    # Create the schema with proper column names for the request data
    log_schema = StructType(
        [
            StructField("_ID", IntegerType(), True),
            StructField("TimeSt", TimestampType(), True),
            StructField("Country", StringType(), True),
            StructField("Province", StringType(), True),
            StructField("City", StringType(), True),
            StructField("Latitude", DoubleType(), True),
            StructField("Longitude", DoubleType(), True)
        ]
    )

    # Load the data
    raw_logs = spark.read.load("/tmp/data/DataSample.csv", format="csv",
                               header="false", schema=log_schema)

    # Test the data count
    raw_logs.count()

    # Clean suspicious data duplicates and null values
    # (Potential problem if the duplicate is not unique to _ID?)
    logs = raw_logs.dropDuplicates(['TimeSt', 'Latitude', 'Longitude'])
    logs = logs.na.drop(how="all")

    # Test resultin data count
    logs.count()

    print("Request Logs clean data")
    logs.show()

    # Create the schema for POI data
    poi_schema = StructType(
        [
            StructField("POIID", StringType(), True),
            StructField("Latitude", DoubleType(), True),
            StructField("Longitude", DoubleType(), True)
        ]
    )

    # Load and clean the data, as with request process
    poi = spark.read.load("/tmp/data/POIList.csv", format="csv", header="false",
                          schema=poi_schema)
    poi = poi.na.drop(how="all")
    poi = poi.dropDuplicates(['Latitude', 'Longitude'])

    print("POI clean data")
    poi.show()


    ### 1. Label ###

    # Find the minimum distance to each POI (naiive approach, better alg possible)
    # First, find all the distances to each POI, in kms
    logs = logs.registerTempTable('logs')
    poi = poi.registerTempTable('poi')
    query = "SELECT l._id as id, p.latitude as poi_lat, p.longitude as poi_lon,\
                l.latitude as log_lat, l.longitude as log_lon,\
                l.timest, l.country, l.province, l.city,\
                acos(sin(radians(l.latitude)) * sin(radians(p.latitude))\
                    + cos(radians(l.latitude)) * cos(radians(p.latitude))\
                    * cos(radians(p.longitude) - radians(l.longitude)))*6371 as dist_km,\
                p.poiid\
            FROM logs l\
            CROSS JOIN (\
            SELECT poiid,\
                    latitude,\
                    longitude\
             FROM poi\
            ) AS p ON 1=1\
         ORDER BY dist_km"
    dist_data = spark.sql(query)
    dist_data.show()

    # Keep only the minimal between the request and the POIs
    partition = Window.partitionBy(['id', 'log_lat', 'log_lon', 'timest']).orderBy('dist_km')
    min_dist = dist_data.withColumn('rn', row_number().over(partition)).where(col('rn') == 1).drop('rn')

    # Testing the data
    min_dist.filter(min_dist.log_lat == 42.4247).show()

    # Testing distance with other package
    # from geopy.distance import great_circle
    # poi_coords = (45.521629, -73.566024)
    # log_coords = (42.4247, -82.1755)
    # print(great_circle(poi_coords, log_coords).kilometers)

    # Table with labels assignment
    print("POI to request assignment")
    min_dist.show()

    ### 2. Analysis ###

    # i. Average and Standard Deviation
    avg_std_dist = min_dist.groupBy("poiid").agg(stddev("dist_km"), avg("dist_km"))
    avg_std_dist = avg_std_dist.select(
        col("poiid"),
        col("stddev_samp(dist_km)").alias("stddev"),
        col("avg(dist_km)").alias("avg"))

    print("Table with AVG and STDDEV of distances")
    avg_std_dist.show()

    # ii. The Radius is the same as the furtherst point from the POI what was
    # assigned to the corresponding POI
    max_cnt_dist = min_dist.groupBy("poiid").agg(max("dist_km"), count("*"))
    max_cnt_dist = max_cnt_dist.select(
        col("poiid"),
        col("max(dist_km)").alias("radius"),
        col("count(1)").alias("request_cnt"))
    max_cnt_dist = max_cnt_dist.withColumn(
        "circle_area", 3.14*pow(max_cnt_dist.radius, 2))
    max_cnt_dist = max_cnt_dist.withColumn(
        "density", max_cnt_dist.request_cnt/max_cnt_dist.circle_area)

    print("Table with Radius and Density data")
    max_cnt_dist.show()


    ### 3. Model ###

    # i. Visualisation
    # First, we need to center the data with the mean, i.e. density mean
    # Finding the density mean
    density_mean = max_cnt_dist.agg(mean("density"))
    density_mean = density_mean.select(col("avg(density)").alias("density_mean"))
    plot_data = max_cnt_dist.crossJoin(density_mean)

    # Centering data by substracting the mean from all the density values
    plot_data = plot_data.withColumn(
        "centered_density", plot_data.density - plot_data.density_mean)

    # Data cleanup step
    plot_data = plot_data.drop("radius", "request_cnt", "circle_area")

    # Suggested model is based on min/max scaling formula:
    # (new_max - new_min) * (value - old_min) / (old_max - old_min) + new_min,
    # where new_min = -10 and new_max = 10
    # Find the min and max values for density in data
    density_max = plot_data.agg(max("centered_density"))
    density_max = density_max.select(
        col("max(centered_density)").alias("centered_density_max"))
    density_min = plot_data.agg(min("centered_density"))
    density_min = density_min.select(
        col("min(centered_density)").alias("centered_density_min"))
    plot_data = plot_data.crossJoin(density_max)
    plot_data = plot_data.crossJoin(density_min)

    # Calculated the scaled density
    plot_data = plot_data.withColumn(
        'scaled_density',
        ((10 - (-10)) *
        (plot_data.centered_density - plot_data.centered_density_min) /
        (plot_data.centered_density_max - plot_data.centered_density_min) + (-10)))

    print("Table with scaled density data for visualization")
    plot_data.show()


if __name__ == '__main__':
    main()
