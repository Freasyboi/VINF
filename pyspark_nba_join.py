from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_replace, trim, col

spark = SparkSession.builder.appName('CSVJoinExample').getOrCreate()

file1_path = 'nba_players.csv'
file2_path = 'nba_players_output/part-00000-58d56bbe-5876-407c-ad2f-daff90d1ef14-c000.csv'
output_path = 'joined_data_pyspark'

try:
    df1 = spark.read.csv(file1_path, header=True, inferSchema=False)
    df2 = spark.read.csv(file2_path, header=True, inferSchema=False)
except Exception as e:
    print(f"Error loading CSV files: {e}")
    spark.stop()
    exit()

join_columns = ["Name", "Birthday"]

joined_df = df1.join(
    df2,
    on=join_columns,
    how='left'
)

joined_df = joined_df.withColumn(
    "Height",
    trim(regexp_replace(col("Height"), '"', ''))
)

joined_df.coalesce(1).write.csv(output_path, header=True, mode='overwrite')

spark.stop()
