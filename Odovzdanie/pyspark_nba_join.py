from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_replace, trim, col, when, lower, coalesce, concat, lit

spark = SparkSession.builder.appName('CSVJoinExample').getOrCreate()

# --- Configuration ---
file1_path = 'nba_players.csv'
file2_path = 'nba_players_output/part-00000-eca15f3d-f189-465e-a286-d9e4c7466ba6-c000.csv'
output_path = 'joined_data_pyspark'
join_columns = ["Name", "Birthday"]

try:
    # 1. Load DataFrames
    df1 = spark.read.csv(file1_path, header=True, inferSchema=False)
    # Rename columns in df2 that conflict with df1,
    # and those we want to keep separate.
    df2 = spark.read.csv(file2_path, header=True, inferSchema=False) \
               .withColumnRenamed("Position", "Position_wiki") \
               .withColumnRenamed("DraftYear", "DraftYear_wiki") \
               .withColumnRenamed("DraftRound", "DraftRound_wiki") \
               .withColumnRenamed("DraftNumber", "DraftNumber_wiki") \
               .withColumnRenamed("Height", "Wiki_Height") \
               .withColumnRenamed("Weight", "Wiki_Weight") \
               .withColumnRenamed("Number", "Wiki_Number")

except Exception as e:
    print(f"Error loading CSV files: {e}")
    spark.stop()
    exit()

# 2. Perform Full Outer Join
# We will use the original column names for df1 and the renamed columns for df2
# for the fields we want to apply specific logic to.
# Note: For columns like 'Nationality', 'Teams', 'MainText' from df2,
# they will be included as new columns in the final DataFrame.

joined_df = df1.join(
    df2,
    on=join_columns,
    how='full_outer'
)

# 3. Apply Conflict Resolution Logic

# A. For Position, DraftYear, DraftRound, DraftNumber:
# "if file1 has data, use file1's data, else use file2's data"
# This is accomplished using the `coalesce` function.

# Position
joined_df = joined_df.withColumn(
    "Position",
    coalesce(col("Position"), col("Position_wiki"))
)

# DraftYear
joined_df = joined_df.withColumn(
    "DraftYear",
    coalesce(col("DraftYear"), col("DraftYear_wiki"))
)

# DraftRound
joined_df = joined_df.withColumn(
    "DraftRound",
    coalesce(col("DraftRound"), col("DraftRound_wiki"))
)


# DraftNumber
joined_df = joined_df.withColumn(
    "DraftNumber",
    coalesce(col("DraftNumber"), col("DraftNumber_wiki"))
)


# Clean the 'Number' column (from df1)
joined_df = joined_df.withColumn(
    "Number",
    trim(regexp_replace(col("Number"), '#', ''))
)

# Clean the 'Height' column (from df1)
joined_df = joined_df.withColumn(
    "Height",
    trim(regexp_replace(col("Height"), '"', ''))
)

# Clean the 'Birthtown' column (from df1)
joined_df = joined_df.withColumn(
    "Birthtown",
    when(lower(col("Birthtown")).contains("rnd"), None).otherwise(col("Birthtown"))
)

# Clean the merged 'DraftYear' column
joined_df = joined_df.withColumn(
    "Draftyear",
    trim(regexp_replace(col("Draftyear"), 'Undrafted', ''))
)

# Clean the 'Birthstate' column (from df1)
joined_df = joined_df.withColumn(
    "Birthstate",
    when(lower(col("Birthstate")).contains("draft"), None).otherwise(col("Birthstate"))
)

joined_df = joined_df.withColumn(
    "Wiki_Weight",
    regexp_replace(col("Wiki_Weight"), "[^0-9]", "")
)

joined_df = joined_df.withColumn(
    "Wiki_Weight",
    concat(col("Wiki_Weight"), lit(" lbs"))
)

joined_df = joined_df.withColumn(
    "MainText",
    regexp_replace(col("MainText"), "[#&|:/=;\\-\\*]", "")
)

columns_to_keep = [c for c in joined_df.columns if not c.endswith("_wiki")]

final_df = joined_df.select(columns_to_keep)

# 6. Write Output
final_df.coalesce(1).write.csv(output_path, header=True, mode='overwrite')

spark.stop()

print(f"Joined data written to {output_path}")
