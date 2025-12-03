from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructType, StructField
from pyspark.sql.functions import regexp_replace, regexp_extract, col
import os
import re

# Set Python executable path for Spark (Windows fix)
os.environ['PYSPARK_PYTHON'] = 'python'
os.environ['PYSPARK_DRIVER_PYTHON'] = 'python'

# Initialize Spark Session
spark = (
    SparkSession.builder
    .appName("NBAPlayersExtractor")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.files.maxPartitionBytes", "67108864")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    .config("spark.default.parallelism", "50")
    .getOrCreate()
)

# File Paths
wiki_path = "enwiki-latest-pages-articles.xml.bz2"
output_path = "nba_players_output"

# ----------------- Regex Patterns (UPDATED) -----------------

BIRTHDATE_YMD_REGEX = re.compile(
    r"birth[_ ]date(?: and age)?\s*=\s*.*?\{\{[^|]*\|(?:[^|]*\|)*(\d{4})\|(\d{1,2})\|(\d{1,2})",
    re.IGNORECASE | re.DOTALL,
)

BIRTHDATE_TEXT_STRING_REGEX = re.compile(
    r"birth[_ ]date(?: and age)?\s*=\s*.*?\{\{[^|]*\|([^}]+)\}\}",
    re.IGNORECASE | re.DOTALL,
)

BIRTHDATE_DIRECT_TEXT_REGEX = re.compile(
    r"birth[_ ]date(?: and age)?\s*=\s*(?!\{\{)([^|\n]+)",
    re.IGNORECASE | re.DOTALL,
)

NATIONALITY_REGEX = re.compile(r"nationality\s*=\s*(.+)", re.IGNORECASE)

TEAM_REGEX = re.compile(
    r"\|\s*team\d+\s*=\s*((?:\[\[.*?\]\]|[^|\n])+)",
    re.IGNORECASE
)

CAREER_HISTORY_REGEX = re.compile(
    r"career_history\s*=\s*(.*?)\n\|", re.IGNORECASE | re.DOTALL
)

MONTH_MAP = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
    "jan": "01", "feb": "02", "mar": "03", "apr": "04", "jun": "06",
    "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11",
    "dec": "12",
}

# ----------------- Helper Functions -----------------

def clean_wiki_text(text):
    if text is None:
        return None

    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\{\{flag\|([^}]+)\}\}", r"\1", text)
    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)
    text = re.sub(r"\[\[|\]\]", "", text)
    text = re.sub(r"\}\}'*", "", text)

    return text.strip()


def clean_title(title):
    return re.sub(r"\s*\(.*?\)", "", title).strip()

# ----------------- Extraction Functions -----------------

def extract_birth_date(page_text):
    m = BIRTHDATE_YMD_REGEX.search(page_text)
    if m:
        y, mm, dd = m.groups()
        return f"{y}-{mm.zfill(2)}-{dd.zfill(2)}"

    text_matches = []
    t1 = BIRTHDATE_TEXT_STRING_REGEX.search(page_text)
    if t1:
        text_matches.append(t1.group(1))

    t2 = BIRTHDATE_DIRECT_TEXT_REGEX.search(page_text)
    if t2:
        text_matches.append(t2.group(1))

    for t in text_matches:
        cleaned = clean_wiki_text(t)
        cleaned = cleaned.split(" (")[0].strip()
        cleaned = cleaned.replace(".", "").strip()

        m1 = re.match(r"([A-Za-z]+)\s*(\d{1,2}),?\s*(\d{4})", cleaned, re.IGNORECASE)
        if m1:
            month, day, year = m1.groups()
            month_num = MONTH_MAP.get(month.lower())
            if month_num:
                return f"{year}-{month_num}-{day.zfill(2)}"

        m2 = re.match(r"(\d{1,2})\s*([A-Za-z]+)\s*(\d{4})", cleaned, re.IGNORECASE)
        if m2:
            day, month, year = m2.groups()
            month_num = MONTH_MAP.get(month.lower())
            if month_num:
                return f"{year}-{month_num}-{day.zfill(2)}"

    return None


def extract_nationality(page_text):
    m = NATIONALITY_REGEX.search(page_text)
    if not m:
        return None

    raw = m.group(1)

    raw = re.sub(r"", "", raw, flags=re.DOTALL)
    raw = re.sub(r"<ref[^>]*>.*?</ref>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"<ref[^>]*/>", "", raw)
    raw = re.sub(r"&lt;ref.*?&lt;/ref&gt;", "", raw, flags=re.DOTALL)

    raw = re.sub(r"\{\{.*?\}\}", "", raw, flags=re.DOTALL)
    raw = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", raw)

    raw = raw.replace("[[", "").replace("]]", "")
    raw = raw.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    raw = raw.strip()
    raw = raw.split("\n")[0]
    raw = raw.split("|")[0]

    raw = raw.replace("/", ",").replace(";", ",")
    raw = re.sub(r"\s*,\s*", ",", raw)

    parts = [p.strip() for p in raw.split(",") if p.strip()]

    final = []
    seen = set()
    for p in parts:
        if re.match(r"^[A-Za-z]+-[A-Za-z]+$", p):
            a, b = p.split("-")
            final.extend([a.strip(), b.strip()])
        else:
            final.append(p)

    unique = []
    for p in final:
        if p not in unique:
            unique.append(p)

    return ", ".join(unique) if unique else None


def extract_teams(page_text):
    teams = []

    for m in TEAM_REGEX.finditer(page_text):
        t = clean_wiki_text(m.group(1))
        t = re.sub(r"^→\s*", "", t)
        t = re.sub(r"\s*\([^)]*\)", "", t).strip()
        if t and t.lower() not in ("-", "none", "n/a"):
            teams.append(t)

    ch = CAREER_HISTORY_REGEX.search(page_text)
    if ch:
        block = ch.group(1)
        for line in block.split("\n"):
            if line.strip().startswith("*"):
                t = clean_wiki_text(line[1:].strip())
                t = re.sub(r"^→\s*", "", t)
                t = re.sub(r"\s*\([^)]*\)", "", t).strip()
                if t and t.lower() not in ("-", "none", "n/a"):
                    teams.append(t)

    final = []
    seen = set()
    for t in teams:
        if t not in seen:
            seen.add(t)
            final.append(t)

    return ", ".join(final) if final else None


# ----------------- RDD Processing -----------------

def group_pages(iterator):
    current = []
    in_page = False
    for line in iterator:
        if "<page>" in line:
            in_page = True
            current = [line]
        elif "</page>" in line:
            if in_page:
                current.append(line)
                yield current
                current = []
                in_page = False
        elif in_page:
            current.append(line)
    if current:
        yield current


def process_page(lines):
    text = "".join(lines)
    indicators = [
        "Infobox basketball biography",
        "Infobox NBA player",
        "Infobox basketball player",
        "{{Basketball name",
        "{{NBAplayer",
    ]
    if not any(i in text for i in indicators):
        return []

    title_match = re.search(r"<title>(.*?)</title>", text)
    if not title_match:
        return []

    title = clean_title(title_match.group(1))

    skip = ["Category:", "Template:", "Wikipedia:", "User:", "File:", "MediaWiki:"]
    if any(s in title for s in skip):
        return []

    return [(
        title,
        extract_birth_date(text),
        extract_nationality(text),
        extract_teams(text)
    )]


def extract_nba_players_robust():
    print("Starting robust NBA players extraction...")
    rdd = spark.sparkContext.textFile(wiki_path, minPartitions=100)
    print(f"Total partitions: {rdd.getNumPartitions()}")

    data = (
        rdd.mapPartitions(group_pages)
        .flatMap(process_page)
        .distinct()
    )

    schema = StructType([
        StructField("Name", StringType(), True),
        StructField("Birthday", StringType(), True),
        StructField("Nationality", StringType(), True),
        StructField("Teams", StringType(), True),
    ])

    df = spark.createDataFrame(data, schema)

    df = (
        df
        .withColumn("Teams", regexp_replace(col("Teams"), r" / ", ", "))
        .withColumn("Nationality", regexp_extract(col("Nationality"), r"^([A-Za-z ,]+)", 1))
    )

    return df


# ----------------- MAIN -----------------

if __name__ == "__main__":
    try:
        players_df = extract_nba_players_robust()

        count = players_df.count()
        print(f"Found {count} NBA players with full details")

        if count > 0:
            players_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)
            print(f"\nResults saved to: {output_path}")
        else:
            print("No players found using the robust extraction method.")

    except Exception as e:
        print(f"Main extraction failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        spark.stop()
        print("\nProcessing completed!")
