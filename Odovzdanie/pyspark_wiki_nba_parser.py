from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructType, StructField
from pyspark.sql.functions import regexp_replace, regexp_extract, col, trim, when, lower
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

# ----------------- Regex Patterns -----------------

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
TEAM_REGEX = re.compile(r"\|\s*team\d+\s*=\s*((?:\[\[.*?\]\]|[^|\n])+)", re.IGNORECASE)
CAREER_HISTORY_REGEX = re.compile(r"career_history\s*=\s*(.*?)\n\|", re.IGNORECASE | re.DOTALL)

# New fields
POSITION_REGEX = re.compile(r"\|\s*position\s*=\s*(.+)", re.IGNORECASE)
HEIGHT_REGEX = re.compile(r"\|\s*height\s*=\s*(.*?)\n\|", re.IGNORECASE | re.DOTALL)
HEIGHT_FT_IN_REGEX = re.compile(r"(\d+)\s*(?:ft|')\s*(\d+)?\s*(?:in|\"|'')?", re.IGNORECASE)
WEIGHT_REGEX = re.compile(r"\|\s*weight\s*=\s*([^\n|]+)", re.IGNORECASE)
DRAFT_REGEX = re.compile(r"\|\s*draft\s*=\s*(.*?)\n\|", re.IGNORECASE | re.DOTALL)
JERSEY_REGEX = re.compile(r"\|\s*number\s*=\s*(.+)", re.IGNORECASE)

# Regex to find the main text (first sentence or block after infobox/templates)
# It searches for bolded text (usually the player's name in the intro) and captures
# everything up to the first period, or a section heading.
MAIN_TEXT_REGEX = re.compile(
    r"'''(?!\s*ratko varda\s*'''\s*\()[^']+'{3}(.*?)(?:\.\s*[A-Z]|\n\n|==)",
    re.IGNORECASE | re.DOTALL,
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
    # Remove HTML/XML tags
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&lt;.*?&gt;", "", text)
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
    raw = re.sub(r"<ref[^>]*>.*?</ref>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"<ref[^>]*/>", "", raw)
    raw = re.sub(r"&lt;ref.*?&lt;/ref&gt;", "", raw, flags=re.DOTALL)
    raw = re.sub(r"\{\{.*?\}\}", "", raw, flags=re.DOTALL)
    raw = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", raw)
    raw = raw.replace("[[", "").replace("]]", "")
    raw = raw.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    raw = raw.strip().split("\n")[0].split("|")[0]
    raw = raw.replace("/", ",").replace(";", ",")
    raw = re.sub(r"\s*,\s*", ",", raw)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    final = []
    for p in parts:
        if re.match(r"^[A-Za-z]+-[A-Za-z]+$", p):
            final.extend(p.split("-"))
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


# ----------------- New Field Extractors -----------------

# Position
POSITIONS_LIST = [
    "Point guard", "Shooting guard", "Guard", "Small forward", "Forward",
    "Power forward", "Center", "Forward–center", "Guard–forward", "Forward–guard",
    "Power_forward", "Small_forward", "Point_guard", "Shooting_guard"
]


def extract_position(text):
    fields = ["position", "career_position"]
    for field in fields:
        # Extract value up to newline
        m = re.search(rf"\|\s*{field}\s*=\s*(.+)", text, re.IGNORECASE)
        if m:
            val = m.group(1).split("\n")[0]  # Stop at newline
            val = clean_wiki_text(val)
            # Take last part after pipe if exists
            if "|" in val:
                val = val.split("|")[-1].strip()
            # Keep only letters, spaces, dash
            val = re.sub(r"[^A-Za-z\- ]", "", val)
            val = val.strip()
            # Compare to known positions list (case-insensitive)
            for pos in POSITIONS_LIST:
                if pos.lower() in val.lower():
                    # Normalize combined values like Guard–forward -> Guard
                    primary = pos.split("–")[0].split("-")[0].strip().lower()

                    # Canonical mapping
                    mapping = {
                        "point guard": "POINTGUARD",
                        "shooting guard": "SHOOTINGGUARD",
                        "guard": "POINTGUARD",  # Can be ambiguous, but map to PG for simplicity or adjust
                        "small forward": "SMALLFORWARD",
                        "power forward": "POWERFORWARD",
                        "forward": "POWERFORWARD",  # Can be ambiguous, but map to PF for simplicity or adjust
                        "center": "CENTER",
                    }

                    # If underscores are used in POSITIONS_LIST, normalize them
                    primary = primary.replace("_", " ")

                    # Return mapped value
                    return mapping.get(primary, None)

    return None


# Height
def extract_height(text):
    # First try combined height
    m = re.search(r"\|\s*height\s*=\s*([^\n|]+)", text, re.IGNORECASE)
    if m:
        raw = clean_wiki_text(m.group(1))
        m2 = re.search(r"(\d+)\s*(?:ft|')\s*(\d+)?", raw)
        if m2:
            ft = m2.group(1)
            inch = m2.group(2) if m2.group(2) else "0"
            return f"{ft}'{inch}"
    # Try separate ft/in
    ft = re.search(r"\|\s*height_ft\s*=\s*(\d+)", text, re.IGNORECASE)
    inch = re.search(r"\|\s*height_in\s*=\s*(\d+)", text, re.IGNORECASE)
    if ft:
        ft_val = ft.group(1)
        inch_val = inch.group(1) if inch else "0"
        return f"{ft_val}'{inch_val}"
    return None


# Weight
def extract_weight(text):
    m = re.search(r"\|\s*weight\s*=\s*([^\n|]+)", text, re.IGNORECASE)
    if m:
        val = clean_wiki_text(m.group(1)).split("(")[0].strip()
        val = val.replace("lb", "").replace("lbs", "").replace("kg", "").strip()
        if val:
            return val
    # Try weight_lb
    m = re.search(r"\|\s*weight_lb\s*=\s*(\d+)", text, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


# Draft
def extract_draft(text):
    # Try combined draft field first
    m = re.search(r"\|\s*draft\s*=\s*(.*?)\n\|", text, re.IGNORECASE | re.DOTALL)
    if m:
        raw = clean_wiki_text(m.group(1))
        temp = re.findall(r"\{\{.*?draft\|(\d{4})\|(\d+)\|(\d+)", raw)
        if temp:
            return temp[0]
        # fallback to regex inside draft text
        year = re.search(r"(19|20)\d{2}", raw)
        round_ = re.search(r"round\s*(\d+)", raw, re.IGNORECASE)
        pick = re.search(r"(?:pick|overall)\s*(\d+)", raw, re.IGNORECASE)
        return (
            year.group(0) if year else None,
            round_.group(1) if round_ else None,
            pick.group(1) if pick else None
        )
    # Try separate draft_year, draft_round, draft_pick
    year = re.search(r"\|\s*draft_year\s*=\s*(\d{4})", text, re.IGNORECASE)
    round_ = re.search(r"\|\s*draft_round\s*=\s*(\d+)", text, re.IGNORECASE)
    pick = re.search(r"\|\s*draft_pick\s*=\s*(\d+)", text, re.IGNORECASE)
    return (
        year.group(1) if year else None,
        round_.group(1) if round_ else None,
        pick.group(1) if pick else None
    )


# Number
def extract_jersey_number(text):
    fields = ["number", "career_number"]
    for field in fields:
        m = re.search(rf"\|\s*{field}\s*=\s*(.+)", text, re.IGNORECASE)
        if m:
            val = clean_wiki_text(m.group(1))
            val = val.split("|")[0].strip()

            if not val:
                return None

            # Extract only numeric jersey numbers
            nums = re.findall(r"\d+", val)
            if not nums:
                return None

            # Reject garbage such as 1491581744616599554 or any number > 2 digits
            if any(len(n) > 2 for n in nums):
                return None

            # Format as: "6, 7, 12"
            cleaned = ", ".join(nums)

            # Validate final format
            if re.fullmatch(r"\d{1,2}(?:, \d{1,2})*", cleaned):
                return cleaned

            return None

    return None


# Main Text Extractor (New Function)
def extract_main_text(page_text):
    """
    Extracts the first sentence or introductory paragraph from the main page content.
    """
    m = MAIN_TEXT_REGEX.search(page_text)
    if not m:
        # Fallback: Look for the first block of text after an infobox or template
        match_block = re.search(r"\}\}\s*'''[^']+'{3}(.*?)(?:\n\n|==)", page_text, re.DOTALL | re.IGNORECASE)
        if match_block:
            raw_text = match_block.group(1)
            # Find the first full sentence (ending with .!?)
            sentence_match = re.search(r"(.*?)[.?!](?=\s*[A-Z]|\s*<|$)|\n", raw_text, re.DOTALL)
            if sentence_match:
                raw = sentence_match.group(1)
            else:
                raw = raw_text.split("\n")[0]  # Just take the first line of the block
        else:
            return None
    else:
        raw = m.group(1)

    if not raw:
        return None

    # Apply general cleanup to remove wiki/html elements
    cleaned = clean_wiki_text(raw)

    # Clean up excess spaces and leading punctuation (e.g., from template removal)
    cleaned = re.sub(r"^\W+", "", cleaned).strip()

    # Ensure it ends with a punctuation mark for a complete sentence feel
    if cleaned and not cleaned.endswith(('.', '!', '?')):
        # Try to find the nearest sentence boundary within the extracted block
        final_sentence_match = re.search(r"(.*?)[.?!]", cleaned, re.DOTALL)
        if final_sentence_match:
            cleaned = final_sentence_match.group(0).strip()
        else:
            # If no sentence break, just trim up to 200 chars for safety
            cleaned = cleaned[:200].strip()

    return cleaned if cleaned else None


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

    main_text = extract_main_text(text)  # <--- New extraction call

    return [(
        title,
        extract_birth_date(text),
        extract_nationality(text),
        extract_teams(text),
        extract_position(text),
        extract_height(text),
        extract_weight(text),
        *extract_draft(text),
        extract_jersey_number(text),
        main_text  # <--- New field added here
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

    # Schema Updated with MainText
    schema = StructType([
        StructField("Name", StringType(), True),
        StructField("Birthday", StringType(), True),
        StructField("Nationality", StringType(), True),
        StructField("Teams", StringType(), True),
        StructField("Position", StringType(), True),
        StructField("Height", StringType(), True),
        StructField("Weight", StringType(), True),
        StructField("DraftYear", StringType(), True),
        StructField("DraftRound", StringType(), True),
        StructField("DraftNumber", StringType(), True),
        StructField("Number", StringType(), True),
        StructField("MainText", StringType(), True),  # <--- New Schema Field
    ])

    df = spark.createDataFrame(data, schema)

    df = (
        df
        .withColumn("Teams", regexp_replace(col("Teams"), r" / ", ", "))
        .withColumn("Nationality", regexp_extract(col("Nationality"), r"^([A-Za-z ,]+)", 1))
    )

    df = df.withColumn("Teams", when(lower(col("Teams")).contains("&"), None).otherwise(col("Teams")))
    df = df.withColumn("Teams", when(lower(col("Teams")).contains("https"), None).otherwise(col("Teams")))
    df = df.withColumn("Teams", trim(regexp_replace(col("Teams"), r"[\{'].*$", "")))
    df = df.withColumn("Nationality", trim(regexp_replace(col("Nationality"), r"[\{'].*$", "")))
    df = df.filter(col("Birthday").rlike(r'^[12][0-9]{3}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])$'))
    df = df.withColumn("Nationality", regexp_replace(col("Nationality"), r"(?i)\b(and|is)\b.*$", ""))
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