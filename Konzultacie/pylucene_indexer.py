import lucene
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, StringField, TextField, IntPoint, NumericDocValuesField, StoredField
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.store import FSDirectory
import os
import csv
from datetime import datetime

class Indexer:

    def __init__(self, index_dir="index_pylucene"):
        self.index_dir = index_dir

        if not lucene.getVMEnv():
            lucene.initVM()

        self.analyzer = StandardAnalyzer()

        if not os.path.exists(index_dir):
            os.makedirs(index_dir)

    def read_csv(self, path):
        try:
            with open(path, encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
                print(f"Loaded {len(rows)} rows from {path}")
                return rows
        except Exception as e:
            print("Error reading CSV:", e)
            return []

    # ------------------------------
    # Helper to safely handle numeric fields
    # ------------------------------
    def fix_numeric(self, value):
        if value is None:
            return None
        s = str(value).strip()
        if s == "" or s in ["Undrafted", "NA", "None", "-"]:
            return None
        try:
            return int(s)
        except:
            return None

    # ------------------------------
    # Main indexing function
    # ------------------------------
    def create_index(self, csv_path):
        data = self.read_csv(csv_path)
        if not data:
            print("No data found.")
            return 0

        store = FSDirectory.open(Paths.get(self.index_dir))
        writer = IndexWriter(store, IndexWriterConfig(self.analyzer))

        count = 0

        for idx, row in enumerate(data):
            try:
                doc = Document()
                doc.add(StringField("doc_id", str(idx), Field.Store.YES))

                # --- Text fields ---
                for field in ["Name", "Position", "Birthtown", "Birthstate",
                              "College", "Summary", "Teams"]:
                    value = row.get(field, "") or ""
                    doc.add(TextField(field.lower(), value, Field.Store.YES))

                # --- String fields (exact match) ---
                for field in ["Birthday", "Number", "Height", "Weight",
                              "Experience", "Nationality"]:
                    value = row.get(field, "") or ""
                    doc.add(StringField(field.lower(), value, Field.Store.YES))

                # --- Numeric fields (safe) ---
                draft_year = self.fix_numeric(row.get("DraftYear"))
                if draft_year is not None:
                    doc.add(IntPoint("draft_year", draft_year))
                    doc.add(NumericDocValuesField("draft_year_sort", draft_year))
                    doc.add(StoredField("draft_year", draft_year))

                draft_round = self.fix_numeric(row.get("DraftRound"))
                if draft_round is not None:
                    doc.add(IntPoint("draft_round", draft_round))
                    doc.add(NumericDocValuesField("draft_round_sort", draft_round))
                    doc.add(StoredField("draft_round", draft_round))

                draft_number = self.fix_numeric(row.get("DraftNumber"))
                if draft_number is not None:
                    doc.add(IntPoint("draft_number", draft_number))
                    doc.add(NumericDocValuesField("draft_number_sort", draft_number))
                    doc.add(StoredField("draft_number", draft_number))

                # --- Combined search field ---
                combined = " ".join([
                    row.get("Name", ""),
                    row.get("Position", ""),
                    row.get("College", ""),
                    row.get("Summary", ""),
                    row.get("Teams", "")
                ])
                doc.add(TextField("full_search", combined, Field.Store.YES))

                # Timestamp
                doc.add(StringField("indexed_at",
                                    datetime.now().isoformat(),
                                    Field.Store.YES))

                writer.addDocument(doc)
                count += 1

            except Exception as e:
                print(f"Error indexing row {idx}: {e}")
                continue

        writer.commit()
        writer.close()
        print(f"Indexing complete. Total indexed: {count}")
        return count


if __name__ == "__main__":
    indexer = Indexer()

    csv_file = "joined_data_pyspark/part-00000-7a6b4e76-d04e-4eab-aa23-04eda1a948af-c000.csv"  # take first CSV found
    print(f"Indexing CSV file: {csv_file}")

    indexer.create_index(csv_file)
