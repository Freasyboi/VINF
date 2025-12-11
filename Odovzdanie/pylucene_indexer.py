import lucene
import os
import csv
import re
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType, StringField, TextField, StoredField, LongPoint # <-- ADD LongPoint
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, IndexOptions
from org.apache.lucene.store import FSDirectory


def init_lucene_vm():
    # Initialize the Lucene VM once
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    print("Lucene VM initialized")


class NBAPlayersIndexer:
    def __init__(self, index_dir="pylucene_indexer_data"):
        self.index_dir = index_dir
        self.analyzer = None

    def create_index(self, csv_file_path):
        self.analyzer = StandardAnalyzer()
        print(f"Creating Lucene index from {csv_file_path}...")

        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)

        directory = FSDirectory.open(Paths.get(self.index_dir))
        config = IndexWriterConfig(self.analyzer)
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        writer = IndexWriter(directory, config)

        try:
            if not os.path.exists(csv_file_path):
                raise FileNotFoundError(f"CSV file not found: {csv_file_path}")

            total_docs = 0
            print(f"Processing single file: {csv_file_path}...")

            with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)

                for row_num, row in enumerate(reader, 1):
                    if not row.get('Name'):
                        continue

                    doc = self.create_document_from_row(row)
                    writer.addDocument(doc)
                    total_docs += 1

                    if total_docs % 1000 == 0:
                        print(f"Indexed {total_docs} total documents...")

            print(f"Successfully indexed {total_docs} total documents")

        finally:
            writer.commit()
            writer.close()
            directory.close()

    def create_document_from_row(self, row):
        doc = Document()

        # Define Field Types
        # 1. Tokenized Field Type (for full-text search)
        text_field_type = FieldType()
        text_field_type.setStored(True)
        text_field_type.setTokenized(True)
        text_field_type.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS)

        # 2. Non-Tokenized Field Type (for exact match filtering)
        string_field_type = FieldType()
        string_field_type.setStored(True)
        string_field_type.setTokenized(False)
        string_field_type.setIndexOptions(IndexOptions.DOCS)

        # --- Indexing Fields ---

        # 1. Tokenized Fields (Full-text search)

        if row.get('Name'):
            doc.add(Field("Name", row['Name'], text_field_type))

        if row.get('Summary'):
            doc.add(Field("Summary", row['Summary'], text_field_type))

        if row.get('College'):
            doc.add(Field("College", row['College'], text_field_type))

        if row.get('Number'):
            doc.add(Field("Number", row['Number'], text_field_type))

        if row.get('Wiki_Number'):
            doc.add(Field("Wiki_Number", row['Wiki_Number'], text_field_type))

        if row.get('Weight'):
            doc.add(Field("Weight", row['Weight'], text_field_type))

        if row.get('Wiki_Weight'):
            doc.add(Field("Wiki_Weight", row['Wiki_Weight'], text_field_type))

        if row.get('Experience'):
            doc.add(Field("Experience", row['Experience'], text_field_type))

        if row.get('Nationality'):
            doc.add(Field("Nationality", row['Nationality'], text_field_type))

        if row.get('Teams'):
            doc.add(Field("Teams", row['Teams'], text_field_type))

        if row.get('MainText'):
            doc.add(Field("MainText", row['MainText'], text_field_type))

        # 2. Non-Tokenized/Exact Fields

        if row.get('Position'):
            doc.add(Field("Position", row['Position'], string_field_type))

        # --- Birthday Indexing for Range Queries ---
        birthday_str = row.get('Birthday')
        if birthday_str:
            # 1. StringField for exact match (e.g., Position:...)
            doc.add(Field("Birthday", birthday_str, string_field_type))

            # 2. LongPoint for efficient range searching (e.g., BirthdayRange:...)
            try:
                # Convert 'yyyy-mm-dd' to an integer YYYYMMDD
                numeric_date_str = birthday_str.replace('-', '')
                if re.match(r'^\d{8}$', numeric_date_str):
                    numeric_date = int(numeric_date_str)
                    # Index the value as a LongPoint
                    doc.add(LongPoint("Birthday_Numeric", [numeric_date]))
            except ValueError:
                # Skip indexing LongPoint if conversion fails
                pass
        # ------------------------------------------

        if row.get('Birthtown'):
            doc.add(Field("Birthtown", row['Birthtown'], string_field_type))

        if row.get('Birthstate'):
            doc.add(Field("Birthstate", row['Birthstate'], string_field_type))

        if row.get('DraftYear'):
            doc.add(Field("DraftYear", row['DraftYear'], string_field_type))

        if row.get('DraftRound'):
            doc.add(Field("DraftRound", row['DraftRound'], string_field_type))

        if row.get('DraftNumber'):
            doc.add(Field("DraftNumber", row['DraftNumber'], string_field_type))

        if row.get('Height'):
            doc.add(Field("Height", row['Height'], string_field_type))

        if row.get('Wiki_Height'):
            doc.add(Field("Wiki_Height", row['Wiki_Height'], string_field_type))

        return doc


def main():
    print("Initializing Lucene VM...")
    init_lucene_vm()

    # NOTE: Update this path to your actual CSV file
    csv_file_path = "joined_data_pyspark/part-00000-9e32e72a-ac27-490e-a621-c64a67950611-c000.csv"

    indexer = NBAPlayersIndexer()
    indexer.create_index(csv_file_path)
    print("Index created successfully!")


if __name__ == "__main__":
    main()
    print("\nNBAPlayersIndexer class with LongPoint for Birthday implemented.")