import lucene
from java.nio.file import Paths
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.queryparser.classic import QueryParser as QP
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.search import BooleanClause, BooleanQuery, TermQuery, BoostQuery, MatchAllDocsQuery
from org.apache.lucene.index import Term
from org.apache.lucene.document import LongPoint
from java.lang import Boolean, IllegalArgumentException
import re


class NBASearchEngine:
    def __init__(self, index_path="pylucene_indexer_data"):
        print("Opening NBA index:", index_path)

        directory = FSDirectory.open(Paths.get(index_path))
        try:
            reader = DirectoryReader.open(directory)
        except Exception as e:
            print(f"Error opening index: {e}")
            print("Please ensure the index path is correct and the index exists.")
            raise

        self.searcher = IndexSearcher(reader)
        self.analyzer = StandardAnalyzer()

        print(f"Loaded index with {reader.numDocs()} documents")

        self.text_fields = ["Name", "Summary", "College", "Number", "Wiki_Number", "Weight", "Wiki_Weight",
                            "Experience", "Nationality", "Teams"]
        self.exact_fields = [
            "Position",
            "Birthday",  # 'Birthday' is still kept here for string-based exact search
            "Birthtown",
            "Birthstate",
            "DraftYear",
            "DraftRound",
            "DraftNumber",
            "Height",
            "Wiki_Height"
        ]

        self.boolean_search_fields = ["Name", "Summary", "College", "Number", "Wiki_Number", "Weight", "Wiki_Weight",
                                      "Experience", "Nationality", "Teams", "Position", "Birthday",
                                      "Birthtown",
                                      "Birthstate", "DraftYear", "DraftRound", "DraftNumber", "Height", "Wiki_Height"]

        self.boosts = {
            "Name": 5,
            "Summary": 0.2,
            "Number": 3,
            "Wiki_Number": 3,
            "Position": 6,
            "College": 3,
            "Nationality": 3,
            "Teams": 3,
            "Weight": 2,
            "Wiki_Weight": 2,
            "Experience": 1,
            "Birthday": 1,
            "Birthtown": 2,
            "Birthstate": 2,
            "DraftYear": 1,
            "DraftRound": 1,
            "DraftNumber": 1,
            "Height": 3,
            "Wiki_Height": 3
        }

    def check_for_boolean_operators(self, query_text):
        """Checks if the query contains explicit Lucene boolean operators (AND, OR, NOT) and prefixes."""
        query_upper = " " + query_text.upper() + " "
        has_operator = (" AND " in query_upper) or (" OR " in query_upper) or (" NOT " in query_upper)
        has_prefix = (":" in query_text)

        return has_operator, has_prefix

    def build_text_query(self, query_text):
        tokens = query_text.split()
        builder = BooleanQuery.Builder()

        for field in self.text_fields:
            qp = QP(field, self.analyzer)
            boost = self.boosts.get(field, 1.0)

            try:
                # Use default operator (OR) within QP
                q = qp.parse(" ".join([f"{t}^{boost}" for t in tokens]))
                builder.add(q, BooleanClause.Occur.SHOULD)
            except Exception:
                continue

        return builder.build()

    def build_exact_match_queries(self, query_text):
        tokens = query_text.upper().split()
        builder = BooleanQuery.Builder()

        for field in self.exact_fields:
            boost = self.boosts.get(field, 1.0)

            for tok in tokens:
                tq = TermQuery(Term(field, tok))
                boosted_query = BoostQuery(tq, float(boost))
                builder.add(boosted_query, BooleanClause.Occur.SHOULD)

        return builder.build()

    def create_birthday_range_query(self, start_date_str, end_date_str):
        """
        Creates a LongPoint range query for the 'Birthday_Numeric' field.
        Dates must be in 'YYYY-MM-DD' format.
        """
        # Java's Long.MIN_VALUE and Long.MAX_VALUE for open ranges
        MIN_LONG = -9223372036854775808
        MAX_LONG = 9223372036854775807

        def safe_parse_date(date_str, default_value):
            """Converts 'yyyy-mm-dd' to YYYYMMDD integer, or returns default."""
            if not date_str:
                return default_value
            try:
                numeric_date_str = date_str.replace('-', '')
                # If the user provides a string that converts to an 8-digit number
                if re.match(r'^\d{8}$', numeric_date_str):
                    return int(numeric_date_str)
            except:
                pass
            return default_value

        lower_bound = safe_parse_date(start_date_str, MIN_LONG)
        upper_bound = safe_parse_date(end_date_str, MAX_LONG)

        # LongPoint.newRangeQuery expects the field name and bounds (inclusive)
        range_query = LongPoint.newRangeQuery(
            "Birthday_Numeric",
            lower_bound,
            upper_bound
        )
        return range_query

    def build_simple_boolean_query(self, query_text):
        """
        Manually constructs a BooleanQuery for simple A OP B queries
        (like 'Jordan AND 23' or 'Jordan NOT 23') by searching all fields.
        Anything more complex (A OP B OP C, A OP NOT B, NOT A) falls back
        to the single QueryParser, which is usually better for complex syntax.
        """
        # Look for a single, space-separated operator (AND, OR, NOT)
        query_parts = re.split(r'\s+(AND|OR|NOT)\s+', query_text.upper())
        final_builder = BooleanQuery.Builder()

        # FIX 1: If it's not exactly A OP B, or if it starts with NOT, use the QueryParser fallback.
        if len(query_parts) != 3 or query_text.upper().strip().startswith("NOT "):

            # Special case: Pure exclusion query like "NOT 6'3"
            if query_text.upper().strip().startswith("NOT "):
                # QueryParser won't search multiple fields, so we revert to the most robust
                # method for exclusion-only queries: MatchAllDocsQuery MUST NOT the term.

                # We expect "NOT term"
                term_to_exclude = query_text.strip()[3:].strip()
                if not term_to_exclude:
                    raise IllegalArgumentException("NOT query must specify a term.")

                print("  -> Manually building pure exclusion query (NOT term).")

                # 1. Add a positive clause (MatchAllDocsQuery)
                final_builder.add(MatchAllDocsQuery(), BooleanClause.Occur.SHOULD)

                # 2. Add the term to exclude across all fields
                exclusion_q = self._create_multi_field_query(term_to_exclude)
                if exclusion_q:
                    final_builder.add(exclusion_q, BooleanClause.Occur.MUST_NOT)
                    return final_builder.build()
                else:
                    raise IllegalArgumentException("Exclusion term failed to parse against all fields.")

            # Fallback for complex syntax (A OP B OP C, A OP NOT B, etc.)
            print("  -> Falling back to default QueryParser for complex boolean syntax.")
            return self.build_boolean_query(query_text, is_complex=False)

            # --- Proceed with manual A OP B parsing ---
        term1 = query_parts[0].strip()
        operator = query_parts[1].upper()
        term2 = query_parts[2].strip()

        # Map operator string to Lucene BooleanClause.Occur
        if operator == "AND":
            occur = BooleanClause.Occur.MUST
        elif operator == "OR":
            occur = BooleanClause.Occur.SHOULD
        elif operator == "NOT":
            occur = BooleanClause.Occur.MUST_NOT
        else:
            return self.build_boolean_query(query_text, is_complex=False)

        # Helper function definition moved outside for clean use below
        def _create_multi_field_query(term):
            q_builder = BooleanQuery.Builder()
            term_parsed_successfully = False
            for field in self.boolean_search_fields:
                qp = QP(field, self.analyzer)
                qp.setDefaultOperator(QP.Operator.OR)
                try:
                    q = qp.parse(term)
                    q_builder.add(q, BooleanClause.Occur.SHOULD)
                    term_parsed_successfully = True
                except Exception:
                    pass
            return q_builder.build() if term_parsed_successfully else None

        # 2. Build the query for Term 1
        q1 = _create_multi_field_query(term1)

        # 3. Build the query for Term 2
        q2 = _create_multi_field_query(term2)

        # 4. Add to final builder based on operator logic
        if q1:
            if operator == "AND":
                # For AND, the first term MUST match
                final_builder.add(q1, BooleanClause.Occur.MUST)
            elif operator == "NOT":
                # For NOT, the first term MUST be a SHOULD (optional)
                final_builder.add(q1, BooleanClause.Occur.SHOULD)
            else:  # OR case
                final_builder.add(q1, BooleanClause.Occur.SHOULD)

        if q2:
            final_builder.add(q2, occur)

        if (q1 is None and operator != "OR") or (q1 is None and q2 is None):
            raise IllegalArgumentException("Required query term failed to parse.")

        return final_builder.build()

    # Define the helper method outside of build_simple_boolean_query for cleaner usage
    def _create_multi_field_query(self, term):
        q_builder = BooleanQuery.Builder()
        term_parsed_successfully = False
        for field in self.boolean_search_fields:
            qp = QP(field, self.analyzer)
            qp.setDefaultOperator(QP.Operator.OR)
            try:
                q = qp.parse(term)
                q_builder.add(q, BooleanClause.Occur.SHOULD)
                term_parsed_successfully = True
            except Exception:
                pass
        return q_builder.build() if term_parsed_successfully else None

    def _default_search_logic(self, query_text, limit):
        builder = BooleanQuery.Builder()
        builder.add(self.build_text_query(query_text), BooleanClause.Occur.SHOULD)
        builder.add(self.build_exact_match_queries(query_text), BooleanClause.Occur.SHOULD)
        return builder.build()

    def build_boolean_query(self, query_text, is_complex=True):
        """
        Used for explicit field prefixed queries (is_complex=True)
        or for fall-back complex simple boolean queries (is_complex=False).
        """

        default_field = "Name"
        qp = QP(default_field, self.analyzer)
        qp.setDefaultOperator(QP.Operator.OR)

        try:
            final_query = qp.parse(query_text)
            return final_query
        except Exception as e:
            raise IllegalArgumentException(f"QueryParser failed on query '{query_text}': {e}")

    # --- MAIN SEARCH METHOD ---
    def search(self, query_text, limit=10):
        if not query_text.strip():
            return []

        # 1. Check for explicit BirthdayRange query
        birthday_range_match = re.search(r'BirthdayRange:(\d{4}-\d{2}-\d{2})\s+TO\s+(\d{4}-\d{2}-\d{2})', query_text,
                                         re.IGNORECASE)
        birthday_range_query = None

        if birthday_range_match:
            start_date, end_date = birthday_range_match.groups()
            try:
                birthday_range_query = self.create_birthday_range_query(start_date, end_date)
                print(f"**Birthday Range Query Detected**: {start_date} to {end_date}")

                # CRITICAL FIX (from previous iteration):
                # Remove the range query part AND any preceding boolean operator (AND/OR/NOT).
                query_text = re.sub(
                    r'(\s+(AND|OR|NOT)\s+)?BirthdayRange:\d{4}-\d{2}-\d{2}\s+TO\s+\d{4}-\d{2}-\d{2}',
                    ' ',  # Replace the whole match (operator + range) with a single space
                    query_text,
                    flags=re.IGNORECASE
                ).strip()

            except Exception as e:
                print(f"Error creating BirthdayRange query: {e}")

        # Determine the base query logic based on the *cleaned* query_text
        final_query = None
        has_operator, has_prefix = self.check_for_boolean_operators(query_text)

        if query_text:
            if has_prefix:
                # 1. COMPLEX BOOLEAN (e.g., Name:X AND Number:Y)
                print(f"**Complex Boolean Search Detected**: Parsing '{query_text}'.")
                try:
                    # Field-prefixed queries MUST use build_boolean_query
                    final_query = self.build_boolean_query(query_text, is_complex=True)
                except Exception as e:
                    print(f"Error parsing complex boolean query: {e}. Falling back to default relevance search.")
                    final_query = self._default_search_logic(query_text, limit)
            elif has_operator:
                # 2. SIMPLE BOOLEAN (e.g., Jordan AND 23, Jordan NOT 23, Jordan AND 23 NOT 6'3)
                print(f"**Simple Boolean Search Detected**: Parsing '{query_text}'.")
                try:
                    # Raw boolean queries MUST use build_simple_boolean_query for multi-field search
                    final_query = self.build_simple_boolean_query(query_text)
                except Exception as e:
                    print(f"Error parsing simple boolean query: {e}. Falling back to default relevance search.")
                    final_query = self._default_search_logic(query_text, limit)
            else:
                # 3. DEFAULT RELEVANCE SEARCH
                print("**Default Multi-Field Search**: Building custom query for relevance.")
                final_query = self._default_search_logic(query_text, limit)

        # 4. Combine with Birthday Range Query (if present)
        if birthday_range_query:
            if final_query is None:
                # If only the BirthdayRange was provided and query_text became empty
                final_query = birthday_range_query
            else:
                # Combine the relevance query and the range query with MUST
                builder = BooleanQuery.Builder()
                builder.add(final_query, BooleanClause.Occur.MUST)
                builder.add(birthday_range_query, BooleanClause.Occur.MUST)
                final_query = builder.build()

        # FINAL CHECK: If no query could be built
        if final_query is None:
            print("Error: Could not build a query.")
            return []

        results = self.searcher.search(final_query, limit)

        clean = []
        for sdoc in results.scoreDocs:
            doc = self.searcher.storedFields().document(sdoc.doc)

            clean.append({
                "score": float(f"{sdoc.score:.4f}"),
                "Name": doc.get("Name") or "Unknown",
                "Position": doc.get("Position") or "Unknown",
                "Height": doc.get("Height") or "Unknown",
                "Wiki_Height": doc.get("Wiki_Height") or "Unknown",
                "Weight": doc.get("Weight") or "Unknown",
                "Wiki_Weight": doc.get("Wiki_Weight") or "Unknown",
                "Nationality": doc.get("Nationality") or "Unknown",
            })

        return clean


def main():
    try:
        lucene.initVM()
    except Exception as e:
        print(f"Error initializing Lucene VM: {e}")
        print("Ensure 'lucene.jar' and required dependencies are in your classpath.")
        return

    print("Lucene VM Ready.")

    try:
        engine = NBASearchEngine()
    except Exception:
        return

    print("\n=== NBA Search CLI ===")
    print("Simple Boolean Search examples: 'Jordan AND 23', 'NOT 6'3', 'Jordan NOT 23'")
    print("Complex Field Search examples: 'Name:James AND Teams:Lakers'")
    print("NEW Range Query examples: 'BirthdayRange:1980-01-01 TO 1990-12-31'")
    print("Combined Query Example: 'Nationality:Serbian AND BirthdayRange:2000-01-01 TO 9999-12-31'")
    print("Type 'exit' to quit\n")

    output_fields = [
        "Score", "Name", "Position",
        "Height", "Wiki_Height",
        "Weight", "Wiki_Weight",
        "Nationality"
    ]
    header = " | ".join(output_fields)
    print("Expected Output Columns: " + header)
    print("-" * (len(header) + len(output_fields) * 3))

    while True:
        query = input("Search> ").strip()
        if query.lower() == "exit":
            break

        results = engine.search(query)
        print("\n--- Results ---\n")

        if not results:
            print("No results.\n")
            continue

        for r in results:
            print_str = (
                f"Score: {r['score']} | "
                f"Name: {r['Name']} | "
                f"Position: {r['Position']} | "
                f"Height: {r['Height']} | "
                f"Wiki_Height: {r['Wiki_Height']} | "
                f"Weight: {r['Weight']} | "
                f"Wiki_Weight: {r['Wiki_Weight']} | "
                f"Nationality: {r['Nationality']} | "
            )
            print(print_str)
        print("\n")


if __name__ == "__main__":
    main()
