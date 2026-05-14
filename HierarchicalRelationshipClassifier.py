import anthropic
import json
import os
import snowflake.connector
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = (
    "You are a Hierarchical Relationship Classifier for Company Profiles. "
    "Given two company domains or names, determine their business relationship and hierarchical structure. "
    "If additional company features are provided (revenue, employees, founded year, description, headquarters, industry, etc.), "
    "use these features to make more accurate relationship classifications. "
    "Classify the relationship type (Parent-Subsidiary, Sister Company, Joint Venture, Competitors, etc.), "
    "hierarchy level (Ultimate Parent, Subsidiary, Branch, etc.), "
    "ownership percentage or control level, "
    "and relationship strength (Direct, Indirect, Affiliate, Strategic Partnership). "
    "Provide a confidence score for your classification. "
    "Use your knowledge or public data of corporate structures and business relationships to make accurate assessments. "
    "Consider company size, industry, founding dates, and other contextual information when determining relationships."
)


class RelationshipInfo(BaseModel):
    relationshipType: str = Field(description="Type of relationship: Parent-Subsidiary, Sister Company, Joint Venture, Franchise, etc.")
    hierarchyLevel: str = Field(description="Position in hierarchy: Ultimate Parent, Intermediate Parent, Subsidiary, Branch, Affiliate, etc.")
    ownershipPercentage: str = Field(description="Estimated ownership percentage or control level")
    relationshipStrength: str = Field(description="Strength of relationship: Direct, Indirect, Affiliate, Strategic Partnership")
    confidenceScore: int = Field(description="Confidence score 1-100 for the relationship classification")


def connect_to_snowflake():
    try:
        private_key_str = os.getenv("SNOWFLAKE_PRIVATE_KEY")
        if "\\n" in private_key_str:
            private_key_str = private_key_str.replace("\\n", "\n")

        private_key = serialization.load_pem_private_key(
            private_key_str.encode(),
            password=None,
            backend=default_backend(),
        )
        pkb = private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        conn = snowflake.connector.connect(
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            user=os.getenv("SNOWFLAKE_USER"),
            private_key=pkb,
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
            database=os.getenv("SNOWFLAKE_DATABASE"),
            schema=os.getenv("SNOWFLAKE_SCHEMA"),
            role=os.getenv("SNOWFLAKE_ROLE"),
        )
        print("✅ Connected to Snowflake successfully!\n")
        return conn
    except Exception as e:
        print(f"❌ Error connecting to Snowflake: {e}\n")
        return None


def fetch_company_features_from_snowflake(conn, domain: str) -> dict:
    try:
        cursor = conn.cursor()
        query = """
            SELECT name, revenue, employees, website, zi_dozi_industries, naics, website_status
            FROM "PROD_MASTER_DB"."COMPANY_ETL"."COMPANY"
            WHERE non_publishable=false
            AND website_status='VALID'
            AND website = %s
            LIMIT 1
        """
        cursor.execute(query, (domain,))
        row = cursor.fetchone()
        cursor.close()
        if row:
            return {
                "name": row[0],
                "revenue": row[1] or "N/A",
                "employees": row[2] or "N/A",
                "website": row[3],
                "industry": row[4] or "N/A",
                "naics": row[5] or "N/A",
                "website_status": row[6],
            }
        print(f"  ⚠️ No features found in Snowflake for: {domain}")
        return {}
    except Exception as e:
        print(f"  ❌ Error fetching features for {domain}: {e}")
        return {}


def load_company_pairs_from_json(file_path: str) -> list[tuple]:
    try:
        with open(file_path) as f:
            data = json.load(f)

        if not isinstance(data, list):
            print("❌ JSON must be a list")
            return []

        pairs = []
        for item in data:
            if isinstance(item, dict) and "company1" in item:
                pairs.append((item["company1"], item["company2"], item.get("features1", {}), item.get("features2", {})))
            elif isinstance(item, list) and len(item) >= 2:
                pairs.append((item[0], item[1], {}, {}))
            else:
                print(f"⚠️ Skipping unrecognized item: {item}")

        print(f"📊 Loaded {len(pairs)} company pairs from {file_path}\n")
        return pairs
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}\n")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}\n")
        return []


def format_features(features: dict) -> str:
    if not features:
        return "No additional features provided."
    return "\n".join(
        f"  - {k.replace('_', ' ').replace('-', ' ').title()}: {v}"
        for k, v in features.items()
    )


def classify_relationship(company1: str, company2: str, features1: dict = None, features2: dict = None) -> RelationshipInfo | str:
    prompt_parts = [f"Analyze the relationship between '{company1}' and '{company2}'"]
    if features1:
        prompt_parts.append(f"\n\nCompany 1 ({company1}) Features:\n{format_features(features1)}")
    if features2:
        prompt_parts.append(f"\n\nCompany 2 ({company2}) Features:\n{format_features(features2)}")

    try:
        response = client.messages.parse(
            model="claude-opus-4-7",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": "".join(prompt_parts)}],
            output_format=RelationshipInfo,
        )
        return response.parsed_output
    except Exception as e:
        return str(e)


# Main execution
print("🤖 Hierarchical Relationship Classifier\n")
print("Choose mode:")
print("1. Read from JSON file")
print("2. Manual input (enter two companies)")

mode = input("\nEnter mode (1 or 2): ").strip()

if mode == "1":
    file_path = input("Enter JSON file path (default: company_pairs.json): ").strip() or "company_pairs.json"
    use_snowflake = input("Fetch company features from Snowflake? (y/n): ").strip().lower() == "y"

    snowflake_conn = None
    if use_snowflake:
        snowflake_conn = connect_to_snowflake()
        if not snowflake_conn:
            print("❌ Failed to connect to Snowflake. Proceeding without feature enrichment.")
            use_snowflake = False

    company_pairs = load_company_pairs_from_json(file_path)
    if company_pairs:
        print(f"\n{'='*80}")
        print("Starting relationship classification...\n")

        results = []
        for idx, (company1, company2, features1, features2) in enumerate(company_pairs, 1):
            print(f"[{idx}/{len(company_pairs)}] Analyzing: {company1} <-> {company2}")

            if use_snowflake and snowflake_conn:
                if not features1:
                    print(f"  Fetching features for {company1}...")
                    features1 = fetch_company_features_from_snowflake(snowflake_conn, company1)
                if not features2:
                    print(f"  Fetching features for {company2}...")
                    features2 = fetch_company_features_from_snowflake(snowflake_conn, company2)

            classification = classify_relationship(company1, company2, features1, features2)
            results.append({"company1": company1, "company2": company2, "relationship": classification})
            print(f"Result: {classification}\n")
            print("-" * 80)

        if snowflake_conn:
            snowflake_conn.close()
            print("✅ Closed Snowflake connection\n")

        print(f"\n✅ Completed classification of {len(results)} relationships!")

        if input("\nSave results to JSON? (y/n): ").strip().lower() == "y":
            output_file = input("Enter output filename (default: relationship_results.json): ").strip() or "relationship_results.json"
            serializable = [
                {
                    "company1": r["company1"],
                    "company2": r["company2"],
                    "relationship": {
                        "type": r["relationship"].relationshipType,
                        "hierarchy": r["relationship"].hierarchyLevel,
                        "ownership": r["relationship"].ownershipPercentage,
                        "strength": r["relationship"].relationshipStrength,
                        "confidence": r["relationship"].confidenceScore,
                    },
                }
                for r in results
            ]
            with open(output_file, "w") as f:
                json.dump(serializable, f, indent=2)
            print(f"✅ Results saved to {output_file}")

elif mode == "2":
    print("\n🤖 Relationship Classifier is ready! Type 'bye' or 'exit' to stop.\n")
    use_snowflake = input("Fetch company features from Snowflake? (y/n): ").strip().lower() == "y"

    snowflake_conn = None
    if use_snowflake:
        snowflake_conn = connect_to_snowflake()
        if not snowflake_conn:
            print("❌ Failed to connect to Snowflake. Proceeding without feature enrichment.")
            use_snowflake = False

    while True:
        company1 = input("\nCompany 1 domain (or 'bye'/'exit' to stop): ").strip()
        if company1.lower() in ["bye", "exit"]:
            print("Classifier: Goodbye 👋")
            break

        company2 = input("Company 2 domain: ").strip()
        features1, features2 = {}, {}

        if use_snowflake and snowflake_conn:
            print(f"Fetching features for {company1}...")
            features1 = fetch_company_features_from_snowflake(snowflake_conn, company1)
            print(f"Fetching features for {company2}...")
            features2 = fetch_company_features_from_snowflake(snowflake_conn, company2)

        classification = classify_relationship(company1, company2, features1, features2)
        print(f"\nRelationship Classification:")
        print(f"  Type:              {classification.relationshipType}")
        print(f"  Hierarchy Level:   {classification.hierarchyLevel}")
        print(f"  Ownership:         {classification.ownershipPercentage}")
        print(f"  Strength:          {classification.relationshipStrength}")
        print(f"  Confidence:        {classification.confidenceScore}/100")
        print("-" * 60)

    if snowflake_conn:
        snowflake_conn.close()
        print("\n✅ Closed Snowflake connection")

else:
    print("❌ Invalid mode selected.")
