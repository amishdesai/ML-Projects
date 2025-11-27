# langchain v1 + Anthropic + create_agent + Hierarchical Relationship Classifier
from langchain.agents import create_agent
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import json
import os
import snowflake.connector
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

load_dotenv()

# configure the LLM
model = "claude-sonnet-4-5-20250929"


class RelationshipInfo(BaseModel):
    """Hierarchical relationship classification information."""
    relationshipType: str = Field(description="Type of relationship: Parent-Subsidiary, Sister Company, Joint Venture, Franchise, etc.")
    hierarchyLevel: str = Field(description="Position in hierarchy: Ultimate Parent, Intermediate Parent, Subsidiary, Branch, Affiliate, etc.")
    ownershipPercentage: str = Field(description="Estimated ownership percentage or control level")
    relationshipStrength: str = Field(description="Strength of relationship: Direct, Indirect, Affiliate, Strategic Partnership")
    confidenceScore: int = Field(description="Confidence score 1-100 for the relationship classification")


agent = create_agent(
    model=model,
    tools=[],
    response_format=RelationshipInfo,
    system_prompt=(
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
    ),
)


# Relationship Classification
relationshipData = {}


def connect_to_snowflake():
    """Establish connection to Snowflake using private key authentication."""
    try:
        # Load the private key from environment variable
        private_key_str = os.getenv('SNOWFLAKE_PRIVATE_KEY')

        # Replace literal \n with actual newlines if needed
        if '\\n' in private_key_str:
            private_key_str = private_key_str.replace('\\n', '\n')

        # Parse the private key
        private_key = serialization.load_pem_private_key(
            private_key_str.encode(),
            password=None,
            backend=default_backend()
        )

        # Get the private key in bytes format
        pkb = private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        conn = snowflake.connector.connect(
            account=os.getenv('SNOWFLAKE_ACCOUNT'),
            user=os.getenv('SNOWFLAKE_USER'),
            private_key=pkb,
            warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
            database=os.getenv('SNOWFLAKE_DATABASE'),
            schema=os.getenv('SNOWFLAKE_SCHEMA'),
            role=os.getenv('SNOWFLAKE_ROLE')
        )
        print("✅ Connected to Snowflake successfully!\n")
        return conn
    except Exception as e:
        print(f"❌ Error connecting to Snowflake: {e}\n")
        return None


def fetch_company_features_from_snowflake(conn, domain):
    """Fetch company features from Snowflake by domain/website."""
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
            features = {
                'name': row[0],
                'revenue': row[1] if row[1] else 'N/A',
                'employees': row[2] if row[2] else 'N/A',
                'website': row[3],
                'industry': row[4] if row[4] else 'N/A',
                'naics': row[5] if row[5] else 'N/A',
                'website_status': row[6]
            }
            return features
        else:
            print(f"  ⚠️ No features found in Snowflake for: {domain}")
            return {}

    except Exception as e:
        print(f"  ❌ Error fetching features for {domain}: {e}")
        return {}


def load_company_pairs_from_json(file_path):
    """Load company pairs with optional features from JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Support different JSON formats
        if isinstance(data, list):
            pairs = []
            for item in data:
                # Format 1: Objects with company1/company2 and optional features
                if isinstance(item, dict) and 'company1' in item:
                    company1 = item['company1']
                    company2 = item['company2']
                    features1 = item.get('features1', {})
                    features2 = item.get('features2', {})
                    pairs.append((company1, company2, features1, features2))
                # Format 2: Simple two-element arrays (no features)
                elif isinstance(item, list) and len(item) >= 2:
                    pairs.append((item[0], item[1], {}, {}))
                else:
                    print(f"⚠️ Skipping unrecognized item: {item}")
                    continue

            print(f"📊 Loaded {len(pairs)} company pairs from {file_path}\n")
            return pairs
        else:
            print("❌ JSON must be a list")
            return []
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}\n")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}\n")
        return []
    except Exception as e:
        print(f"❌ Error loading file: {e}\n")
        return []


def format_features(features):
    """Format features dictionary into readable string."""
    if not features:
        return "No additional features provided."

    lines = []
    for key, value in features.items():
        # Convert key from camelCase/snake_case to Title Case
        formatted_key = key.replace('_', ' ').replace('-', ' ').title()
        lines.append(f"  - {formatted_key}: {value}")

    return "\n".join(lines)


def classify_relationship(company1, company2, features1=None, features2=None):
    """Classify the relationship between two companies using the agent."""
    # Build prompt with optional features
    prompt_parts = [f"Analyze the relationship between '{company1}' and '{company2}'"]

    if features1:
        prompt_parts.append(f"\n\nCompany 1 ({company1}) Features:\n{format_features(features1)}")

    if features2:
        prompt_parts.append(f"\n\nCompany 2 ({company2}) Features:\n{format_features(features2)}")

    prompt = "".join(prompt_parts)
    messages = [{"role": "user", "content": prompt}]
    result = agent.invoke({"messages": messages})

    try:
        return result["structured_response"]
    except Exception as e:
        return str(e)


# Main execution
print("🤖 Hierarchical Relationship Classifier\n")
print("Choose mode:")
print("1. Read from JSON file")
print("2. Manual input (enter two companies)")

mode = input("\nEnter mode (1 or 2): ").strip()

if mode == "1":
    # JSON file mode
    file_path = input("Enter JSON file path (default: company_pairs.json): ").strip()
    if not file_path:
        file_path = "company_pairs.json"

    # Ask if user wants to fetch features from Snowflake
    use_snowflake = input("Fetch company features from Snowflake? (y/n): ").strip().lower() == 'y'

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

            # Fetch features from Snowflake if enabled and features not already provided
            if use_snowflake and snowflake_conn:
                if not features1:
                    print(f"  Fetching features for {company1}...")
                    features1 = fetch_company_features_from_snowflake(snowflake_conn, company1)
                if not features2:
                    print(f"  Fetching features for {company2}...")
                    features2 = fetch_company_features_from_snowflake(snowflake_conn, company2)

            # Show features if provided
            if features1:
                print(f"  Company 1 features: {list(features1.keys())}")
            if features2:
                print(f"  Company 2 features: {list(features2.keys())}")

            classification = classify_relationship(company1, company2, features1, features2)
            results.append({
                "company1": company1,
                "company2": company2,
                "relationship": classification
            })
            print(f"Result: {classification}\n")
            print("-"*80)

        if snowflake_conn:
            snowflake_conn.close()
            print("✅ Closed Snowflake connection\n")

        print(f"\n✅ Completed classification of {len(results)} relationships!")

        # Save results
        save_output = input("\nSave results to JSON? (y/n): ").strip().lower()
        if save_output == 'y':
            output_file = input("Enter output filename (default: relationship_results.json): ").strip()
            if not output_file:
                output_file = "relationship_results.json"

            # Convert results to serializable format
            serializable_results = []
            for r in results:
                serializable_results.append({
                    "company1": r["company1"],
                    "company2": r["company2"],
                    "relationship": {
                        "type": r["relationship"].relationshipType,
                        "hierarchy": r["relationship"].hierarchyLevel,
                        "ownership": r["relationship"].ownershipPercentage,
                        "strength": r["relationship"].relationshipStrength,
                        "confidence": r["relationship"].confidenceScore
                    }
                })

            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)

            print(f"✅ Results saved to {output_file}")

elif mode == "2":
    # Manual input mode
    print("\n🤖 Relationship Classifier is ready! Type 'bye' or 'exit' to stop.\n")

    # Ask if user wants to use Snowflake for feature fetching
    use_snowflake = input("Fetch company features from Snowflake? (y/n): ").strip().lower() == 'y'

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
        print("\n")

        # Fetch features from Snowflake if enabled
        features1 = {}
        features2 = {}

        if use_snowflake and snowflake_conn:
            print(f"Fetching features for {company1}...")
            features1 = fetch_company_features_from_snowflake(snowflake_conn, company1)

            print(f"Fetching features for {company2}...")
            features2 = fetch_company_features_from_snowflake(snowflake_conn, company2)

            if features1:
                print(f"✓ Found features for {company1}: {list(features1.keys())}")
            if features2:
                print(f"✓ Found features for {company2}: {list(features2.keys())}")
            print()

        classification = classify_relationship(company1, company2, features1, features2)
        print(f"Relationship Classification:\n")
        print(f"  Type:              {classification.relationshipType}")
        print(f"  Hierarchy Level:   {classification.hierarchyLevel}")
        print(f"  Ownership:         {classification.ownershipPercentage}")
        print(f"  Strength:          {classification.relationshipStrength}")
        print(f"  Confidence:        {classification.confidenceScore}/100")
        print("-"*60)

    if snowflake_conn:
        snowflake_conn.close()
        print("\n✅ Closed Snowflake connection")

else:
    print("❌ Invalid mode selected. Please run again and choose 1 or 2.")