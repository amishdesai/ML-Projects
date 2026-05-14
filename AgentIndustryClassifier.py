import anthropic
import os
import snowflake.connector
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = (
    "You are Industry Classifier for Company Profiles. "
    "Understand the top level domain and generate 'Primary Industry', 2022 "
    "'North American Industry Classification System' code and 2022 'SIC' code for a given domain. "
    "Give me confidence score you feel this is right values. "
    "Can you crosscheck the NAICS with North American Industry Classification System for 2022."
)


class IndustryInfo(BaseModel):
    industryCodes: str = Field(description="Primary Industry of the Domain")
    naicsCodes: str = Field(description="Primary 2022 NAICS code of the Domain")
    sicCodes: str = Field(description="Primary SIC code of the Domain")
    crosscheckValid: str = Field(description="Crosscheck NAICS code with North American Industry Classification System and classify Valid or Non-Valid")
    accuracyScore: int = Field(description="Primary accuracy score of the Domain")


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


def fetch_domains_from_snowflake(conn, table_name, domain_column, limit=None):
    try:
        cursor = conn.cursor()
        query = f"SELECT {domain_column} FROM {table_name} WHERE non_publishable=false and website !='' and revenue_band='$1B-$5B'"
        if limit:
            query += f" LIMIT {limit}"
        cursor.execute(query)
        domains = [row[0] for row in cursor.fetchall()]
        cursor.close()
        print(f"📊 Fetched {len(domains)} domains from {table_name}\n")
        return domains
    except Exception as e:
        print(f"❌ Error fetching domains: {e}\n")
        return []


def classify_domain(domain: str) -> IndustryInfo | str:
    try:
        response = client.messages.parse(
            model="claude-opus-4-7",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": domain}],
            output_format=IndustryInfo,
        )
        return response.parsed_output
    except Exception as e:
        return str(e)


# Main execution
print("🤖 Industry Classifier\n")
print("Choose mode:")
print("1. Read from Snowflake")
print("2. Manual input")

mode = input("\nEnter mode (1 or 2): ").strip()

if mode == "1":
    conn = connect_to_snowflake()
    if conn:
        table_name = input("Enter table name: ").strip()
        domain_column = input("Enter domain column name: ").strip()
        limit_input = input("Enter limit (or press Enter for all): ").strip()
        limit = int(limit_input) if limit_input else None

        domains = fetch_domains_from_snowflake(conn, table_name, domain_column, limit)
        if domains:
            print(f"\n{'='*80}")
            print("Starting classification...\n")
            for idx, domain in enumerate(domains, 1):
                print(f"[{idx}/{len(domains)}] Classifying: {domain}")
                classification = classify_domain(domain)
                print(f"Result: {classification}\n")
                print("-" * 80)
        conn.close()

elif mode == "2":
    print("\n🤖 Industry Classifier is ready! Type 'bye' or 'exit' to stop.\n")
    while True:
        user_input = input("Company Domain: ").strip()
        if user_input.lower() in ["bye", "exit"]:
            print("Classifier: Goodbye 👋")
            break
        classification = classify_domain(user_input)
        print(f"Classification: {classification}\n")
        print("-" * 60)

else:
    print("❌ Invalid mode selected.")
