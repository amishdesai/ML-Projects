# langchain v1 + Anthropic + create_agent + Simple Industry Classifier
from langchain.agents import create_agent
from dotenv import load_dotenv
from pydantic import BaseModel,Field
import snowflake.connector
import os
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

load_dotenv()

# configure the LLM
model = "claude-sonnet-4-5-20250929"


class IndustryInfo(BaseModel):
    """Contact information for a person."""
    industryCodes: str = Field(description="Primary Industry of the Domain")
    naicsCodes: str = Field(description="Primary 2022 NAICS code of the Domain")
    sicCodes: str = Field(description="Primary SIC code of the Domain")
    crosscheckValid: str = Field(description="Crosscheck NAICS code with  North American Industry Classification System and classify Valid or Non-Valid")
    accuracyScore:int = Field(description="Primary accuracy score of the Domain")


agent = create_agent(
    model=model,
     tools=[],
    response_format=IndustryInfo,
    system_prompt=(
        "You are Industry Classifier for Company Profiles "
        "Understand the top level domain and generate 'Primary Industry', 2022 'North American Industry Classification System' code and 2022 'SIC' code for a given domain. "
        "Give me confidence score you feel this is right values."
        "Can you crosscheck the NAICS with  North American Industry Classification System for 2022 "
    ),
)

# Industry Classification
industryCodes = {}

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
            password=None,  # Set to password if your key is encrypted
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


def fetch_domains_from_snowflake(conn, table_name, domain_column, limit=None):
    """Fetch company domains from Snowflake table."""
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


def classify_domain(domain):
    """Classify a single domain using the agent."""
    messages = [{"role": "user", "content": domain}]
    result = agent.invoke({"messages": messages})

    try:
        return result["structured_response"]
    except Exception as e:
        return str(e)


# Main execution
print("🤖 Industry Classifier with Snowflake Integration\n")
print("Choose mode:")
print("1. Read from Snowflake")
print("2. Manual input (original mode)")

mode = input("\nEnter mode (1 or 2): ").strip()

if mode == "1":
    # Snowflake mode
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

            results = []
            for idx, domain in enumerate(domains, 1):
                print(f"[{idx}/{len(domains)}] Classifying: {domain}")
                classification = classify_domain(domain)
                results.append({"domain": domain, "classification": classification})
                print(f"Result: {classification}\n")
                print("-"*80)

            print(f"\n✅ Completed classification of {len(results)} domains!")

        conn.close()

elif mode == "2":
    # Manual input mode (original)
    print("\n🤖 Industry Classifier is ready! Type 'bye' or 'exit' to stop.\n")

    while True:
        user_input = input("Company Domain: ").strip()
        print("\n")
        if user_input.lower() in ["bye", "exit"]:
            print("Classifier: Goodbye 👋")
            break

        classification = classify_domain(user_input)
        print(f"Assistant: {classification}\n")
        print("-"*60)

else:
    print("❌ Invalid mode selected. Please run again and choose 1 or 2.")
