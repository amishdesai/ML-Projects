import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

with client.messages.stream(
    model="claude-opus-4-7",
    max_tokens=16000,
    system="You are a helpful chat assistant. Be clear, concise, and polite. Understand the user's intent and respond directly. Stay professional and safe.",
    messages=[{"role": "user", "content": "Explain Machine Learning in short"}],
) as stream:
    result = stream.get_final_message()

print(result.content[0].text)
