import anthropic
import os
from dotenv import load_dotenv

load_dotenv(verbose=True)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def simple_agent(prompt: str) -> str:
    with client.messages.stream(
        model="claude-opus-4-7",
        max_tokens=16000,
        system="You are a helpful AI assistant.",
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        return stream.get_final_message().content[0].text


if __name__ == "__main__":
    print("Welcome to your first AI agent!")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Agent: Goodbye!")
            break
        print(f"Agent: {simple_agent(user_input)}")
