import anthropic
import os
import dotenv

from dotenv import load_dotenv

# Set your Claude API key as an environment variable or directly in the code
# For security, using environment variables is recommended
load_dotenv(verbose=True)
client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


def simple_agent(prompt):
    """
    A simple AI agent that uses Claude's API to respond to a prompt.
    """
    try:
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",  # Or other Claude models like "claude-3-opus-20240229"
            max_tokens=1024,
            system="You are a helpful AI assistant.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    except Exception as e:
        return f"An error occurred: {e}"


if __name__ == "__main__":
    print("Welcome to your first AI agent!")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Agent: Goodbye!")
            break

        agent_response = simple_agent(user_input)
        print(f"Agent: {agent_response}")