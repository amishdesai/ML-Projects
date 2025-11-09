from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()
agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[],
    system_prompt="You are a helpful chat assistant. Be clear, concise, and polite. Understand the user’s intent and respond directly. Stay professional and safe."
)

result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "Explain Machine Learning in short"}]
    }
)

print(result["messages"][1].content)