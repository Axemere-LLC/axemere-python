"""Example 04: ReAct Agent with Tool Calling

Demonstrates:
- Defining @tool functions (weather and population lookups)
- Using create_react_agent from langgraph.prebuilt
- Every LLM round-trip in the agent loop passes through Axemere AI Gateway governance
- Tool call formatting works transparently through the gateway

Run:
    python 04_tool_agent.py
"""

import sys

from langchain_core.tools import tool

from axemere.gateway.langchain import ChatAiGateway
from axemere.gateway import AiGatewayConfig, PolicyDeniedError, GatewayError


@tool
def get_weather(city: str) -> str:
    """Return current weather for a city. Returns a mock response for demo purposes."""
    mock_weather = {
        "london": "Cloudy, 12C, light rain",
        "paris": "Sunny, 18C, light breeze",
        "new york": "Partly cloudy, 22C, humid",
        "tokyo": "Clear, 25C, calm",
    }
    key = city.lower().strip()
    return mock_weather.get(key, f"Weather data unavailable for {city}")


@tool
def get_population(city: str) -> str:
    """Return the approximate population of a city."""
    populations = {
        "london": "9.0 million (Greater London)",
        "paris": "2.2 million (city proper), 12.3 million (metro)",
        "new york": "8.3 million (city), 20 million (metro)",
        "tokyo": "14 million (city), 37 million (metro)",
    }
    key = city.lower().strip()
    return populations.get(key, f"Population data unavailable for {city}")


def main() -> None:
    cfg = AiGatewayConfig.from_env()
    print(f"Gateway: {cfg.gateway_url}")

    try:
        try:
            from langchain.agents import create_agent as create_react_agent
        except ImportError:
            from langgraph.prebuilt import create_react_agent
    except ImportError:
        print("langgraph not installed. Run: pip install langgraph")
        sys.exit(1)

    llm = ChatAiGateway(
        provider="openai",
        model="gpt-4o-mini",
        max_tokens=512,
        config=cfg,
    )

    tools = [get_weather, get_population]
    agent = create_react_agent(llm, tools)

    query = (
        "What is the weather and population of London and Paris? "
        "Give me a brief comparison."
    )
    print(f"\nQuery: {query}\n")

    try:
        result = agent.invoke({"messages": [("human", query)]})
    except PolicyDeniedError as exc:
        print(f"Policy denied: {exc}")
        print("Check that your gateway policies allow this workload/project.")
        sys.exit(1)
    except GatewayError as exc:
        print(f"Gateway error: {exc}")
        print("Is the Axemere AI Gateway running? Try: docker compose up -d")
        sys.exit(1)

    messages = result.get("messages", [])
    print(f"Agent completed in {len(messages)} messages (including tool calls)\n")

    # Print the final AI response
    for msg in reversed(messages):
        from langchain_core.messages import AIMessage

        if isinstance(msg, AIMessage) and msg.content:
            print("Agent response:")
            print(msg.content)
            break


if __name__ == "__main__":
    main()
