"""Example 13: Structured Extraction with Instructor

Demonstrates:
- Using instructor with axemere.gateway.openai.OpenAI for structured output
- Pydantic model extraction from unstructured text
- Using instructor with axemere.gateway.anthropic.Anthropic
- Error handling for connectivity and policy errors

Run:
    python 13_instructor.py
"""

import sys
from typing import List

import instructor
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from axemere.gateway.anthropic import Anthropic
from axemere.gateway.openai import OpenAI
from axemere.gateway import AiGatewayConfig


class Person(BaseModel):
    name: str = Field(description="The person's full name")
    age: int = Field(description="The person's age in years")
    occupation: str = Field(description="The person's job or occupation")


class MeetingNotes(BaseModel):
    attendees: List[str] = Field(description="Names of meeting attendees")
    action_items: List[str] = Field(description="List of action items from the meeting")
    summary: str = Field(description="One-sentence summary of the meeting")


def demo_openai_extraction(cfg: AiGatewayConfig) -> None:
    print("\n--- Structured extraction via OpenAI ---")
    raw_client = OpenAI(config=cfg)
    client = instructor.from_openai(raw_client)

    text = "Alice Johnson is a 32-year-old software engineer who loves building distributed systems."
    person = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=256,
        response_model=Person,
        messages=[{"role": "user", "content": f"Extract the person's details from: {text}"}],
    )
    print(f"Name: {person.name}")
    print(f"Age: {person.age}")
    print(f"Occupation: {person.occupation}")


def demo_anthropic_extraction(cfg: AiGatewayConfig) -> None:
    print("\n--- Structured extraction via Anthropic ---")
    raw_client = Anthropic(config=cfg)
    client = instructor.from_anthropic(raw_client)

    notes_text = (
        "Meeting with Bob, Carol, and Dave. "
        "We decided to launch the new feature next Friday. "
        "Bob will write tests, Carol will update the docs, Dave will deploy."
    )
    notes = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        response_model=MeetingNotes,
        messages=[
            {"role": "user", "content": f"Extract structured meeting notes from: {notes_text}"}
        ],
    )
    print(f"Attendees: {', '.join(notes.attendees)}")
    print(f"Action items:")
    for item in notes.action_items:
        print(f"  - {item}")
    print(f"Summary: {notes.summary}")


def main() -> None:
    load_dotenv()

    cfg = AiGatewayConfig.from_env()
    print(f"Gateway: {cfg.gateway_url}")
    print(f"Workload: {cfg.workload_id}  Project: {cfg.project_id}")

    try:
        demo_openai_extraction(cfg)
        demo_anthropic_extraction(cfg)
    except Exception as exc:
        msg = str(exc).lower()
        if "connection" in msg or "connect" in msg or "refused" in msg:
            print(f"\nConnectivity error: {exc}")
            print("Is the Axemere AI Gateway running? Try: docker compose up -d")
        else:
            print(f"\nError: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
