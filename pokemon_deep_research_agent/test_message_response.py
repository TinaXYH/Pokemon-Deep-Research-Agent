#!/usr/bin/env python3
"""
Test script to verify message response mechanism
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import logging

from src.core.communication import MessageBus
from src.core.models import Message, MessageType

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


async def test_message_response():
    """Test the message response mechanism."""

    # Load environment
    from dotenv import load_dotenv

    load_dotenv()

    print("🔍 Testing message response mechanism...")

    # Initialize message bus
    message_bus = MessageBus()
    await message_bus.start()

    # Track received messages
    received_messages = []

    async def agent_a_handler(message):
        print(f"Agent A received: {message.message_type} from {message.sender}")
        received_messages.append(("A", message))

        # If this is a request, send a response
        if message.content.get("action") == "test_request":
            print("Agent A sending response...")
            response = Message(
                sender="agent_a",
                recipient=message.sender,
                message_type=MessageType.TASK_RESULT,
                content={"result": "test_response"},
                response_to=message.id,
            )
            await message_bus.publish(f"agent.{message.sender}", response)

    async def agent_b_handler(message):
        print(f"Agent B received: {message.message_type} from {message.sender}")
        received_messages.append(("B", message))

    # Subscribe agents
    await message_bus.subscribe("agent.agent_a", agent_a_handler)
    await message_bus.subscribe("agent.agent_b", agent_b_handler)

    print("✅ Agents subscribed")

    # Test: Agent B sends a message to Agent A and waits for response
    print("\n📤 Agent B sending request to Agent A...")

    response = await message_bus.send_message(
        sender="agent_b",
        recipient="agent_a",
        message_type=MessageType.TASK_ASSIGNMENT,
        content={"action": "test_request", "data": "test_data"},
        requires_response=True,
        timeout=10.0,
    )

    if response:
        print(f"✅ Agent B received response: {response.content}")
        print(f"   Response ID: {response.id}")
        print(f"   Response to: {response.response_to}")
    else:
        print("❌ Agent B did not receive response")

    print(f"\n📊 Total messages received: {len(received_messages)}")
    for agent, msg in received_messages:
        print(f"   {agent}: {msg.message_type} - {msg.content}")

    await message_bus.stop()
    print("✅ Test completed")


if __name__ == "__main__":
    asyncio.run(test_message_response())
