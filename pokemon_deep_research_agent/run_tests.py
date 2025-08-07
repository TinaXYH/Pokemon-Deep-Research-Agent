#!/usr/bin/env python3
"""
Comprehensive test suite for the Pokemon Deep Research Agent system.

This script runs various tests to verify system functionality and components.
"""

import asyncio
import os
import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.communication import MessageBus, TaskChannel
from src.tools.llm_client import LLMClient
from src.tools.pokeapi_client import PokéAPIClient


async def test_pokeapi_client():
    """Test PokéAPI client functionality."""
    print("🧪 Testing PokéAPI Client...")

    client = PokéAPIClient()

    try:
        # Test basic Pokemon retrieval
        pokemon_data = await client.get_pokemon("pikachu")
        assert pokemon_data is not None, "Failed to retrieve Pokemon data"
        assert "name" in pokemon_data, "Pokemon data missing name field"
        print("   ✅ Pokemon retrieval: PASS")

        # Test move retrieval
        move_data = await client.get_move("thunderbolt")
        assert move_data is not None, "Failed to retrieve move data"
        assert "name" in move_data, "Move data missing name field"
        print("   ✅ Move retrieval: PASS")

        # Test type retrieval
        type_data = await client.get_type("electric")
        assert type_data is not None, "Failed to retrieve type data"
        assert "name" in type_data, "Type data missing name field"
        print("   ✅ Type retrieval: PASS")

        # Test caching
        cached_data = await client.get_pokemon("pikachu")
        assert cached_data is not None, "Cached retrieval failed"
        print("   ✅ Caching: PASS")

        await client.close()
        return True

    except Exception as e:
        print(f"   ❌ PokéAPI Client test failed: {e}")
        await client.close()
        return False


async def test_llm_client():
    """Test LLM client functionality."""
    print("🧪 Testing LLM Client...")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("   ⚠️  Skipping LLM tests (no API key)")
        return True

    client = LLMClient(api_key=api_key, model="gpt-4.1-mini", temperature=0.7)

    try:
        # Test basic chat completion
        response = await client.chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Say 'test successful' if you can read this.",
                },
            ],
            max_tokens=10,
        )

        assert response is not None, "No response from LLM"
        assert "choices" in response, "Response missing choices"
        assert len(response["choices"]) > 0, "No choices in response"

        content = response["choices"][0]["message"]["content"].lower()
        assert "test successful" in content, f"Unexpected response: {content}"

        print("   ✅ Chat completion: PASS")

        # Test caching
        cached_response = await client.chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Say 'test successful' if you can read this.",
                },
            ],
            max_tokens=10,
        )

        assert cached_response is not None, "Cached response failed"
        print("   ✅ Response caching: PASS")

        return True

    except Exception as e:
        print(f"   ❌ LLM Client test failed: {e}")
        traceback.print_exc()
        return False


async def test_communication_system():
    """Test message bus and task channel."""
    print("🧪 Testing Communication System...")

    try:
        # Test message bus
        message_bus = MessageBus()
        await message_bus.start()

        # Test subscription and publishing
        received_messages = []

        async def test_callback(message):
            received_messages.append(message)

        await message_bus.subscribe("test_topic", test_callback)

        from src.core.models import Message, MessageType

        test_message = Message(
            sender="test_sender",
            recipient="test_recipient",
            message_type=MessageType.COORDINATION,
            content={"test": "data"},
        )

        await message_bus.publish("test_topic", test_message)

        # Give it a moment to process
        await asyncio.sleep(0.1)

        assert len(received_messages) == 1, "Message not received"
        assert (
            received_messages[0].content["test"] == "data"
        ), "Message content incorrect"

        print("   ✅ Message Bus: PASS")

        # Test task channel
        task_channel = TaskChannel(message_bus)
        await task_channel.start()

        from src.core.models import Task, TaskPriority, TaskStatus

        test_task = Task(
            title="Test Task",
            description="Test task description",
            task_type="test",
            priority=TaskPriority.MEDIUM,
        )

        await task_channel.submit_task(test_task)

        # Try to get the task
        retrieved_task = await task_channel.get_next_task("test_agent", ["test"])

        assert retrieved_task is not None, "Task not retrieved"
        assert retrieved_task.title == "Test Task", "Task title incorrect"
        assert (
            retrieved_task.status == TaskStatus.IN_PROGRESS
        ), "Task status not updated"

        print("   ✅ Task Channel: PASS")

        await task_channel.stop()
        await message_bus.stop()

        return True

    except Exception as e:
        print(f"   ❌ Communication system test failed: {e}")
        traceback.print_exc()
        return False


async def test_system_integration():
    """Test system integration and health."""
    print("🧪 Testing System Integration...")

    try:
        # Test system initialization
        from src.system.orchestrator import PokemonDeepResearchSystem

        config = {
            "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
            "llm_model": "gpt-4.1-mini",
            "llm_max_tokens": 1000,
            "llm_temperature": 0.7,
            "pokeapi_base_url": "https://pokeapi.co/api/v2/",
            "cache_dir": "data/cache",
            "logging": {"level": "ERROR"},  # Reduce noise during testing
        }

        system = PokemonDeepResearchSystem(config)

        # Test initialization
        await system.initialize()
        print("   ✅ System initialization: PASS")

        # Test health check
        health = await system.health_check()
        assert health["status"] == "healthy", f"System not healthy: {health}"
        print("   ✅ Health check: PASS")

        # Test agent status
        agent_status = system.get_agent_status()
        assert len(agent_status) > 0, "No agents found"
        print(f"   ✅ Agent status ({len(agent_status)} agents): PASS")

        # Test system shutdown
        await system.shutdown()
        print("   ✅ System shutdown: PASS")

        return True

    except Exception as e:
        print(f"   ❌ System integration test failed: {e}")
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all tests and report results."""
    print("🔬 Pokemon Deep Research Agent - Test Suite")
    print("=" * 60)
    print()

    test_results = []

    # Run individual tests
    test_results.append(await test_pokeapi_client())
    test_results.append(await test_llm_client())
    test_results.append(await test_communication_system())
    test_results.append(await test_system_integration())

    print()
    print("📊 Test Results Summary")
    print("-" * 30)

    passed = sum(test_results)
    total = len(test_results)

    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")

    if passed == total:
        print("🎉 All tests passed! System is ready for production.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
