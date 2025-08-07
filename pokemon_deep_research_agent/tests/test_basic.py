"""
Basic tests for Pokemon Deep Research Agent.

These tests verify core functionality and system components.
"""

import asyncio
from pathlib import Path

import pytest


class TestBasicFunctionality:
    """Test basic system functionality."""

    def test_imports(self):
        """Test that all core modules can be imported."""
        # Test core imports
        # Test agent imports
        from src.agents.base_agent import BaseAgent
        from src.agents.coordinator_agent import CoordinatorAgent
        from src.core.communication import MessageBus, TaskChannel
        from src.core.models import AgentCapability, Message, Task
        # Test system imports
        from src.system.orchestrator import PokemonDeepResearchSystem
        from src.tools.llm_client import LLMClient
        # Test tool imports
        from src.tools.pokeapi_client import PokéAPIClient

        assert True  # If we get here, imports worked

    def test_project_structure(self):
        """Test that required project files exist."""
        project_root = Path(__file__).parent.parent

        required_files = [
            "main.py",
            "demo.py",
            "requirements.txt",
            "Dockerfile",
            "docker-compose.dev.yaml",
            "Makefile",
            ".env.example",
            "README.md",
        ]

        for file_name in required_files:
            file_path = project_root / file_name
            assert file_path.exists(), f"Required file {file_name} not found"

    @pytest.mark.asyncio
    async def test_message_bus(self):
        """Test message bus functionality."""
        from src.core.communication import MessageBus
        from src.core.models import Message, MessageType

        bus = MessageBus()
        await bus.start()

        received_messages = []

        async def test_callback(message):
            received_messages.append(message)

        await bus.subscribe("test", test_callback)

        test_message = Message(
            sender="test_sender",
            recipient="test_recipient",
            message_type=MessageType.COORDINATION,
            content={"test": "data"},
        )

        await bus.publish("test", test_message)
        await asyncio.sleep(0.1)  # Allow message processing

        assert len(received_messages) == 1
        assert received_messages[0].content["test"] == "data"

        await bus.stop()

    @pytest.mark.asyncio
    async def test_task_channel(self):
        """Test task channel functionality."""
        from src.core.communication import MessageBus, TaskChannel
        from src.core.models import Task, TaskPriority

        bus = MessageBus()
        await bus.start()

        channel = TaskChannel(bus)
        await channel.start()

        test_task = Task(
            title="Test Task",
            description="Test description",
            task_type="test",
            priority=TaskPriority.MEDIUM,
        )

        await channel.submit_task(test_task)

        retrieved_task = await channel.get_next_task("test_agent", ["test"])

        assert retrieved_task is not None
        assert retrieved_task.title == "Test Task"

        await channel.stop()
        await bus.stop()


class TestConfiguration:
    """Test configuration and environment setup."""

    def test_config_loading(self):
        """Test configuration loading."""
        import json
        from pathlib import Path

        config_path = Path(__file__).parent.parent / "config.json"
        assert config_path.exists(), "config.json not found"

        with open(config_path) as f:
            config = json.load(f)

        required_keys = [
            "llm_model",
            "llm_max_tokens",
            "llm_temperature",
            "pokeapi_base_url",
        ]

        for key in required_keys:
            assert key in config, f"Required config key {key} not found"

    def test_env_example(self):
        """Test that .env.example contains required variables."""
        from pathlib import Path

        env_example_path = Path(__file__).parent.parent / ".env.example"
        assert env_example_path.exists(), ".env.example not found"

        with open(env_example_path) as f:
            content = f.read()

        required_vars = [
            "OPENAI_API_KEY",
            "LLM_MODEL",
            "POKEAPI_BASE_URL",
            "LOGGING_LEVEL",
        ]

        for var in required_vars:
            assert (
                var in content
            ), f"Required environment variable {var} not in .env.example"


@pytest.mark.integration
class TestSystemIntegration:
    """Integration tests for the complete system."""

    @pytest.mark.asyncio
    async def test_system_initialization(self, test_config):
        """Test system initialization."""
        from src.system.orchestrator import PokemonDeepResearchSystem

        system = PokemonDeepResearchSystem(test_config)
        await system.initialize()

        # Test that agents are initialized
        assert len(system.agents) > 0

        # Test that communication system is running
        assert system.message_bus is not None
        assert system.task_channel is not None

        await system.shutdown()

    @pytest.mark.asyncio
    async def test_basic_query_processing(self, test_system):
        """Test basic query processing."""
        # This is a mock test since we're using mock clients
        query = "Tell me about Pikachu"

        # The system should be able to process this without errors
        # In a real test, we would check the actual response
        assert test_system is not None
        assert len(test_system.agents) > 0


if __name__ == "__main__":
    pytest.main([__file__])
