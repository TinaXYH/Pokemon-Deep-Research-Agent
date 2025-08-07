"""
pytest configuration for Pokemon Deep Research Agent tests.

This module provides shared fixtures and configuration for all tests.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import AsyncGenerator, Generator

import pytest
from dotenv import load_dotenv

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def pytest_configure(config):
    """
    Pytest hook that runs before test collection.
    We use this to load our test environment variables.
    This runs in the same Python session as the tests,
    before any test modules or fixtures are imported.
    """
    print("\n------------ Loading test environment ------------")
    test_env_path = Path(__file__).parent / "test.env"
    if not load_dotenv(str(test_env_path), override=True):
        raise RuntimeError(
            f"Failed to load test environment variables from {test_env_path}"
        )
    print(f"Test environment loaded from {test_env_path}")
    print("--------------------------------------------------\n")


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_config() -> dict:
    """Provide test configuration."""
    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY", "test-key"),
        "llm_model": "gpt-4.1-mini",
        "llm_max_tokens": 100,  # Reduced for testing
        "llm_temperature": 0.0,  # Deterministic for testing
        "llm_cache": False,  # Disable caching for tests
        "pokeapi_base_url": "https://pokeapi.co/api/v2/",
        "cache_dir": "tests/cache",
        "pokeapi_rate_limit": 0.0,  # No rate limiting in tests
        "pokeapi_max_concurrent": 1,  # Sequential for predictable tests
        "query_timeout": 30,  # Shorter timeout for tests
        "logging": {"level": "ERROR"},  # Reduce noise in test output
    }


@pytest.fixture
async def mock_pokeapi_client():
    """Provide a mock PokéAPI client for testing."""
    from src.tools.pokeapi_client import PokéAPIClient

    client = PokéAPIClient()

    # Mock some common responses
    client._cache = {
        "pokemon:pikachu": {
            "name": "pikachu",
            "id": 25,
            "types": [{"type": {"name": "electric"}}],
            "stats": [
                {"base_stat": 35, "stat": {"name": "hp"}},
                {"base_stat": 55, "stat": {"name": "attack"}},
                {"base_stat": 40, "stat": {"name": "defense"}},
                {"base_stat": 50, "stat": {"name": "special-attack"}},
                {"base_stat": 50, "stat": {"name": "special-defense"}},
                {"base_stat": 90, "stat": {"name": "speed"}},
            ],
        }
    }

    yield client
    await client.close()


@pytest.fixture
async def mock_llm_client(test_config):
    """Provide a mock LLM client for testing."""
    from src.tools.llm_client import LLMClient

    class MockLLMClient(LLMClient):
        async def chat_completion(self, messages, **kwargs):
            # Return a mock response
            return {
                "choices": [
                    {"message": {"content": "This is a mock response for testing."}}
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                },
            }

    client = MockLLMClient(
        api_key=test_config["openai_api_key"],
        model=test_config["llm_model"],
        temperature=test_config["llm_temperature"],
    )

    yield client


@pytest.fixture
async def test_system(test_config, mock_pokeapi_client, mock_llm_client):
    """Provide a test instance of the Pokemon Deep Research System."""
    from src.system.orchestrator import PokemonDeepResearchSystem

    system = PokemonDeepResearchSystem(test_config)

    # Replace clients with mocks
    system.pokeapi_client = mock_pokeapi_client
    system.llm_client = mock_llm_client

    await system.initialize()

    yield system

    await system.shutdown()


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Provide a temporary cache directory for tests."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return str(cache_dir)


@pytest.fixture(autouse=True)
def cleanup_cache():
    """Clean up cache files after each test."""
    yield
    # Cleanup logic here if needed
    pass


# Pytest markers
pytest_plugins = []


# Custom markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "api: mark test as requiring API access")


# You can add more configuration or fixtures here if needed
