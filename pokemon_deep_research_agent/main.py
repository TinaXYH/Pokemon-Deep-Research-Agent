#!/usr/bin/env python3
"""
Main entry point for the Pokemon Deep Research Agent system.

This script provides the command-line interface for running the multi-agent
Pokemon research system with various modes and configurations.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.system.orchestrator import PokemonDeepResearchSystem


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file."""
    if not os.path.exists(config_path):
        return {}

    with open(config_path, "r") as f:
        if config_path.endswith(".json"):
            return json.load(f)
        else:
            # Simple key=value format
            config = {}
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    config[key.strip()] = value.strip()
            return config


def create_default_config() -> Dict[str, Any]:
    """Create default system configuration."""
    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "llm_model": "gpt-4",
        "llm_max_tokens": 4000,
        "llm_temperature": 0.7,
        "llm_cache": True,
        "pokeapi_base_url": "https://pokeapi.co/api/v2/",
        "cache_dir": "data/cache",
        "pokeapi_rate_limit": 0.1,
        "pokeapi_max_concurrent": 10,
        "query_timeout": 300,
        "logging": {"level": "INFO"},
    }


async def run_interactive_mode(config: Dict[str, Any]) -> None:
    """Run the system in interactive mode."""
    system = PokemonDeepResearchSystem(config)

    try:
        await system.initialize()
        await system.run_interactive_mode()
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        print(f"System error: {e}")
        return 1
    finally:
        await system.shutdown()

    return 0


async def run_single_query(config: Dict[str, Any], query: str) -> None:
    """Run a single query and exit."""
    system = PokemonDeepResearchSystem(config)

    try:
        await system.initialize()

        print(f"🔍 Processing query: {query}")
        print()

        result = await system.process_user_query(query)

        # Display result
        if result.get("status") == "completed":
            query_result = result.get("result", {})

            if "markdown_report" in query_result:
                print(query_result["markdown_report"])
            elif "synthesis" in query_result:
                synthesis = query_result["synthesis"]
                print(synthesis.get("synthesis_text", "Analysis completed"))
            else:
                print("Query completed successfully")
                print(f"Confidence: {query_result.get('confidence', 'Unknown')}")
        else:
            print(f"Query failed: {result.get('error', 'Unknown error')}")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        await system.shutdown()


async def run_system_status(config: Dict[str, Any]) -> None:
    """Check system status and exit."""
    system = PokemonDeepResearchSystem(config)

    try:
        await system.initialize()

        status = await system.get_system_status()

        print("🔍 Pokemon Deep Research Agent - System Status")
        print("=" * 50)
        print(f"Status: {status['status']}")
        print(f"System ID: {status['system_id']}")
        print(f"Uptime: {status['uptime_seconds']:.1f} seconds")
        print(f"Active Queries: {status['active_queries']}")
        print()

        print("Agents:")
        for agent_id, agent_status in status["agents"].items():
            running_status = "✅" if agent_status["is_running"] else "❌"
            print(f"  {running_status} {agent_status['name']}")
            print(
                f"     Tasks: {agent_status['completed_tasks']} completed, {agent_status['current_tasks']} active"
            )
        print()

        print("Components:")
        for component, comp_status in status["components"].items():
            comp_health = "✅" if comp_status["status"] == "healthy" else "❌"
            print(f"  {comp_health} {component}: {comp_status['status']}")

        return 0

    except Exception as e:
        print(f"Error checking status: {e}")
        return 1
    finally:
        await system.shutdown()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Pokemon Deep Research Agent - Multi-Agent Pokemon Research System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run in interactive mode
  python main.py

  # Run a single query
  python main.py --query "What are the best Fire-type Pokemon for beginners?"

  # Check system status
  python main.py --status

  # Use custom configuration
  python main.py --config config.json

  # Set OpenAI API key
  python main.py --api-key sk-your-key-here

Environment Variables:
  OPENAI_API_KEY    OpenAI API key for LLM access
        """,
    )

    parser.add_argument("--query", "-q", type=str, help="Run a single query and exit")

    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Start interactive mode for continuous queries",
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.json",
        help="Configuration file path (default: config.json)",
    )

    parser.add_argument(
        "--api-key", type=str, help="OpenAI API key (overrides config and environment)"
    )

    parser.add_argument(
        "--status", action="store_true", help="Check system status and exit"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="LLM model to use (default: gpt-4.1-mini)",
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/cache",
        help="Cache directory path (default: data/cache)",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Query timeout in seconds (default: 300)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Load configuration
    config = create_default_config()

    # Load from config file if it exists
    if os.path.exists(args.config):
        file_config = load_config(args.config)
        # Only update non-empty values from file config
        for key, value in file_config.items():
            if value:  # Only override if the file config value is not empty
                config[key] = value

    # Apply command line overrides
    if args.api_key:
        config["openai_api_key"] = args.api_key

    # Ensure we have an API key from environment if config is empty
    if not config.get("openai_api_key"):
        config["openai_api_key"] = os.getenv("OPENAI_API_KEY")

    if args.model:
        config["llm_model"] = args.model

    if args.cache_dir:
        config["cache_dir"] = args.cache_dir

    if args.timeout:
        config["query_timeout"] = args.timeout

    if args.verbose:
        config["logging"]["level"] = "DEBUG"

    # Validate required configuration
    if not config.get("openai_api_key"):
        print("❌ Error: OpenAI API key is required")
        print("   Set OPENAI_API_KEY environment variable or use --api-key")
        return 1

    # Create cache directory
    cache_dir = Path(config["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Run appropriate mode
    try:
        if args.status:
            return asyncio.run(run_system_status(config))
        elif args.query:
            return asyncio.run(run_single_query(config, args.query))
        elif args.interactive:
            return asyncio.run(run_interactive_mode(config))
        else:
            # Default to interactive mode if no specific mode is specified
            return asyncio.run(run_interactive_mode(config))

    except KeyboardInterrupt:
        print("\nGoodbye!")
        return 0
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
