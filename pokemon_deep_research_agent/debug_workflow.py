#!/usr/bin/env python3
"""
Debug script to trace the Pokemon Deep Research Agent workflow
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import logging

from src.core.models import Task, TaskPriority
from src.system.orchestrator import PokemonDeepResearchSystem

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


async def debug_workflow():
    """Debug the workflow step by step."""

    # Load environment
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found")
        return

    # Initialize system
    config = {
        "openai_api_key": api_key,
        "llm_model": "gpt-4.1-mini",
        "llm_max_tokens": 4000,
        "llm_temperature": 0.7,
        "llm_cache": True,
        "pokeapi_base_url": "https://pokeapi.co/api/v2/",
        "cache_dir": "data/cache",
        "pokeapi_rate_limit": 0.1,
        "pokeapi_max_concurrent": 10,
        "query_timeout": 300,
        "memory_dir": "data/memory",
        "embedding_model": "text-embedding-3-small",
        "max_memory_entries": 10000,
        "similarity_threshold": 0.7,
        "logging": {"level": "DEBUG"},
    }

    orchestrator = PokemonDeepResearchSystem(config)

    try:
        await orchestrator.initialize()

        print("🔍 System initialized. Checking agent capabilities...")

        # Print agent capabilities
        for agent_id, agent in orchestrator.agents.items():
            capabilities = [cap.name for cap in agent.config.capabilities]
            print(f"  {agent_id}: {capabilities}")

        print("\n🎯 Testing query: 'Tell me about Pikachu's competitive viability'")

        # Create a research query task
        query = "Tell me about Pikachu's competitive viability"
        task = Task(
            title=f"Research Query: {query}",
            description=f"Process user research query: {query}",
            task_type="research_query",
            priority=TaskPriority.HIGH,
            metadata={
                "query_data": {"query": query, "user_preferences": {}, "parameters": {}}
            },
        )

        print(f"📝 Created task: {task.title}")
        print(f"   Task type: {task.task_type}")
        print(f"   Metadata: {task.metadata}")

        # Submit task
        await orchestrator.task_channel.submit_task(task)
        print("✅ Task submitted to task channel")

        # Wait a bit and check what happens
        print("\n⏳ Waiting 10 seconds to see task processing...")
        await asyncio.sleep(10)

        # Check task status
        print(f"\n📊 Task status: {task.status}")
        if hasattr(task, "result") and task.result:
            print(f"📋 Task result: {task.result}")

        # Check if any subtasks were created
        print(f"\n🔍 Checking for subtasks in task channel...")
        pending_tasks = orchestrator.task_channel.pending_tasks
        print(f"   Pending tasks: {len(pending_tasks)}")

        for task_id, pending_task in pending_tasks.items():
            print(f"   - {pending_task.title} (type: {pending_task.task_type})")
            print(
                f"     Required capability: {pending_task.metadata.get('required_capability', 'None')}"
            )
            print(f"     Status: {pending_task.status}")

        # Check completed tasks
        completed_tasks = orchestrator.task_channel.completed_tasks
        print(f"\n✅ Completed tasks: {len(completed_tasks)}")
        for task_id, completed_task in completed_tasks.items():
            print(f"   - {completed_task.title}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(debug_workflow())
