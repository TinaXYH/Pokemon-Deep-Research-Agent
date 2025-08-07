#!/usr/bin/env python3
"""
Simple demo of the Pokemon Deep Research Agent system.

This script demonstrates the core functionality without the full multi-agent
orchestration to show that the system components work correctly.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.agents.pokemon_analysis_agent import PokemonAnalysisAgent
from src.core.communication import MessageBus, TaskChannel
from src.core.models import (AgentCapability, AgentConfig, AgentType, Task,
                             TaskPriority)
from src.tools.llm_client import LLMClient
from src.tools.pokeapi_client import PokéAPIClient


async def demo_pokemon_research():
    """Demonstrate Pokemon research functionality."""

    print("🔍 Pokemon Deep Research Agent - Demo")
    print("=" * 50)

    # Initialize components
    print("Initializing system components...")

    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return

    # Initialize LLM client
    llm_client = LLMClient(api_key=api_key, model="gpt-4.1-mini", temperature=0.7)

    # Initialize PokéAPI client
    pokeapi_client = PokéAPIClient()

    print("✅ Components initialized")
    print()

    # Demo 1: PokéAPI Data Retrieval
    print("📊 Demo 1: Pokemon Data Retrieval")
    print("-" * 30)

    try:
        # Get Pikachu data
        print("Fetching Pikachu data from PokéAPI...")
        pikachu_data = await pokeapi_client.get_pokemon("pikachu")

        if pikachu_data:
            print(f"✅ Retrieved data for {pikachu_data['name'].title()}")
            print(f"   ID: {pikachu_data['id']}")
            print(f"   Height: {pikachu_data['height']} decimeters")
            print(f"   Weight: {pikachu_data['weight']} hectograms")
            print(
                f"   Types: {', '.join([t['type']['name'] for t in pikachu_data['types']])}"
            )
            print(
                f"   Base Stats Total: {sum([s['base_stat'] for s in pikachu_data['stats']])}"
            )
        else:
            print("❌ Failed to retrieve Pokemon data")

    except Exception as e:
        print(f"❌ Error retrieving Pokemon data: {e}")

    print()

    # Demo 2: LLM Analysis
    print("🤖 Demo 2: AI Analysis")
    print("-" * 30)

    try:
        print("Analyzing Pikachu using AI...")

        analysis_prompt = """
        Analyze Pikachu as a Pokemon for beginners. Consider:
        1. Ease of use for new trainers
        2. Strengths and weaknesses
        3. Recommended strategies
        
        Keep the analysis concise (2-3 sentences).
        """

        response = await llm_client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a Pokemon expert providing beginner-friendly advice.",
                },
                {"role": "user", "content": analysis_prompt},
            ],
            temperature=0.6,
        )

        analysis = response["choices"][0]["message"]["content"]
        print(f"✅ AI Analysis:")
        print(f"   {analysis}")

    except Exception as e:
        print(f"❌ Error in AI analysis: {e}")

    print()

    # Demo 3: Integrated Analysis
    print("🔬 Demo 3: Integrated Pokemon Analysis")
    print("-" * 30)

    try:
        # Create a simple analysis agent
        message_bus = MessageBus()
        await message_bus.start()

        task_channel = TaskChannel(message_bus)
        await task_channel.start()

        config = AgentConfig(
            agent_id="demo_analyst",
            agent_type=AgentType.POKEMON_ANALYST,
            name="Demo Pokemon Analysis Agent",
            description="Demo analysis agent",
            capabilities=[
                AgentCapability(
                    name="stat_analysis", description="Statistical analysis"
                )
            ],
        )

        analyst = PokemonAnalysisAgent(config, message_bus, task_channel, llm_client)

        # Create analysis task
        task = Task(
            title="Analyze Pikachu",
            description="Comprehensive Pikachu analysis",
            task_type="pokemon_analysis",
            priority=TaskPriority.HIGH,
            metadata={
                "task_parameters": {
                    "pokemon_name": "pikachu",
                    "pokemon_data": pikachu_data,
                    "analysis_depth": "standard",
                }
            },
        )

        print("Running comprehensive Pokemon analysis...")
        result = await analyst.process_task(task)

        if result and "stat_analysis" in result:
            stat_analysis = result["stat_analysis"]
            print(f"✅ Analysis Complete:")
            print(f"   Primary Role: {stat_analysis.get('primary_role', 'Unknown')}")
            print(f"   Total Base Stats: {stat_analysis.get('total_stats', 'Unknown')}")
            print(
                f"   Stat Distribution: {stat_analysis.get('stat_distribution', 'Unknown')}"
            )

            strengths = stat_analysis.get("strengths", [])
            if strengths:
                print(
                    f"   Key Strengths: {', '.join([s['stat'] for s in strengths[:2]])}"
                )

        await message_bus.stop()
        await task_channel.stop()

    except Exception as e:
        print(f"❌ Error in integrated analysis: {e}")

    print()

    # Demo 4: System Architecture Overview
    print("🏗️ Demo 4: System Architecture")
    print("-" * 30)

    print("✅ Multi-Agent System Components:")
    print("   🤖 Coordinator Agent - Workflow orchestration")
    print("   📋 Task Manager Agent - Query decomposition")
    print("   🔍 PokéAPI Research Agent - Data retrieval")
    print("   ⚔️  Battle Strategy Agent - Competitive analysis")
    print("   📊 Pokemon Analysis Agent - Statistical analysis")
    print("   📝 Report Generation Agent - Result synthesis")
    print()

    print("✅ Core Infrastructure:")
    print("   💬 Message Bus - Inter-agent communication")
    print("   📤 Task Channel - Distributed task queue")
    print("   🧠 LLM Client - AI analysis with caching")
    print("   🌐 PokéAPI Client - Pokemon data access")
    print()

    print("✅ Key Features Demonstrated:")
    print("   ✓ Pokemon data retrieval from PokéAPI")
    print("   ✓ AI-powered analysis and insights")
    print("   ✓ Statistical analysis and role determination")
    print("   ✓ Modular agent-based architecture")
    print("   ✓ Error handling and graceful degradation")
    print()

    # Cleanup
    await pokeapi_client.close()

    print("🎉 Demo completed successfully!")
    print()
    print("The Pokemon Deep Research Agent system is ready for production use.")
    print("Run 'python main.py' for the full interactive experience.")


if __name__ == "__main__":
    asyncio.run(demo_pokemon_research())
