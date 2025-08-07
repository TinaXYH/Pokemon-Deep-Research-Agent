#!/usr/bin/env python3
"""
Quick Start Pokemon Research Agent - Simplified Version

This script provides immediate access to Pokemon research capabilities
without the complexity of the full multi-agent system.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.tools.llm_client import LLMClient
from src.tools.pokeapi_client import PokéAPIClient


class SimplePokemonAgent:
    """Simplified Pokemon research agent for quick queries."""

    def __init__(self, api_key: str, model: str = "gpt-4.1-mini"):
        self.llm_client = LLMClient(api_key=api_key, model=model)
        self.pokeapi_client = PokéAPIClient()

    async def research_pokemon(self, query: str) -> str:
        """Research a Pokemon-related query and return a comprehensive answer."""

        # Analyze the query
        analysis = await self.llm_client.analyze_pokemon_query(query)
        query_type = analysis.get("query_type", "general")
        pokemon_mentioned = analysis.get("pokemon_mentioned", [])

        # Gather Pokemon data if specific Pokemon are mentioned
        pokemon_data = []
        for pokemon_name in pokemon_mentioned:
            try:
                pokemon = await self.pokeapi_client.get_pokemon(pokemon_name)
                pokemon_data.append(
                    {
                        "name": pokemon.name,
                        "id": pokemon.id,
                        "types": [t["type"]["name"] for t in pokemon.types],
                        "stats": {
                            stat["stat"]["name"]: stat["base_stat"]
                            for stat in pokemon.stats
                        },
                        "abilities": [a["ability"]["name"] for a in pokemon.abilities],
                        "height": pokemon.height,
                        "weight": pokemon.weight,
                    }
                )
            except Exception as e:
                print(f"⚠️  Could not retrieve data for {pokemon_name}: {e}")

        # Create comprehensive prompt
        if pokemon_data:
            data_text = "\n".join(
                [
                    f"**{p['name'].title()}** (#{p['id']}): "
                    f"Types: {', '.join(p['types'])}, "
                    f"HP: {p['stats'].get('hp', 'N/A')}, "
                    f"Attack: {p['stats'].get('attack', 'N/A')}, "
                    f"Defense: {p['stats'].get('defense', 'N/A')}, "
                    f"Sp.Atk: {p['stats'].get('special-attack', 'N/A')}, "
                    f"Sp.Def: {p['stats'].get('special-defense', 'N/A')}, "
                    f"Speed: {p['stats'].get('speed', 'N/A')}, "
                    f"Abilities: {', '.join(p['abilities'])}"
                    for p in pokemon_data
                ]
            )

            prompt = f"""You are a Pokemon expert. Answer this question: {query}

Pokemon Data Available:
{data_text}

Provide a comprehensive, detailed analysis based on the data and your Pokemon knowledge. Include:
- Competitive viability and tier placement
- Strengths and weaknesses
- Recommended movesets and items
- Team synergies
- Usage tips and strategies

Be specific and provide actionable advice."""
        else:
            prompt = f"""You are a Pokemon expert. Answer this question comprehensively: {query}

Provide detailed analysis including:
- Specific Pokemon recommendations
- Competitive strategies
- Team building advice
- Usage tips
- Any relevant data or statistics

Be thorough and helpful."""

        # Get AI response
        response = await self.llm_client.chat_completion(
            [{"role": "user", "content": prompt}]
        )

        return response["choices"][0]["message"]["content"]

    async def close(self):
        """Close the API clients."""
        await self.pokeapi_client.close()


async def interactive_mode():
    """Run in interactive mode for continuous queries."""

    # Load environment
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found")
        print("   Please set your API key in the .env file")
        return

    agent = SimplePokemonAgent(api_key)

    print("🎮 Pokemon Research Agent - Quick Start Mode")
    print("=" * 60)
    print("Ask me anything about Pokemon! Type 'quit' to exit.")
    print("\nExample questions:")
    print("- Tell me about Garchomp's competitive viability")
    print("- What are the best Fire-type Pokemon for beginners?")
    print("- How do I build a team around Dragonite?")
    print("- Compare Alakazam vs Gengar for VGC doubles")
    print("=" * 60)

    try:
        while True:
            query = input("\n🔍 Your question: ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                break

            if not query:
                continue

            print("\n🤖 Researching...")
            try:
                answer = await agent.research_pokemon(query)
                print(f"\n📝 **Answer:**\n{answer}")
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try rephrasing your question.")

    except KeyboardInterrupt:
        print("\n\nGoodbye!")

    finally:
        await agent.close()


async def single_query(query: str):
    """Process a single query and exit."""

    # Load environment
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found")
        return

    agent = SimplePokemonAgent(api_key)

    try:
        print(f"🔍 Question: {query}")
        print("\n🤖 Researching...")

        answer = await agent.research_pokemon(query)
        print(f"\n📝 **Answer:**\n{answer}")

    except Exception as e:
        print(f"\n❌ Error: {e}")

    finally:
        await agent.close()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Pokemon Research Agent - Quick Start",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python quick_start.py
  
  # Single query
  python quick_start.py --query "Tell me about Pikachu"
  
  # Help
  python quick_start.py --help
        """,
    )

    parser.add_argument(
        "--query", "-q", type=str, help="Ask a single question and exit"
    )

    args = parser.parse_args()

    if args.query:
        asyncio.run(single_query(args.query))
    else:
        asyncio.run(interactive_mode())


if __name__ == "__main__":
    main()
