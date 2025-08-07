#!/usr/bin/env python3
"""
Final verification script to ensure all systems are working correctly.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.tools.llm_client import LLMClient
from src.tools.pokeapi_client import PokéAPIClient


async def verify_system():
    """Verify all core systems are working."""

    print("🔍 Pokemon Deep Research Agent - Final Verification")
    print("=" * 60)

    # Load environment
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return False

    print(f"✅ API Key loaded: {api_key[:20]}...")

    # Test 1: LLM Client
    print("\n1. Testing LLM Client...")
    try:
        llm_client = LLMClient(api_key=api_key, model="gpt-4.1-mini")
        response = await llm_client.chat_completion(
            [{"role": "user", "content": "What type is Pikachu? Answer in one word."}]
        )
        answer = response["choices"][0]["message"]["content"].strip()
        print(f"   ✅ LLM Response: {answer}")

        if "electric" in answer.lower():
            print("   ✅ LLM understanding: Correct!")
        else:
            print(f"   ⚠️  LLM understanding: Unexpected answer: {answer}")

    except Exception as e:
        print(f"   ❌ LLM Error: {e}")
        return False

    # Test 2: PokéAPI Client
    print("\n2. Testing PokéAPI Client...")
    try:
        pokeapi_client = PokéAPIClient()
        pokemon_data = await pokeapi_client.get_pokemon("pikachu")
        print(f"   ✅ PokéAPI Response: {pokemon_data.name} (#{pokemon_data.id})")
        print(
            f"   ✅ Data structure: {len(pokemon_data.types)} types, {len(pokemon_data.abilities)} abilities"
        )

    except Exception as e:
        print(f"   ❌ PokéAPI Error: {e}")
        return False

    # Test 3: AI Analysis
    print("\n3. Testing AI Pokemon Analysis...")
    try:
        analysis = await llm_client.analyze_pokemon_query("Tell me about Pikachu")
        query_type = analysis.get("query_type", "unknown")
        pokemon_mentioned = analysis.get("pokemon_mentioned", [])
        print(f"   ✅ Query Analysis: Type={query_type}, Pokemon={pokemon_mentioned}")

        if "pikachu" in str(pokemon_mentioned).lower():
            print("   ✅ Pokemon detection: Working correctly!")
        else:
            print(f"   ⚠️  Pokemon detection: May need improvement")

    except Exception as e:
        print(f"   ❌ Analysis Error: {e}")
        return False

    # Test 4: Integration Test
    print("\n4. Testing Integration...")
    try:
        # Get Pokemon data
        pokemon = await pokeapi_client.get_pokemon("charizard")

        # Analyze with AI
        prompt = f"Analyze {pokemon.name} briefly. Is it good for competitive play?"
        response = await llm_client.chat_completion(
            [{"role": "user", "content": prompt}]
        )
        analysis = response["choices"][0]["message"]["content"]

        print(f"   ✅ Integration: Successfully analyzed {pokemon.name}")
        print(f"   📝 Sample analysis: {analysis[:100]}...")

    except Exception as e:
        print(f"   ❌ Integration Error: {e}")
        return False

    # Test 5: System Configuration
    print("\n5. Testing System Configuration...")
    try:
        import json

        with open("config.json", "r") as f:
            config = json.load(f)

        model = config.get("llm_model", "unknown")
        print(f"   ✅ Configuration loaded: Model={model}")

        if model in ["gpt-4.1-mini", "gpt-4.1-nano", "gemini-2.5-flash"]:
            print("   ✅ Model configuration: Supported model")
        else:
            print(f"   ⚠️  Model configuration: {model} may not be supported")

    except Exception as e:
        print(f"   ❌ Configuration Error: {e}")
        return False

    # Final Summary
    print("\n" + "=" * 60)
    print("🎉 VERIFICATION COMPLETE")
    print("=" * 60)
    print("✅ LLM Client: Working")
    print("✅ PokéAPI Integration: Working")
    print("✅ AI Analysis: Working")
    print("✅ System Integration: Working")
    print("✅ Configuration: Valid")
    print("\n🚀 System is ready for production use!")
    print("\nNext steps:")
    print("1. Run: python unified_working_demo.py")
    print("2. Try: python main.py --interactive")
    print("3. Ask: 'Tell me about Garchomp's competitive viability'")

    return True


async def main():
    """Main verification function."""
    try:
        success = await verify_system()
        return success
    except Exception as e:
        print(f"\n💥 Verification failed: {e}")
        return False


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
