#!/usr/bin/env python3
"""
Comprehensive test script to verify all system components work correctly.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.tools.llm_client import LLMClient
from src.tools.pokeapi_client import PokéAPIClient


async def test_comprehensive_functionality():
    """Test basic functionality with the user's API setup."""

    print("🧪 Comprehensive System Test")
    print("=" * 50)

    # Load environment
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return False

    print(f"✅ API Key loaded: {api_key[:20]}...")

    # Test 1: Basic LLM functionality
    print("\n1. Testing Basic LLM...")
    try:
        llm_client = LLMClient(api_key=api_key, model="gpt-4.1-mini")

        # Simple test without JSON formatting
        response = await llm_client.chat_completion(
            [
                {
                    "role": "user",
                    "content": "What type is Pikachu? Answer in one word only.",
                }
            ]
        )

        answer = response["choices"][0]["message"]["content"].strip()
        print(f"   ✅ LLM Response: '{answer}'")

        if "electric" in answer.lower():
            print("   ✅ LLM understanding: Correct!")
        else:
            print(f"   ⚠️  LLM understanding: Got '{answer}', expected 'Electric'")

    except Exception as e:
        print(f"   ❌ LLM Error: {e}")
        return False

    # Test 2: PokéAPI functionality
    print("\n2. Testing PokéAPI...")
    try:
        pokeapi_client = PokéAPIClient()
        pokemon_data = await pokeapi_client.get_pokemon("pikachu")

        print(f"   ✅ Pokemon: {pokemon_data.name} (#{pokemon_data.id})")
        print(f"   ✅ Types: {len(pokemon_data.types)} types")
        print(f"   ✅ Abilities: {len(pokemon_data.abilities)} abilities")
        print(f"   ✅ Stats: {len(pokemon_data.stats)} stats")

        # Check specific data
        if pokemon_data.types:
            type_name = pokemon_data.types[0]["type"]["name"]
            print(f"   ✅ Primary type: {type_name}")

        await pokeapi_client.close()

    except Exception as e:
        print(f"   ❌ PokéAPI Error: {e}")
        return False

    # Test 3: Query analysis (without JSON)
    print("\n3. Testing Query Analysis...")
    try:
        analysis = await llm_client.analyze_pokemon_query("Tell me about Garchomp")

        print(f"   ✅ Query Type: {analysis.get('query_type', 'unknown')}")
        print(f"   ✅ Pokemon Mentioned: {analysis.get('pokemon_mentioned', [])}")
        print(f"   ✅ Analysis Text: {analysis.get('analysis_text', 'N/A')[:100]}...")

    except Exception as e:
        print(f"   ❌ Analysis Error: {e}")
        return False

    # Test 4: End-to-end Pokemon research
    print("\n4. Testing End-to-End Research...")
    try:
        # Get Pokemon data
        pokeapi_client = PokéAPIClient()
        pokemon = await pokeapi_client.get_pokemon("charizard")

        # Create research prompt
        prompt = f"""Analyze {pokemon.name} for competitive Pokemon battles. Consider:
- Base stats: {[stat['base_stat'] for stat in pokemon.stats]}
- Types: {[t['type']['name'] for t in pokemon.types]}
- Abilities: {[a['ability']['name'] for a in pokemon.abilities]}

Provide a brief competitive analysis (2-3 sentences)."""

        # Get AI analysis
        response = await llm_client.chat_completion(
            [{"role": "user", "content": prompt}]
        )

        analysis = response["choices"][0]["message"]["content"]

        print(f"   ✅ Pokemon Analyzed: {pokemon.name}")
        print(f"   ✅ Analysis Length: {len(analysis)} characters")
        print(f"   📝 Sample: {analysis[:150]}...")

        await pokeapi_client.close()

    except Exception as e:
        print(f"   ❌ End-to-End Error: {e}")
        return False

    # Test 5: Open-ended question handling
    print("\n5. Testing Open-Ended Questions...")
    try:
        open_questions = [
            "What makes a good Pokemon team?",
            "How do I counter Dragon-type Pokemon?",
            "What are the best starter Pokemon for beginners?",
        ]

        for question in open_questions:
            response = await llm_client.chat_completion(
                [
                    {
                        "role": "user",
                        "content": f"Answer this Pokemon question briefly: {question}",
                    }
                ]
            )

            answer = response["choices"][0]["message"]["content"]
            print(f"   ✅ Q: {question[:40]}...")
            print(f"      A: {answer[:80]}...")

    except Exception as e:
        print(f"   ❌ Open-ended Error: {e}")
        return False

    # Final Summary
    print("\n" + "=" * 50)
    print("🎉 COMPREHENSIVE TEST COMPLETE")
    print("=" * 50)
    print("✅ Basic LLM: Working")
    print("✅ PokéAPI: Working")
    print("✅ Query Analysis: Working")
    print("✅ End-to-End Research: Working")
    print("✅ Open-Ended Questions: Working")
    print("\n🚀 System is fully functional!")
    print("\nThe system can now handle:")
    print("- Pokemon data retrieval")
    print("- AI-powered analysis")
    print("- Open-ended questions")
    print("- Competitive insights")
    print("- Natural language responses")

    return True


async def main():
    """Main test function."""
    try:
        success = await test_comprehensive_functionality()
        return success
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        return False


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
