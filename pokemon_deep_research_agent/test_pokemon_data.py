#!/usr/bin/env python3
"""
Test script to examine Pokemon data structure from PokéAPI.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.tools.pokeapi_client import PokéAPIClient


async def test_pokemon_data():
    """Test Pokemon data structure."""

    client = PokéAPIClient()

    try:
        # Get Pikachu data
        pokemon = await client.get_pokemon("pikachu")

        print("🔍 Pokemon Data Structure Analysis")
        print("=" * 50)
        print(f"Name: {pokemon.name}")
        print(f"ID: {pokemon.id}")
        print(f"Height: {pokemon.height}")
        print(f"Weight: {pokemon.weight}")

        print(f"\nTypes structure: {type(pokemon.types)}")
        print(f"Types content: {pokemon.types}")

        print(f"\nAbilities structure: {type(pokemon.abilities)}")
        print(f"Abilities content: {pokemon.abilities}")

        print(f"\nStats structure: {type(pokemon.stats)}")
        print(f"Stats content: {pokemon.stats}")

        # Try to extract type names
        if pokemon.types:
            print(f"\nFirst type: {pokemon.types[0]}")
            if isinstance(pokemon.types[0], dict) and "type" in pokemon.types[0]:
                print(f"Type name: {pokemon.types[0]['type']['name']}")

        # Try to extract ability names
        if pokemon.abilities:
            print(f"\nFirst ability: {pokemon.abilities[0]}")
            if (
                isinstance(pokemon.abilities[0], dict)
                and "ability" in pokemon.abilities[0]
            ):
                print(f"Ability name: {pokemon.abilities[0]['ability']['name']}")

        # Try to extract stats
        if pokemon.stats:
            print(f"\nFirst stat: {pokemon.stats[0]}")
            if isinstance(pokemon.stats[0], dict) and "stat" in pokemon.stats[0]:
                print(f"Stat name: {pokemon.stats[0]['stat']['name']}")
                print(f"Base stat: {pokemon.stats[0]['base_stat']}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(test_pokemon_data())
