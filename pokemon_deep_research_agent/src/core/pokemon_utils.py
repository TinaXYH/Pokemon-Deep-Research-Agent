"""
Utility functions for processing Pokemon data from PokéAPI.
"""

from typing import Any, Dict, List

from .models import PokemonData


def extract_type_names(pokemon_data: PokemonData) -> List[str]:
    """Extract type names from Pokemon data."""
    if not pokemon_data.types:
        return []

    type_names = []
    for type_entry in pokemon_data.types:
        if isinstance(type_entry, dict) and "type" in type_entry:
            type_names.append(type_entry["type"]["name"])

    return type_names


def extract_ability_names(pokemon_data: PokemonData) -> List[str]:
    """Extract ability names from Pokemon data."""
    if not pokemon_data.abilities:
        return []

    ability_names = []
    for ability_entry in pokemon_data.abilities:
        if isinstance(ability_entry, dict) and "ability" in ability_entry:
            ability_names.append(ability_entry["ability"]["name"])

    return ability_names


def extract_stats_dict(pokemon_data: PokemonData) -> Dict[str, int]:
    """Extract stats as a dictionary from Pokemon data."""
    if not pokemon_data.stats:
        return {}

    stats_dict = {}
    for stat_entry in pokemon_data.stats:
        if (
            isinstance(stat_entry, dict)
            and "stat" in stat_entry
            and "base_stat" in stat_entry
        ):
            stat_name = stat_entry["stat"]["name"]
            base_stat = stat_entry["base_stat"]

            # Normalize stat names for easier access
            normalized_name = stat_name.replace("-", "_")
            stats_dict[normalized_name] = base_stat

            # Also add common aliases
            if stat_name == "hp":
                stats_dict["health"] = base_stat
            elif stat_name == "special-attack":
                stats_dict["sp_attack"] = base_stat
                stats_dict["special_attack"] = base_stat
            elif stat_name == "special-defense":
                stats_dict["sp_defense"] = base_stat
                stats_dict["special_defense"] = base_stat

    return stats_dict


def format_pokemon_summary(pokemon_data: PokemonData) -> str:
    """Create a formatted summary of Pokemon data."""
    types = extract_type_names(pokemon_data)
    abilities = extract_ability_names(pokemon_data)
    stats = extract_stats_dict(pokemon_data)

    summary = f"""Pokemon: {pokemon_data.name.title()} (#{pokemon_data.id})
Types: {', '.join(type_name.title() for type_name in types)}
Abilities: {', '.join(ability_name.title().replace('-', ' ') for ability_name in abilities)}
Base Stats:
  HP: {stats.get('hp', 'N/A')}
  Attack: {stats.get('attack', 'N/A')}
  Defense: {stats.get('defense', 'N/A')}
  Sp. Attack: {stats.get('special_attack', 'N/A')}
  Sp. Defense: {stats.get('special_defense', 'N/A')}
  Speed: {stats.get('speed', 'N/A')}
Physical: {pokemon_data.height/10}m tall, {pokemon_data.weight/10}kg"""

    return summary


def get_stat_total(pokemon_data: PokemonData) -> int:
    """Calculate total base stat value."""
    stats = extract_stats_dict(pokemon_data)
    return sum(stats.values())


def compare_pokemon_stats(
    pokemon1: PokemonData, pokemon2: PokemonData
) -> Dict[str, Any]:
    """Compare stats between two Pokemon."""
    stats1 = extract_stats_dict(pokemon1)
    stats2 = extract_stats_dict(pokemon2)

    comparison = {
        "pokemon1": {
            "name": pokemon1.name,
            "stats": stats1,
            "total": sum(stats1.values()),
        },
        "pokemon2": {
            "name": pokemon2.name,
            "stats": stats2,
            "total": sum(stats2.values()),
        },
        "differences": {},
        "winner_by_stat": {},
    }

    # Calculate differences and winners for each stat
    for stat_name in [
        "hp",
        "attack",
        "defense",
        "special_attack",
        "special_defense",
        "speed",
    ]:
        val1 = stats1.get(stat_name, 0)
        val2 = stats2.get(stat_name, 0)

        comparison["differences"][stat_name] = val1 - val2

        if val1 > val2:
            comparison["winner_by_stat"][stat_name] = pokemon1.name
        elif val2 > val1:
            comparison["winner_by_stat"][stat_name] = pokemon2.name
        else:
            comparison["winner_by_stat"][stat_name] = "tie"

    # Overall winner
    if comparison["pokemon1"]["total"] > comparison["pokemon2"]["total"]:
        comparison["overall_winner"] = pokemon1.name
    elif comparison["pokemon2"]["total"] > comparison["pokemon1"]["total"]:
        comparison["overall_winner"] = pokemon2.name
    else:
        comparison["overall_winner"] = "tie"

    return comparison
