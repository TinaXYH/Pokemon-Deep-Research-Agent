"""
PokéAPI Research Agent for the Pokémon Deep Research Agent system.

The PokéAPI Research Agent is specialized in:
- Executing API queries against PokéAPI endpoints
- Caching and managing API response data
- Performing data validation and error handling
- Extracting relevant information for specific research needs
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from ..agents.base_agent import BaseAgent
from ..core.communication import MessageBus, TaskChannel
from ..core.models import (AgentConfig, Message, MessageType, ResearchResult,
                           Task)
from ..tools.pokeapi_client import PokéAPIClient


class PokéAPIResearchAgent(BaseAgent):
    """
    PokéAPI Research Agent specialized in Pokemon data retrieval and analysis.

    This agent handles all interactions with the PokéAPI, providing comprehensive
    Pokemon data retrieval, caching, and preprocessing for other agents.
    """

    def __init__(
        self,
        config: AgentConfig,
        message_bus: MessageBus,
        task_channel: TaskChannel,
        pokeapi_client: PokéAPIClient,
    ):
        super().__init__(config, message_bus, task_channel)
        self.pokeapi_client = pokeapi_client

        # Research specializations
        self.research_handlers = {
            "pokemon_research": self._handle_pokemon_research,
            "pokemon_identification": self._handle_pokemon_identification,
            "battle_research": self._handle_battle_research,
            "location_research": self._handle_location_research,
            "location_search": self._handle_location_search,
            "type_research": self._handle_type_research,
            "move_research": self._handle_move_research,
            "ability_research": self._handle_ability_research,
            "general_research": self._handle_general_research,
        }

        # Common Pokemon filters and categories
        self.competitive_pokemon = [
            "garchomp",
            "metagross",
            "salamence",
            "tyranitar",
            "dragonite",
            "charizard",
            "blastoise",
            "venusaur",
            "alakazam",
            "gengar",
            "machamp",
            "golem",
            "lapras",
            "snorlax",
            "umbreon",
            "espeon",
        ]

        self.beginner_friendly_pokemon = [
            "pikachu",
            "eevee",
            "lucario",
            "gardevoir",
            "blaziken",
            "swampert",
            "sceptile",
            "infernape",
            "empoleon",
            "torterra",
        ]

        # Type effectiveness chart (simplified)
        self.type_chart = {
            "normal": {"weak_to": ["fighting"], "resists": [], "immune_to": ["ghost"]},
            "fire": {
                "weak_to": ["water", "ground", "rock"],
                "resists": ["fire", "grass", "ice", "bug", "steel", "fairy"],
                "immune_to": [],
            },
            "water": {
                "weak_to": ["electric", "grass"],
                "resists": ["fire", "water", "ice", "steel"],
                "immune_to": [],
            },
            "electric": {
                "weak_to": ["ground"],
                "resists": ["electric", "flying", "steel"],
                "immune_to": [],
            },
            "grass": {
                "weak_to": ["fire", "ice", "poison", "flying", "bug"],
                "resists": ["water", "electric", "grass", "ground"],
                "immune_to": [],
            },
            "ice": {
                "weak_to": ["fire", "fighting", "rock", "steel"],
                "resists": ["ice"],
                "immune_to": [],
            },
            "fighting": {
                "weak_to": ["flying", "psychic", "fairy"],
                "resists": ["bug", "rock", "dark"],
                "immune_to": [],
            },
            "poison": {
                "weak_to": ["ground", "psychic"],
                "resists": ["grass", "fighting", "poison", "bug", "fairy"],
                "immune_to": [],
            },
            "ground": {
                "weak_to": ["water", "grass", "ice"],
                "resists": ["poison", "rock"],
                "immune_to": ["electric"],
            },
            "flying": {
                "weak_to": ["electric", "ice", "rock"],
                "resists": ["grass", "fighting", "bug"],
                "immune_to": ["ground"],
            },
            "psychic": {
                "weak_to": ["bug", "ghost", "dark"],
                "resists": ["fighting", "psychic"],
                "immune_to": [],
            },
            "bug": {
                "weak_to": ["fire", "flying", "rock"],
                "resists": ["grass", "fighting", "ground"],
                "immune_to": [],
            },
            "rock": {
                "weak_to": ["water", "grass", "fighting", "ground", "steel"],
                "resists": ["normal", "fire", "poison", "flying"],
                "immune_to": [],
            },
            "ghost": {
                "weak_to": ["ghost", "dark"],
                "resists": ["poison", "bug"],
                "immune_to": ["normal", "fighting"],
            },
            "dragon": {
                "weak_to": ["ice", "dragon", "fairy"],
                "resists": ["fire", "water", "electric", "grass"],
                "immune_to": [],
            },
            "dark": {
                "weak_to": ["fighting", "bug", "fairy"],
                "resists": ["ghost", "dark"],
                "immune_to": ["psychic"],
            },
            "steel": {
                "weak_to": ["fire", "fighting", "ground"],
                "resists": [
                    "normal",
                    "grass",
                    "ice",
                    "flying",
                    "psychic",
                    "bug",
                    "rock",
                    "dragon",
                    "steel",
                    "fairy",
                ],
                "immune_to": ["poison"],
            },
            "fairy": {
                "weak_to": ["poison", "steel"],
                "resists": ["fighting", "bug", "dark"],
                "immune_to": ["dragon"],
            },
        }

    async def _initialize(self) -> None:
        """Initialize PokéAPI client and resources."""
        await self.pokeapi_client.start()
        self.logger.info("PokéAPI Research Agent initialized")

    async def _cleanup(self) -> None:
        """Cleanup PokéAPI client resources."""
        await self.pokeapi_client.close()

    async def process_task(self, task: Task) -> Dict[str, Any]:
        """Process PokéAPI research tasks."""
        task_type = task.task_type

        if task_type not in self.research_handlers:
            raise ValueError(f"Unknown research task type: {task_type}")

        handler = self.research_handlers[task_type]

        try:
            result = await handler(task)

            # Add metadata to result
            result.update(
                {
                    "agent_id": self.config.agent_id,
                    "task_id": str(task.id),
                    "timestamp": datetime.now().isoformat(),
                    "api_stats": self.pokeapi_client.get_cache_stats(),
                }
            )

            return result

        except Exception as e:
            self.logger.error(f"Research task failed: {e}")
            raise

    async def _handle_pokemon_research(self, task: Task) -> Dict[str, Any]:
        """Handle comprehensive Pokemon data research."""
        params = task.metadata.get("task_parameters", {})

        # Extract parameters
        pokemon_name = params.get("pokemon_name")
        type_filter = params.get("type_filter")
        competitive_filter = params.get("competitive_filter", False)
        game_version = params.get("game_version", "general")
        include_stats = params.get("include_stats", True)
        include_moves = params.get("include_moves", True)
        include_abilities = params.get("include_abilities", True)

        results = []

        if pokemon_name:
            # Research specific Pokemon
            pokemon_data = await self._get_detailed_pokemon_data(
                pokemon_name, include_stats, include_moves, include_abilities
            )
            results.append(pokemon_data)

        elif type_filter:
            # Research Pokemon by type
            type_pokemon = await self.pokeapi_client.search_pokemon_by_type(type_filter)

            # Limit to reasonable number for detailed analysis
            limited_pokemon = (
                type_pokemon[:20] if len(type_pokemon) > 20 else type_pokemon
            )

            for pokemon_ref in limited_pokemon:
                try:
                    pokemon_data = await self._get_detailed_pokemon_data(
                        pokemon_ref["pokemon"]["name"],
                        include_stats,
                        include_moves,
                        include_abilities,
                    )
                    results.append(pokemon_data)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to get data for {pokemon_ref['pokemon']['name']}: {e}"
                    )
                    continue

        elif competitive_filter:
            # Research competitive Pokemon
            for pokemon_name in self.competitive_pokemon:
                try:
                    pokemon_data = await self._get_detailed_pokemon_data(
                        pokemon_name, include_stats, include_moves, include_abilities
                    )
                    results.append(pokemon_data)
                except Exception as e:
                    self.logger.warning(f"Failed to get data for {pokemon_name}: {e}")
                    continue

        return {
            "type": "pokemon_research",
            "results": results,
            "count": len(results),
            "filters_applied": {
                "pokemon_name": pokemon_name,
                "type_filter": type_filter,
                "competitive_filter": competitive_filter,
                "game_version": game_version,
            },
            "confidence": 1.0,
            "sources": ["pokeapi.co"],
        }

    async def _handle_pokemon_identification(self, task: Task) -> Dict[str, Any]:
        """Handle Pokemon identification based on query context."""
        params = task.metadata.get("task_parameters", {})
        query_context = params.get("query_context", "")
        game_version = params.get("game_version", "general")

        # Analyze query context to identify Pokemon
        identified_pokemon = []

        # Look for keywords that might indicate specific Pokemon or criteria
        query_lower = query_context.lower()

        if "easy" in query_lower and "train" in query_lower:
            # Easy to train Pokemon
            identified_pokemon = self.beginner_friendly_pokemon[:5]

        elif (
            "ruby" in query_lower
            or "sapphire" in query_lower
            or "emerald" in query_lower
        ):
            # Hoenn region Pokemon that are beginner-friendly
            hoenn_beginner = [
                "blaziken",
                "swampert",
                "sceptile",
                "gardevoir",
                "metagross",
            ]
            identified_pokemon = hoenn_beginner

        elif "sea" in query_lower or "water" in query_lower:
            # Water-type or sea-dwelling Pokemon
            water_pokemon = ["lapras", "gyarados", "starmie", "tentacruel", "kingdra"]
            identified_pokemon = water_pokemon

        elif "bug" in query_lower:
            # Bug-type Pokemon
            bug_pokemon = ["scizor", "heracross", "forretress", "skarmory", "volcarona"]
            identified_pokemon = bug_pokemon

        else:
            # Default to popular/versatile Pokemon
            identified_pokemon = [
                "pikachu",
                "charizard",
                "blastoise",
                "venusaur",
                "alakazam",
            ]

        # Get detailed data for identified Pokemon
        results = []
        for pokemon_name in identified_pokemon:
            try:
                pokemon_data = await self._get_detailed_pokemon_data(
                    pokemon_name, True, True, True
                )
                results.append(pokemon_data)
            except Exception as e:
                self.logger.warning(f"Failed to get data for {pokemon_name}: {e}")
                continue

        return {
            "type": "pokemon_identification",
            "identified_pokemon": identified_pokemon,
            "results": results,
            "query_context": query_context,
            "identification_reasoning": self._explain_identification(
                query_context, identified_pokemon
            ),
            "confidence": 0.8,
            "sources": ["pokeapi.co"],
        }

    def _explain_identification(self, query: str, pokemon_list: List[str]) -> str:
        """Explain why specific Pokemon were identified."""
        query_lower = query.lower()

        if "easy" in query_lower and "train" in query_lower:
            return (
                "Identified beginner-friendly Pokemon with good stats and easy movesets"
            )
        elif "ruby" in query_lower:
            return "Identified Hoenn region Pokemon suitable for Pokemon Ruby"
        elif "sea" in query_lower or "water" in query_lower:
            return "Identified Water-type and sea-dwelling Pokemon"
        elif "bug" in query_lower:
            return "Identified strong Bug-type Pokemon for team building"
        else:
            return "Identified popular and versatile Pokemon suitable for various situations"

    async def _handle_battle_research(self, task: Task) -> Dict[str, Any]:
        """Handle battle-specific Pokemon research."""
        params = task.metadata.get("task_parameters", {})
        pokemon_name = params.get("pokemon_name")
        battle_format = params.get("battle_format", "singles")
        include_movesets = params.get("include_movesets", True)
        include_counters = params.get("include_counters", True)

        if not pokemon_name:
            raise ValueError("Pokemon name required for battle research")

        # Get comprehensive Pokemon data
        pokemon_data = await self._get_detailed_pokemon_data(
            pokemon_name, True, True, True
        )

        # Analyze battle viability
        battle_analysis = await self._analyze_battle_viability(
            pokemon_data, battle_format
        )

        # Get type effectiveness information
        types = [t["type"]["name"] for t in pokemon_data.get("types", [])]
        type_analysis = self._analyze_type_effectiveness(types)

        # Recommend movesets
        moveset_recommendations = await self._recommend_movesets(
            pokemon_data, battle_format
        )

        result = {
            "type": "battle_research",
            "pokemon": pokemon_data,
            "battle_analysis": battle_analysis,
            "type_effectiveness": type_analysis,
            "recommended_movesets": moveset_recommendations,
            "battle_format": battle_format,
            "confidence": 0.9,
            "sources": ["pokeapi.co"],
        }

        if include_counters:
            result["counters"] = self._identify_counters(types, pokemon_data)

        return result

    async def _handle_location_research(self, task: Task) -> Dict[str, Any]:
        """Handle location-based Pokemon research."""
        params = task.metadata.get("task_parameters", {})
        location_name = params.get("location_name")
        game_version = params.get("game_version", "general")
        include_encounter_rates = params.get("include_encounter_rates", True)

        if not location_name:
            raise ValueError("Location name required for location research")

        try:
            # Try to get location data
            location_data = await self.pokeapi_client.get_location(location_name)

            # Get location areas
            areas = location_data.get("areas", [])
            pokemon_encounters = []

            for area_ref in areas:
                try:
                    area_data = await self.pokeapi_client.get_location_area(
                        area_ref["name"]
                    )
                    encounters = area_data.get("pokemon_encounters", [])

                    for encounter in encounters:
                        pokemon_name = encounter["pokemon"]["name"]
                        version_details = encounter.get("version_details", [])

                        pokemon_encounters.append(
                            {
                                "pokemon": pokemon_name,
                                "area": area_ref["name"],
                                "encounter_details": version_details,
                            }
                        )

                except Exception as e:
                    self.logger.warning(
                        f"Failed to get area data for {area_ref['name']}: {e}"
                    )
                    continue

            return {
                "type": "location_research",
                "location": location_name,
                "pokemon_encounters": pokemon_encounters,
                "total_pokemon": len(set(e["pokemon"] for e in pokemon_encounters)),
                "game_version": game_version,
                "confidence": 0.9,
                "sources": ["pokeapi.co"],
            }

        except Exception as e:
            self.logger.error(f"Location research failed for {location_name}: {e}")
            return {
                "type": "location_research",
                "location": location_name,
                "error": f"Location data not available: {str(e)}",
                "pokemon_encounters": [],
                "confidence": 0.0,
                "sources": ["pokeapi.co"],
            }

    async def _handle_location_search(self, task: Task) -> Dict[str, Any]:
        """Handle general location search based on criteria."""
        params = task.metadata.get("task_parameters", {})
        search_criteria = params.get("search_criteria", "")
        game_version = params.get("game_version", "general")

        # This is a simplified implementation
        # In a full implementation, you'd search through location data

        suggested_locations = []

        criteria_lower = search_criteria.lower()
        if "sea" in criteria_lower or "water" in criteria_lower:
            suggested_locations = [
                {
                    "name": "route-19",
                    "description": "Water route with many Water-type Pokemon",
                },
                {"name": "seafoam-islands", "description": "Cave system near the sea"},
                {
                    "name": "cinnabar-island",
                    "description": "Island location with water access",
                },
            ]
        elif "cave" in criteria_lower:
            suggested_locations = [
                {
                    "name": "rock-tunnel",
                    "description": "Dark cave with Rock and Ground types",
                },
                {
                    "name": "cerulean-cave",
                    "description": "Deep cave with powerful Pokemon",
                },
            ]
        elif "forest" in criteria_lower or "grass" in criteria_lower:
            suggested_locations = [
                {
                    "name": "viridian-forest",
                    "description": "Forest with Bug and Grass types",
                },
                {"name": "route-2", "description": "Route with forest areas"},
            ]

        return {
            "type": "location_search",
            "search_criteria": search_criteria,
            "suggested_locations": suggested_locations,
            "game_version": game_version,
            "confidence": 0.7,
            "sources": ["pokeapi.co"],
        }

    async def _handle_type_research(self, task: Task) -> Dict[str, Any]:
        """Handle Pokemon type research and effectiveness."""
        params = task.metadata.get("task_parameters", {})
        type_name = params.get("type_name")

        if not type_name:
            raise ValueError("Type name required for type research")

        type_data = await self.pokeapi_client.get_type(type_name)

        return {
            "type": "type_research",
            "type_data": type_data.model_dump(),
            "effectiveness_analysis": self._analyze_type_effectiveness([type_name]),
            "confidence": 1.0,
            "sources": ["pokeapi.co"],
        }

    async def _handle_move_research(self, task: Task) -> Dict[str, Any]:
        """Handle Pokemon move research."""
        params = task.metadata.get("task_parameters", {})
        move_name = params.get("move_name")
        move_list = params.get("move_list", [])

        results = []

        if move_name:
            move_data = await self.pokeapi_client.get_move(move_name)
            results.append(move_data.model_dump())

        if move_list:
            for move in move_list:
                try:
                    move_data = await self.pokeapi_client.get_move(move)
                    results.append(move_data.model_dump())
                except Exception as e:
                    self.logger.warning(f"Failed to get move data for {move}: {e}")

        return {
            "type": "move_research",
            "moves": results,
            "count": len(results),
            "confidence": 1.0,
            "sources": ["pokeapi.co"],
        }

    async def _handle_ability_research(self, task: Task) -> Dict[str, Any]:
        """Handle Pokemon ability research."""
        params = task.metadata.get("task_parameters", {})
        ability_name = params.get("ability_name")

        if not ability_name:
            raise ValueError("Ability name required for ability research")

        ability_data = await self.pokeapi_client.get_ability(ability_name)

        return {
            "type": "ability_research",
            "ability_data": ability_data,
            "confidence": 1.0,
            "sources": ["pokeapi.co"],
        }

    async def _handle_general_research(self, task: Task) -> Dict[str, Any]:
        """Handle general research queries."""
        params = task.metadata.get("task_parameters", {})
        query = params.get("query", "")

        # Try to extract Pokemon names from the query
        query_words = query.lower().split()

        # Simple keyword matching for Pokemon names
        potential_pokemon = []
        for word in query_words:
            if len(word) > 3:  # Avoid very short words
                try:
                    # Try to get Pokemon data to see if it's a valid name
                    pokemon_data = await self.pokeapi_client.get_pokemon(word)
                    potential_pokemon.append(word)
                except:
                    continue

        if potential_pokemon:
            # Research identified Pokemon
            results = []
            for pokemon_name in potential_pokemon:
                pokemon_data = await self._get_detailed_pokemon_data(
                    pokemon_name, True, True, True
                )
                results.append(pokemon_data)

            return {
                "type": "general_research",
                "query": query,
                "identified_pokemon": potential_pokemon,
                "results": results,
                "confidence": 0.8,
                "sources": ["pokeapi.co"],
            }
        else:
            # No specific Pokemon identified, return general guidance
            return {
                "type": "general_research",
                "query": query,
                "message": "No specific Pokemon identified in query. Consider providing more specific Pokemon names or criteria.",
                "suggestions": [
                    "Try mentioning specific Pokemon names",
                    "Use type-based queries",
                    "Specify game versions",
                ],
                "confidence": 0.3,
                "sources": ["pokeapi.co"],
            }

    async def _get_detailed_pokemon_data(
        self,
        pokemon_name: str,
        include_stats: bool = True,
        include_moves: bool = True,
        include_abilities: bool = True,
    ) -> Dict[str, Any]:
        """Get comprehensive Pokemon data."""

        # Get basic Pokemon data
        pokemon_data = await self.pokeapi_client.get_pokemon(pokemon_name)
        result = pokemon_data.model_dump()

        # Add additional analysis
        if include_stats:
            result["stat_analysis"] = self._analyze_stats(result["stats"])

        if include_moves:
            result["move_analysis"] = self._analyze_moves(result["moves"])

        if include_abilities:
            result["ability_analysis"] = self._analyze_abilities(result["abilities"])

        # Add type effectiveness
        types = [t["type"]["name"] for t in result.get("types", [])]
        result["type_effectiveness"] = self._analyze_type_effectiveness(types)

        return result

    def _analyze_stats(self, stats: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze Pokemon stats."""
        stat_dict = {stat["stat"]["name"]: stat["base_stat"] for stat in stats}

        total_stats = sum(stat_dict.values())

        # Identify stat distribution
        highest_stat = max(stat_dict.items(), key=lambda x: x[1])
        lowest_stat = min(stat_dict.items(), key=lambda x: x[1])

        # Categorize Pokemon role based on stats
        role = "balanced"
        if stat_dict.get("attack", 0) > 100 or stat_dict.get("special-attack", 0) > 100:
            role = "attacker"
        elif (
            stat_dict.get("defense", 0) > 100
            or stat_dict.get("special-defense", 0) > 100
        ):
            role = "tank"
        elif stat_dict.get("speed", 0) > 100:
            role = "speed"
        elif stat_dict.get("hp", 0) > 100:
            role = "hp_tank"

        return {
            "total_stats": total_stats,
            "highest_stat": {"name": highest_stat[0], "value": highest_stat[1]},
            "lowest_stat": {"name": lowest_stat[0], "value": lowest_stat[1]},
            "suggested_role": role,
            "stat_distribution": stat_dict,
        }

    def _analyze_moves(self, moves: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze Pokemon moves."""
        total_moves = len(moves)

        # Categorize moves by learning method
        level_moves = [
            m
            for m in moves
            if any(
                d["move_learn_method"]["name"] == "level-up"
                for d in m["version_group_details"]
            )
        ]
        tm_moves = [
            m
            for m in moves
            if any(
                d["move_learn_method"]["name"] == "machine"
                for d in m["version_group_details"]
            )
        ]

        return {
            "total_moves": total_moves,
            "level_up_moves": len(level_moves),
            "tm_moves": len(tm_moves),
            "move_variety": (
                "high" if total_moves > 50 else "medium" if total_moves > 25 else "low"
            ),
        }

    def _analyze_abilities(self, abilities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze Pokemon abilities."""
        ability_names = [a["ability"]["name"] for a in abilities]
        hidden_abilities = [
            a["ability"]["name"] for a in abilities if a.get("is_hidden", False)
        ]

        return {
            "abilities": ability_names,
            "hidden_abilities": hidden_abilities,
            "ability_count": len(ability_names),
        }

    def _analyze_type_effectiveness(self, types: List[str]) -> Dict[str, Any]:
        """Analyze type effectiveness for given types."""
        if not types:
            return {}

        # Calculate combined weaknesses and resistances
        weaknesses = set()
        resistances = set()
        immunities = set()

        for type_name in types:
            type_info = self.type_chart.get(type_name, {})
            weaknesses.update(type_info.get("weak_to", []))
            resistances.update(type_info.get("resists", []))
            immunities.update(type_info.get("immune_to", []))

        # Remove overlaps (resistances cancel weaknesses)
        effective_weaknesses = weaknesses - resistances - immunities
        effective_resistances = resistances - weaknesses

        return {
            "types": types,
            "weaknesses": list(effective_weaknesses),
            "resistances": list(effective_resistances),
            "immunities": list(immunities),
            "weakness_count": len(effective_weaknesses),
            "resistance_count": len(effective_resistances),
            "defensive_rating": self._calculate_defensive_rating(
                effective_weaknesses, effective_resistances, immunities
            ),
        }

    def _calculate_defensive_rating(
        self, weaknesses: set, resistances: set, immunities: set
    ) -> str:
        """Calculate a simple defensive rating."""
        score = len(resistances) * 2 + len(immunities) * 3 - len(weaknesses)

        if score >= 5:
            return "excellent"
        elif score >= 2:
            return "good"
        elif score >= -1:
            return "average"
        else:
            return "poor"

    async def _analyze_battle_viability(
        self, pokemon_data: Dict[str, Any], battle_format: str
    ) -> Dict[str, Any]:
        """Analyze Pokemon's battle viability."""
        stats = pokemon_data.get("stat_analysis", {})
        total_stats = stats.get("total_stats", 0)

        # Simple viability assessment
        viability = "low"
        if total_stats >= 600:
            viability = "legendary"
        elif total_stats >= 500:
            viability = "high"
        elif total_stats >= 400:
            viability = "medium"

        return {
            "viability_tier": viability,
            "total_stats": total_stats,
            "battle_format": battle_format,
            "recommended_role": stats.get("suggested_role", "balanced"),
            "strengths": self._identify_strengths(pokemon_data),
            "weaknesses": self._identify_weaknesses(pokemon_data),
        }

    def _identify_strengths(self, pokemon_data: Dict[str, Any]) -> List[str]:
        """Identify Pokemon's battle strengths."""
        strengths = []

        stats = pokemon_data.get("stat_analysis", {})
        type_eff = pokemon_data.get("type_effectiveness", {})

        if stats.get("total_stats", 0) >= 500:
            strengths.append("High base stats")

        if len(type_eff.get("resistances", [])) >= 3:
            strengths.append("Good defensive typing")

        if stats.get("suggested_role") == "attacker":
            strengths.append("Strong offensive capabilities")

        move_analysis = pokemon_data.get("move_analysis", {})
        if move_analysis.get("move_variety") == "high":
            strengths.append("Diverse movepool")

        return strengths

    def _identify_weaknesses(self, pokemon_data: Dict[str, Any]) -> List[str]:
        """Identify Pokemon's battle weaknesses."""
        weaknesses = []

        stats = pokemon_data.get("stat_analysis", {})
        type_eff = pokemon_data.get("type_effectiveness", {})

        if len(type_eff.get("weaknesses", [])) >= 3:
            weaknesses.append("Multiple type weaknesses")

        if stats.get("total_stats", 0) < 400:
            weaknesses.append("Low base stats")

        lowest_stat = stats.get("lowest_stat", {})
        if lowest_stat.get("value", 0) < 50:
            weaknesses.append(f"Very low {lowest_stat.get('name', 'stat')}")

        return weaknesses

    async def _recommend_movesets(
        self, pokemon_data: Dict[str, Any], battle_format: str
    ) -> List[Dict[str, Any]]:
        """Recommend movesets for the Pokemon."""
        # This is a simplified implementation
        # In practice, you'd analyze the Pokemon's stats, type, and available moves

        suggested_role = pokemon_data.get("stat_analysis", {}).get(
            "suggested_role", "balanced"
        )

        movesets = []

        if suggested_role == "attacker":
            movesets.append(
                {
                    "name": "Physical Attacker",
                    "description": "Focus on high-damage physical moves",
                    "item_suggestions": ["Choice Band", "Life Orb"],
                    "nature_suggestion": "Adamant",
                    "priority_stats": ["attack", "speed"],
                }
            )
        elif suggested_role == "tank":
            movesets.append(
                {
                    "name": "Defensive Wall",
                    "description": "Focus on survivability and support",
                    "item_suggestions": ["Leftovers", "Rocky Helmet"],
                    "nature_suggestion": "Bold",
                    "priority_stats": ["hp", "defense"],
                }
            )
        else:
            movesets.append(
                {
                    "name": "Balanced",
                    "description": "Versatile moveset for various situations",
                    "item_suggestions": ["Leftovers", "Expert Belt"],
                    "nature_suggestion": "Modest",
                    "priority_stats": ["hp", "special-attack"],
                }
            )

        return movesets

    def _identify_counters(
        self, types: List[str], pokemon_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify Pokemon that counter this Pokemon."""
        counters = []

        # Find types that are super effective
        type_eff = pokemon_data.get("type_effectiveness", {})
        weaknesses = type_eff.get("weaknesses", [])

        for weakness in weaknesses:
            counters.append(
                {
                    "counter_type": weakness,
                    "reason": f"Super effective against {', '.join(types)}",
                    "effectiveness": "2x damage",
                }
            )

        return counters[:5]  # Limit to top 5 counters

    def get_research_stats(self) -> Dict[str, Any]:
        """Get research agent statistics."""
        status = self.get_status()
        api_stats = self.pokeapi_client.get_cache_stats()

        status.update(
            {
                "research_handlers": list(self.research_handlers.keys()),
                "api_statistics": api_stats,
                "competitive_pokemon_count": len(self.competitive_pokemon),
                "beginner_pokemon_count": len(self.beginner_friendly_pokemon),
            }
        )

        return status
