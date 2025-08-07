"""
Pokemon Analysis Agent for the Pokémon Deep Research Agent system.

The Pokemon Analysis Agent specializes in:
- Statistical analysis and comparisons
- Move and ability deep-dive analysis  
- Type effectiveness calculations
- Performance metrics and viability assessments
"""

import asyncio
import logging
import statistics
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from ..agents.base_agent import BaseAgent
from ..core.communication import MessageBus, TaskChannel
from ..core.models import (
    AgentConfig, Message, MessageType, Task
)
from ..tools.llm_client import LLMClient


class PokemonAnalysisAgent(BaseAgent):
    """
    Pokemon Analysis Agent specialized in detailed Pokemon analysis.
    
    This agent performs in-depth analysis of Pokemon stats, abilities,
    movesets, and competitive viability.
    """
    
    def __init__(
        self, 
        config: AgentConfig, 
        message_bus: MessageBus, 
        task_channel: TaskChannel,
        llm_client: LLMClient
    ):
        super().__init__(config, message_bus, task_channel)
        self.llm_client = llm_client
        
        # Analysis handlers
        self.analysis_handlers = {
            "pokemon_analysis": self._handle_pokemon_analysis,
            "stat_analysis": self._handle_stat_analysis,
            "comparative_analysis": self._handle_comparative_analysis,
            "move_analysis": self._handle_move_analysis,
            "ability_analysis": self._handle_ability_analysis,
            "type_effectiveness_analysis": self._handle_type_effectiveness_analysis,
            "training_analysis": self._handle_training_analysis,
            "viability_assessment": self._handle_viability_assessment
        }
        
        # Stat benchmarks for analysis
        self.stat_benchmarks = {
            "hp": {"low": 60, "average": 80, "high": 100, "excellent": 120},
            "attack": {"low": 60, "average": 80, "high": 100, "excellent": 130},
            "defense": {"low": 60, "average": 80, "high": 100, "excellent": 130},
            "special-attack": {"low": 60, "average": 80, "high": 100, "excellent": 130},
            "special-defense": {"low": 60, "average": 80, "high": 100, "excellent": 130},
            "speed": {"low": 60, "average": 80, "high": 100, "excellent": 120}
        }
        
        # Move categories for analysis
        self.move_categories = {
            "physical": ["physical", "contact"],
            "special": ["special", "non-contact"],
            "status": ["status", "non-damaging"],
            "priority": ["priority", "quick"],
            "setup": ["stat-boost", "setup"],
            "recovery": ["healing", "recovery"],
            "utility": ["utility", "support"]
        }
        
        # Type effectiveness chart
        self.type_effectiveness = {
            "normal": {
                "super_effective": [],
                "not_very_effective": ["rock", "steel"],
                "no_effect": ["ghost"]
            },
            "fire": {
                "super_effective": ["grass", "ice", "bug", "steel"],
                "not_very_effective": ["fire", "water", "rock", "dragon"],
                "no_effect": []
            },
            "water": {
                "super_effective": ["fire", "ground", "rock"],
                "not_very_effective": ["water", "grass", "dragon"],
                "no_effect": []
            },
            "electric": {
                "super_effective": ["water", "flying"],
                "not_very_effective": ["electric", "grass", "dragon"],
                "no_effect": ["ground"]
            },
            "grass": {
                "super_effective": ["water", "ground", "rock"],
                "not_very_effective": ["fire", "grass", "poison", "flying", "bug", "dragon", "steel"],
                "no_effect": []
            },
            "ice": {
                "super_effective": ["grass", "ground", "flying", "dragon"],
                "not_very_effective": ["fire", "water", "ice", "steel"],
                "no_effect": []
            },
            "fighting": {
                "super_effective": ["normal", "ice", "rock", "dark", "steel"],
                "not_very_effective": ["poison", "flying", "psychic", "bug", "fairy"],
                "no_effect": ["ghost"]
            },
            "poison": {
                "super_effective": ["grass", "fairy"],
                "not_very_effective": ["poison", "ground", "rock", "ghost"],
                "no_effect": ["steel"]
            },
            "ground": {
                "super_effective": ["fire", "electric", "poison", "rock", "steel"],
                "not_very_effective": ["grass", "bug"],
                "no_effect": ["flying"]
            },
            "flying": {
                "super_effective": ["electric", "grass", "fighting", "bug"],
                "not_very_effective": ["electric", "rock", "steel"],
                "no_effect": []
            },
            "psychic": {
                "super_effective": ["fighting", "poison"],
                "not_very_effective": ["psychic", "steel"],
                "no_effect": ["dark"]
            },
            "bug": {
                "super_effective": ["grass", "psychic", "dark"],
                "not_very_effective": ["fire", "fighting", "poison", "flying", "ghost", "steel", "fairy"],
                "no_effect": []
            },
            "rock": {
                "super_effective": ["fire", "ice", "flying", "bug"],
                "not_very_effective": ["fighting", "ground", "steel"],
                "no_effect": []
            },
            "ghost": {
                "super_effective": ["psychic", "ghost"],
                "not_very_effective": ["dark"],
                "no_effect": ["normal"]
            },
            "dragon": {
                "super_effective": ["dragon"],
                "not_very_effective": ["steel"],
                "no_effect": ["fairy"]
            },
            "dark": {
                "super_effective": ["psychic", "ghost"],
                "not_very_effective": ["fighting", "dark", "fairy"],
                "no_effect": []
            },
            "steel": {
                "super_effective": ["ice", "rock", "fairy"],
                "not_very_effective": ["fire", "water", "electric", "steel"],
                "no_effect": []
            },
            "fairy": {
                "super_effective": ["fighting", "dragon", "dark"],
                "not_very_effective": ["fire", "poison", "steel"],
                "no_effect": []
            }
        }
    
    async def _initialize(self) -> None:
        """Initialize analysis resources."""
        self.logger.info("Pokemon Analysis Agent initialized")
    
    async def process_task(self, task: Task) -> Dict[str, Any]:
        """Process Pokemon analysis tasks."""
        task_type = task.task_type
        
        if task_type not in self.analysis_handlers:
            raise ValueError(f"Unknown analysis task type: {task_type}")
        
        handler = self.analysis_handlers[task_type]
        
        try:
            result = await handler(task)
            
            # Add metadata to result
            result.update({
                "agent_id": self.config.agent_id,
                "task_id": str(task.id),
                "timestamp": datetime.now().isoformat(),
                "analysis_type": "pokemon_analysis"
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Analysis task failed: {e}")
            raise
    
    async def _handle_pokemon_analysis(self, task: Task) -> Dict[str, Any]:
        """Handle comprehensive Pokemon analysis."""
        params = task.metadata.get("task_parameters", {})
        pokemon_name = params.get("pokemon_name")
        game_version = params.get("game_version", "general")
        analysis_depth = params.get("analysis_depth", "standard")
        competitive_analysis = params.get("competitive_analysis", False)
        
        if not pokemon_name:
            raise ValueError("Pokemon name required for analysis")
        
        # This would normally receive Pokemon data from the research agent
        # For now, we'll simulate having the data
        pokemon_data = params.get("pokemon_data", {})
        
        if not pokemon_data:
            return {
                "type": "pokemon_analysis",
                "error": "No Pokemon data provided for analysis",
                "pokemon_name": pokemon_name,
                "confidence": 0.0
            }
        
        # Perform comprehensive analysis
        stat_analysis = self._analyze_pokemon_stats(pokemon_data)
        type_analysis = self._analyze_pokemon_types(pokemon_data)
        ability_analysis = self._analyze_pokemon_abilities(pokemon_data)
        movepool_analysis = self._analyze_pokemon_movepool(pokemon_data)
        
        # Generate overall assessment
        overall_assessment = await self._generate_pokemon_assessment(
            pokemon_name, stat_analysis, type_analysis, ability_analysis, movepool_analysis
        )
        
        # Add competitive analysis if requested
        competitive_assessment = {}
        if competitive_analysis:
            competitive_assessment = self._analyze_competitive_viability(
                pokemon_data, stat_analysis, type_analysis
            )
        
        return {
            "type": "pokemon_analysis",
            "pokemon_name": pokemon_name,
            "game_version": game_version,
            "stat_analysis": stat_analysis,
            "type_analysis": type_analysis,
            "ability_analysis": ability_analysis,
            "movepool_analysis": movepool_analysis,
            "overall_assessment": overall_assessment,
            "competitive_assessment": competitive_assessment,
            "analysis_depth": analysis_depth,
            "confidence": 0.9,
            "sources": ["stat_calculation", "type_chart", "movepool_analysis"]
        }
    
    def _analyze_pokemon_stats(self, pokemon_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Pokemon's base stats."""
        stats = pokemon_data.get("stats", [])
        if not stats:
            return {"error": "No stat data available"}
        
        # Extract base stats
        base_stats = {}
        for stat in stats:
            stat_name = stat["stat"]["name"]
            base_value = stat["base_stat"]
            base_stats[stat_name] = base_value
        
        # Calculate total and averages
        total_stats = sum(base_stats.values())
        average_stat = total_stats / len(base_stats)
        
        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []
        
        for stat_name, value in base_stats.items():
            benchmarks = self.stat_benchmarks.get(stat_name, {})
            
            if value >= benchmarks.get("excellent", 120):
                strengths.append({"stat": stat_name, "value": value, "rating": "excellent"})
            elif value >= benchmarks.get("high", 100):
                strengths.append({"stat": stat_name, "value": value, "rating": "high"})
            elif value <= benchmarks.get("low", 60):
                weaknesses.append({"stat": stat_name, "value": value, "rating": "low"})
        
        # Determine stat distribution type
        stat_values = list(base_stats.values())
        stat_std = statistics.stdev(stat_values) if len(stat_values) > 1 else 0
        
        if stat_std < 15:
            distribution_type = "balanced"
        elif stat_std < 25:
            distribution_type = "moderately_specialized"
        else:
            distribution_type = "highly_specialized"
        
        # Identify primary role based on stats
        primary_role = self._determine_primary_role(base_stats)
        
        return {
            "base_stats": base_stats,
            "total_stats": total_stats,
            "average_stat": round(average_stat, 1),
            "stat_distribution": distribution_type,
            "primary_role": primary_role,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "stat_ratings": self._rate_all_stats(base_stats)
        }
    
    def _determine_primary_role(self, base_stats: Dict[str, int]) -> str:
        """Determine Pokemon's primary role based on stats."""
        attack = base_stats.get("attack", 0)
        sp_attack = base_stats.get("special-attack", 0)
        defense = base_stats.get("defense", 0)
        sp_defense = base_stats.get("special-defense", 0)
        speed = base_stats.get("speed", 0)
        hp = base_stats.get("hp", 0)
        
        # Determine role based on highest stats
        if max(attack, sp_attack) >= 100:
            if speed >= 100:
                return "sweeper"
            elif max(attack, sp_attack) >= 120:
                return "wallbreaker"
            else:
                return "attacker"
        elif max(defense, sp_defense) >= 100:
            if hp >= 100:
                return "tank"
            else:
                return "wall"
        elif speed >= 110:
            return "support/utility"
        else:
            return "balanced"
    
    def _rate_all_stats(self, base_stats: Dict[str, int]) -> Dict[str, str]:
        """Rate all stats according to benchmarks."""
        ratings = {}
        
        for stat_name, value in base_stats.items():
            benchmarks = self.stat_benchmarks.get(stat_name, {})
            
            if value >= benchmarks.get("excellent", 120):
                ratings[stat_name] = "excellent"
            elif value >= benchmarks.get("high", 100):
                ratings[stat_name] = "high"
            elif value >= benchmarks.get("average", 80):
                ratings[stat_name] = "average"
            elif value >= benchmarks.get("low", 60):
                ratings[stat_name] = "below_average"
            else:
                ratings[stat_name] = "poor"
        
        return ratings
    
    def _analyze_pokemon_types(self, pokemon_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Pokemon's typing."""
        types = pokemon_data.get("types", [])
        if not types:
            return {"error": "No type data available"}
        
        type_names = [t["type"]["name"] for t in types]
        
        # Calculate type effectiveness
        offensive_effectiveness = self._calculate_offensive_effectiveness(type_names)
        defensive_effectiveness = self._calculate_defensive_effectiveness(type_names)
        
        # Analyze type synergy
        type_synergy = self._analyze_type_synergy(type_names)
        
        return {
            "types": type_names,
            "type_count": len(type_names),
            "offensive_effectiveness": offensive_effectiveness,
            "defensive_effectiveness": defensive_effectiveness,
            "type_synergy": type_synergy,
            "stab_types": type_names  # Same-Type Attack Bonus
        }
    
    def _calculate_offensive_effectiveness(self, types: List[str]) -> Dict[str, Any]:
        """Calculate offensive type effectiveness."""
        super_effective_against = set()
        not_very_effective_against = set()
        no_effect_against = set()
        
        for type_name in types:
            type_data = self.type_effectiveness.get(type_name, {})
            super_effective_against.update(type_data.get("super_effective", []))
            not_very_effective_against.update(type_data.get("not_very_effective", []))
            no_effect_against.update(type_data.get("no_effect", []))
        
        # Remove overlaps
        effective_super = super_effective_against - not_very_effective_against
        effective_resisted = not_very_effective_against - super_effective_against
        
        return {
            "super_effective_against": list(effective_super),
            "not_very_effective_against": list(effective_resisted),
            "no_effect_against": list(no_effect_against),
            "coverage_score": len(effective_super) - len(effective_resisted) * 0.5
        }
    
    def _calculate_defensive_effectiveness(self, types: List[str]) -> Dict[str, Any]:
        """Calculate defensive type effectiveness."""
        weaknesses = set()
        resistances = set()
        immunities = set()
        
        # Calculate what types are effective against this Pokemon
        for attacking_type, effectiveness in self.type_effectiveness.items():
            multiplier = 1.0
            
            for defending_type in types:
                if defending_type in effectiveness.get("super_effective", []):
                    multiplier *= 2.0
                elif defending_type in effectiveness.get("not_very_effective", []):
                    multiplier *= 0.5
                elif defending_type in effectiveness.get("no_effect", []):
                    multiplier = 0.0
                    break
            
            if multiplier > 1.0:
                weaknesses.add(attacking_type)
            elif multiplier < 1.0 and multiplier > 0:
                resistances.add(attacking_type)
            elif multiplier == 0:
                immunities.add(attacking_type)
        
        # Calculate defensive rating
        defensive_score = len(resistances) * 0.5 + len(immunities) - len(weaknesses)
        
        return {
            "weaknesses": list(weaknesses),
            "resistances": list(resistances),
            "immunities": list(immunities),
            "defensive_score": defensive_score,
            "weakness_count": len(weaknesses),
            "resistance_count": len(resistances)
        }
    
    def _analyze_type_synergy(self, types: List[str]) -> Dict[str, Any]:
        """Analyze synergy between types."""
        if len(types) == 1:
            return {
                "synergy_rating": "single_type",
                "synergy_description": "Single typing - no internal synergy",
                "coverage_gaps": "May need coverage moves for type weaknesses"
            }
        
        # Analyze dual-type synergy
        type1, type2 = types[0], types[1]
        
        # Check if types complement each other offensively
        type1_coverage = set(self.type_effectiveness.get(type1, {}).get("super_effective", []))
        type2_coverage = set(self.type_effectiveness.get(type2, {}).get("super_effective", []))
        
        combined_coverage = type1_coverage.union(type2_coverage)
        coverage_overlap = type1_coverage.intersection(type2_coverage)
        
        synergy_score = len(combined_coverage) - len(coverage_overlap) * 0.3
        
        if synergy_score >= 8:
            synergy_rating = "excellent"
        elif synergy_score >= 6:
            synergy_rating = "good"
        elif synergy_score >= 4:
            synergy_rating = "average"
        else:
            synergy_rating = "poor"
        
        return {
            "synergy_rating": synergy_rating,
            "combined_coverage": list(combined_coverage),
            "coverage_overlap": list(coverage_overlap),
            "synergy_score": round(synergy_score, 1),
            "synergy_description": f"{type1.title()}/{type2.title()} typing provides {synergy_rating} offensive synergy"
        }
    
    def _analyze_pokemon_abilities(self, pokemon_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Pokemon's abilities."""
        abilities = pokemon_data.get("abilities", [])
        if not abilities:
            return {"error": "No ability data available"}
        
        ability_analysis = []
        
        for ability in abilities:
            ability_name = ability["ability"]["name"]
            is_hidden = ability.get("is_hidden", False)
            
            # Analyze ability impact (simplified)
            impact_rating = self._rate_ability_impact(ability_name)
            
            ability_info = {
                "name": ability_name,
                "is_hidden": is_hidden,
                "impact_rating": impact_rating,
                "description": f"Analysis of {ability_name} ability",
                "competitive_viability": self._assess_ability_competitive_viability(ability_name)
            }
            
            ability_analysis.append(ability_info)
        
        # Determine best ability
        best_ability = max(ability_analysis, key=lambda x: self._ability_score(x["impact_rating"]))
        
        return {
            "abilities": ability_analysis,
            "ability_count": len(abilities),
            "has_hidden_ability": any(a["is_hidden"] for a in abilities),
            "best_ability": best_ability,
            "ability_versatility": "high" if len(abilities) > 2 else "medium" if len(abilities) == 2 else "low"
        }
    
    def _rate_ability_impact(self, ability_name: str) -> str:
        """Rate the impact of an ability (simplified)."""
        # This would normally have a comprehensive ability database
        high_impact_abilities = [
            "levitate", "wonder-guard", "huge-power", "pure-power", "speed-boost",
            "drought", "drizzle", "sand-stream", "snow-warning", "intimidate"
        ]
        
        medium_impact_abilities = [
            "sturdy", "multiscale", "regenerator", "magic-guard", "technician",
            "adaptability", "skill-link", "serene-grace"
        ]
        
        if ability_name in high_impact_abilities:
            return "high"
        elif ability_name in medium_impact_abilities:
            return "medium"
        else:
            return "low"
    
    def _assess_ability_competitive_viability(self, ability_name: str) -> str:
        """Assess ability's competitive viability."""
        # Simplified assessment
        competitive_abilities = [
            "intimidate", "levitate", "regenerator", "magic-guard", "multiscale",
            "drought", "drizzle", "sand-stream", "speed-boost"
        ]
        
        if ability_name in competitive_abilities:
            return "high"
        else:
            return "situational"
    
    def _ability_score(self, impact_rating: str) -> int:
        """Convert impact rating to numeric score."""
        scores = {"high": 3, "medium": 2, "low": 1}
        return scores.get(impact_rating, 1)
    
    def _analyze_pokemon_movepool(self, pokemon_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Pokemon's movepool."""
        moves = pokemon_data.get("moves", [])
        if not moves:
            return {"error": "No move data available"}
        
        # Categorize moves
        move_categories = {
            "physical": 0,
            "special": 0,
            "status": 0,
            "priority": 0,
            "setup": 0,
            "recovery": 0,
            "utility": 0
        }
        
        # This is simplified - would normally analyze each move
        total_moves = len(moves)
        
        # Estimate move distribution (simplified)
        move_categories["physical"] = int(total_moves * 0.4)
        move_categories["special"] = int(total_moves * 0.3)
        move_categories["status"] = int(total_moves * 0.2)
        move_categories["utility"] = total_moves - sum(move_categories.values())
        
        # Calculate movepool versatility
        versatility_score = min(total_moves / 50, 1.0)  # Normalize to 0-1
        
        if versatility_score >= 0.8:
            versatility_rating = "excellent"
        elif versatility_score >= 0.6:
            versatility_rating = "good"
        elif versatility_score >= 0.4:
            versatility_rating = "average"
        else:
            versatility_rating = "limited"
        
        return {
            "total_moves": total_moves,
            "move_categories": move_categories,
            "versatility_rating": versatility_rating,
            "versatility_score": round(versatility_score, 2),
            "coverage_assessment": "Requires detailed move analysis",
            "notable_moves": "Requires move-by-move analysis"
        }
    
    async def _generate_pokemon_assessment(
        self,
        pokemon_name: str,
        stat_analysis: Dict[str, Any],
        type_analysis: Dict[str, Any],
        ability_analysis: Dict[str, Any],
        movepool_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate overall Pokemon assessment using LLM."""
        
        assessment_prompt = f"""
        Provide a comprehensive assessment of {pokemon_name} based on the following analysis:
        
        Stats: {stat_analysis.get('primary_role', 'unknown')} role, {stat_analysis.get('total_stats', 0)} BST
        Typing: {'/'.join(type_analysis.get('types', []))}
        Key Strengths: {[s['stat'] for s in stat_analysis.get('strengths', [])]}
        Key Weaknesses: {[w['stat'] for w in stat_analysis.get('weaknesses', [])]}
        
        Provide:
        1. Overall viability rating (S/A/B/C/D tier)
        2. Best use cases
        3. Main strengths and weaknesses
        4. Recommended roles
        5. Training difficulty (beginner/intermediate/advanced)
        
        Keep the assessment concise but informative.
        """
        
        try:
            response = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are a Pokemon analysis expert."},
                    {"role": "user", "content": assessment_prompt}
                ],
                temperature=0.6
            )
            
            assessment_text = response["choices"][0]["message"]["content"]
            
            return {
                "overall_assessment": assessment_text,
                "viability_tier": self._extract_tier_from_assessment(assessment_text),
                "recommended_roles": stat_analysis.get("primary_role", "balanced"),
                "training_difficulty": self._assess_training_difficulty(stat_analysis, type_analysis),
                "summary": f"{pokemon_name} is a {stat_analysis.get('primary_role', 'balanced')} Pokemon with {type_analysis.get('type_count', 1)} type(s)"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate assessment: {e}")
            return {
                "overall_assessment": "Assessment generation failed",
                "viability_tier": "C",
                "recommended_roles": stat_analysis.get("primary_role", "balanced"),
                "training_difficulty": "intermediate",
                "summary": f"Basic analysis of {pokemon_name}"
            }
    
    def _extract_tier_from_assessment(self, assessment_text: str) -> str:
        """Extract tier rating from assessment text."""
        assessment_lower = assessment_text.lower()
        
        if "s tier" in assessment_lower or "s-tier" in assessment_lower:
            return "S"
        elif "a tier" in assessment_lower or "a-tier" in assessment_lower:
            return "A"
        elif "b tier" in assessment_lower or "b-tier" in assessment_lower:
            return "B"
        elif "c tier" in assessment_lower or "c-tier" in assessment_lower:
            return "C"
        elif "d tier" in assessment_lower or "d-tier" in assessment_lower:
            return "D"
        else:
            return "B"  # Default
    
    def _assess_training_difficulty(self, stat_analysis: Dict[str, Any], type_analysis: Dict[str, Any]) -> str:
        """Assess training difficulty based on stats and typing."""
        total_stats = stat_analysis.get("total_stats", 0)
        weakness_count = type_analysis.get("defensive_effectiveness", {}).get("weakness_count", 0)
        
        if total_stats >= 600:
            return "advanced"  # Legendary-tier
        elif total_stats >= 500 and weakness_count <= 2:
            return "intermediate"
        elif weakness_count >= 4:
            return "advanced"  # Difficult due to many weaknesses
        else:
            return "beginner"
    
    def _analyze_competitive_viability(
        self,
        pokemon_data: Dict[str, Any],
        stat_analysis: Dict[str, Any],
        type_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze competitive viability."""
        total_stats = stat_analysis.get("total_stats", 0)
        primary_role = stat_analysis.get("primary_role", "balanced")
        weakness_count = type_analysis.get("defensive_effectiveness", {}).get("weakness_count", 0)
        
        # Calculate competitive score
        competitive_score = 0
        
        # Stat total contribution
        if total_stats >= 600:
            competitive_score += 40
        elif total_stats >= 500:
            competitive_score += 30
        elif total_stats >= 450:
            competitive_score += 20
        else:
            competitive_score += 10
        
        # Role clarity contribution
        if primary_role in ["sweeper", "wallbreaker", "tank"]:
            competitive_score += 20
        elif primary_role in ["attacker", "wall"]:
            competitive_score += 15
        else:
            competitive_score += 10
        
        # Defensive typing contribution
        if weakness_count <= 1:
            competitive_score += 20
        elif weakness_count <= 2:
            competitive_score += 15
        elif weakness_count <= 3:
            competitive_score += 10
        else:
            competitive_score += 0
        
        # Movepool and ability (simplified)
        competitive_score += 20  # Placeholder
        
        # Determine tier
        if competitive_score >= 85:
            tier = "OU"
        elif competitive_score >= 70:
            tier = "UU"
        elif competitive_score >= 55:
            tier = "RU"
        elif competitive_score >= 40:
            tier = "NU"
        else:
            tier = "PU"
        
        return {
            "competitive_score": competitive_score,
            "estimated_tier": tier,
            "viability_factors": {
                "stat_total": total_stats,
                "role_clarity": primary_role,
                "defensive_typing": f"{weakness_count} weaknesses",
                "movepool_quality": "requires_analysis"
            },
            "competitive_roles": self._suggest_competitive_roles(primary_role, stat_analysis),
            "meta_relevance": "requires_current_meta_analysis"
        }
    
    def _suggest_competitive_roles(self, primary_role: str, stat_analysis: Dict[str, Any]) -> List[str]:
        """Suggest competitive roles based on analysis."""
        base_stats = stat_analysis.get("base_stats", {})
        
        roles = []
        
        if primary_role == "sweeper":
            if base_stats.get("speed", 0) >= 100:
                roles.append("Setup Sweeper")
            if base_stats.get("attack", 0) >= 100:
                roles.append("Physical Sweeper")
            if base_stats.get("special-attack", 0) >= 100:
                roles.append("Special Sweeper")
        
        elif primary_role == "tank":
            roles.extend(["Defensive Pivot", "Status Absorber", "Hazard Setter"])
        
        elif primary_role == "wall":
            if base_stats.get("defense", 0) >= 100:
                roles.append("Physical Wall")
            if base_stats.get("special-defense", 0) >= 100:
                roles.append("Special Wall")
        
        else:
            roles.append("Utility Pokemon")
        
        return roles if roles else ["Support"]
    
    async def _handle_stat_analysis(self, task: Task) -> Dict[str, Any]:
        """Handle detailed stat analysis."""
        params = task.metadata.get("task_parameters", {})
        pokemon_data = params.get("pokemon_data", {})
        
        if not pokemon_data:
            return {"error": "No Pokemon data provided", "confidence": 0.0}
        
        stat_analysis = self._analyze_pokemon_stats(pokemon_data)
        
        # Add detailed stat calculations
        detailed_analysis = self._calculate_detailed_stats(pokemon_data)
        
        return {
            "type": "stat_analysis",
            "basic_analysis": stat_analysis,
            "detailed_analysis": detailed_analysis,
            "confidence": 1.0,
            "sources": ["base_stat_calculation"]
        }
    
    def _calculate_detailed_stats(self, pokemon_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate detailed stat information."""
        stats = pokemon_data.get("stats", [])
        base_stats = {stat["stat"]["name"]: stat["base_stat"] for stat in stats}
        
        # Calculate stats at different levels (simplified)
        level_50_stats = {}
        level_100_stats = {}
        
        for stat_name, base_value in base_stats.items():
            if stat_name == "hp":
                # HP calculation
                level_50_stats[stat_name] = int((2 * base_value + 31 + 63) * 50 / 100) + 10
                level_100_stats[stat_name] = int((2 * base_value + 31 + 252) * 100 / 100) + 10
            else:
                # Other stats calculation
                level_50_stats[stat_name] = int(((2 * base_value + 31 + 63) * 50 / 100) + 5)
                level_100_stats[stat_name] = int(((2 * base_value + 31 + 252) * 100 / 100) + 5)
        
        return {
            "base_stats": base_stats,
            "level_50_stats": level_50_stats,
            "level_100_stats": level_100_stats,
            "stat_ranges": self._calculate_stat_ranges(base_stats)
        }
    
    def _calculate_stat_ranges(self, base_stats: Dict[str, int]) -> Dict[str, Dict[str, int]]:
        """Calculate min/max stat ranges."""
        ranges = {}
        
        for stat_name, base_value in base_stats.items():
            if stat_name == "hp":
                min_stat = int((2 * base_value) * 100 / 100) + 10
                max_stat = int((2 * base_value + 31 + 252) * 100 / 100) + 10
            else:
                min_stat = int(((2 * base_value) * 100 / 100) + 5) * 0.9  # With hindering nature
                max_stat = int(((2 * base_value + 31 + 252) * 100 / 100) + 5) * 1.1  # With beneficial nature
            
            ranges[stat_name] = {
                "min": int(min_stat),
                "max": int(max_stat)
            }
        
        return ranges
    
    async def _handle_comparative_analysis(self, task: Task) -> Dict[str, Any]:
        """Handle comparative analysis between Pokemon."""
        params = task.metadata.get("task_parameters", {})
        pokemon_list = params.get("pokemon_list", [])
        comparison_criteria = params.get("comparison_criteria", ["stats", "abilities"])
        
        if len(pokemon_list) < 2:
            return {"error": "At least 2 Pokemon required for comparison", "confidence": 0.0}
        
        # This would normally receive Pokemon data for all Pokemon
        # For now, we'll provide a structured comparison framework
        
        comparison_results = {
            "pokemon_compared": pokemon_list,
            "criteria": comparison_criteria,
            "stat_comparison": self._compare_stats(pokemon_list),
            "type_comparison": self._compare_types(pokemon_list),
            "role_comparison": self._compare_roles(pokemon_list),
            "recommendations": await self._generate_comparison_recommendations(pokemon_list)
        }
        
        return {
            "type": "comparative_analysis",
            "comparison_results": comparison_results,
            "confidence": 0.8,
            "sources": ["stat_analysis", "type_analysis"]
        }
    
    def _compare_stats(self, pokemon_list: List[str]) -> Dict[str, Any]:
        """Compare stats between Pokemon (simplified)."""
        return {
            "comparison_type": "stat_comparison",
            "pokemon": pokemon_list,
            "note": "Detailed stat comparison requires Pokemon data",
            "framework": {
                "total_stats": "Compare base stat totals",
                "stat_distribution": "Compare stat allocation",
                "role_optimization": "Compare role-specific stats"
            }
        }
    
    def _compare_types(self, pokemon_list: List[str]) -> Dict[str, Any]:
        """Compare typing between Pokemon."""
        return {
            "comparison_type": "type_comparison",
            "pokemon": pokemon_list,
            "analysis_points": [
                "Offensive type coverage",
                "Defensive resistances and weaknesses",
                "STAB move availability",
                "Type synergy in team context"
            ]
        }
    
    def _compare_roles(self, pokemon_list: List[str]) -> Dict[str, Any]:
        """Compare roles between Pokemon."""
        return {
            "comparison_type": "role_comparison",
            "pokemon": pokemon_list,
            "role_categories": [
                "Primary battle role",
                "Secondary capabilities",
                "Team support functions",
                "Niche applications"
            ]
        }
    
    async def _generate_comparison_recommendations(self, pokemon_list: List[str]) -> List[str]:
        """Generate recommendations from comparison."""
        
        prompt = f"""
        Provide 3-5 recommendations for choosing between these Pokemon: {', '.join(pokemon_list)}
        
        Consider:
        - Different use cases and team roles
        - Strengths and weaknesses of each
        - Situational advantages
        - Beginner vs advanced player considerations
        
        Make recommendations specific and actionable.
        """
        
        try:
            response = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are a Pokemon comparison expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6
            )
            
            recommendations_text = response["choices"][0]["message"]["content"]
            recommendations = [line.strip() for line in recommendations_text.split('\n') if line.strip() and not line.strip().startswith('#')]
            
            return recommendations[:5]
            
        except Exception as e:
            self.logger.error(f"Failed to generate comparison recommendations: {e}")
            return [
                "Consider your team's current needs",
                "Evaluate type coverage requirements",
                "Match Pokemon to your playstyle",
                "Consider availability and training requirements"
            ]
    
    async def _handle_move_analysis(self, task: Task) -> Dict[str, Any]:
        """Handle move analysis."""
        params = task.metadata.get("task_parameters", {})
        
        return {
            "type": "move_analysis",
            "note": "Move analysis requires detailed move data",
            "framework": "Comprehensive move analysis system",
            "confidence": 0.5
        }
    
    async def _handle_ability_analysis(self, task: Task) -> Dict[str, Any]:
        """Handle ability analysis."""
        params = task.metadata.get("task_parameters", {})
        
        return {
            "type": "ability_analysis",
            "note": "Ability analysis requires detailed ability data",
            "framework": "Comprehensive ability analysis system",
            "confidence": 0.5
        }
    
    async def _handle_type_effectiveness_analysis(self, task: Task) -> Dict[str, Any]:
        """Handle type effectiveness analysis."""
        params = task.metadata.get("task_parameters", {})
        
        return {
            "type": "type_effectiveness_analysis",
            "note": "Type effectiveness analysis system",
            "framework": "Complete type chart analysis",
            "confidence": 0.8
        }
    
    async def _handle_training_analysis(self, task: Task) -> Dict[str, Any]:
        """Handle training analysis."""
        params = task.metadata.get("task_parameters", {})
        
        return {
            "type": "training_analysis",
            "note": "Training analysis for ease of use",
            "framework": "Beginner-friendly assessment",
            "confidence": 0.7
        }
    
    async def _handle_viability_assessment(self, task: Task) -> Dict[str, Any]:
        """Handle viability assessment."""
        params = task.metadata.get("task_parameters", {})
        
        return {
            "type": "viability_assessment",
            "note": "Competitive viability assessment",
            "framework": "Multi-factor viability analysis",
            "confidence": 0.8
        }
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get analysis agent statistics."""
        status = self.get_status()
        status.update({
            "analysis_handlers": list(self.analysis_handlers.keys()),
            "stat_benchmarks": len(self.stat_benchmarks),
            "type_effectiveness_entries": len(self.type_effectiveness),
            "move_categories": len(self.move_categories)
        })
        return status

