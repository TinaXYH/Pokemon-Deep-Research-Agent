#!/usr/bin/env python3
"""
Pokemon Deep Research Agent - Clean Interactive Demo
Professional version with minimal emojis and clean output formatting
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import logging

from src.core.conversation_memory import ConversationMemoryManager
from src.core.query_disambiguator import QueryDisambiguator
from src.core.specification_manager import (SpecificationManager,
                                            TaskSpecification)
from src.tools.llm_client import LLMClient
from src.tools.pokeapi_client import PokéAPIClient

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class CleanPokemonResearcher:
    """
    Pokemon researcher with clean, professional output formatting.

    Features:
    - Minimal emoji usage
    - Clean text-based formatting
    - Professional interaction flow
    - Specification confirmation
    - Conversation memory
    """

    def __init__(self, api_key: str):
        self.llm_client = LLMClient(
            api_key=api_key, model="gpt-4.1-mini", max_tokens=4000, temperature=0.7
        )

        self.pokeapi_client = PokéAPIClient(
            base_url="https://pokeapi.co/api/v2/",
            cache_dir="data/cache",
            rate_limit_delay=0.1,
            max_concurrent_requests=10,
        )

        # Initialize specification and memory systems
        self.memory_manager = ConversationMemoryManager()
        self.spec_manager = SpecificationManager(self.llm_client)
        self.disambiguator = QueryDisambiguator(self.spec_manager, self.llm_client)

        self.current_conversation_id = None

    def start_new_session(self) -> str:
        """Start a new conversation session."""
        self.current_conversation_id = self.memory_manager.start_new_conversation()
        return self.current_conversation_id

    async def process_query_with_spec_confirmation(self, query: str) -> Dict[str, Any]:
        """
        Process Pokemon query with specification confirmation workflow.
        """

        if not self.current_conversation_id:
            self.start_new_session()

        print(f"\n" + "=" * 80)
        print(f"POKEMON DEEP RESEARCH AGENT - SPECIFICATION CONFIRMATION")
        print(f"=" * 80)
        print(f"Query: {query}")

        # Get conversation context
        conversation_context = self.memory_manager.get_persistent_context(
            self.current_conversation_id
        )

        if conversation_context.get("mentioned_pokemon"):
            print(
                f"Context: Previously discussed {', '.join(conversation_context['mentioned_pokemon'])}"
            )

        try:
            # PHASE 1: Query Analysis & Specification Generation
            print(f"\n[PHASE 1] Query Analysis & Specification Generation")
            print(f"   > Analyzing query and generating research specification...")

            spec = await self.spec_manager.analyze_query_and_generate_spec(
                query, conversation_context
            )

            print(
                f"   > Query classified as: {spec.query_type.value.replace('_', ' ').title()}"
            )
            print(f"   > Initial specification generated")

            # Auto-complete from conversation context
            spec = await self.disambiguator.auto_complete_from_context(
                spec, conversation_context
            )

            # PHASE 2: Interactive Specification Confirmation
            print(f"\n[PHASE 2] Interactive Specification Confirmation")

            confirmed_spec = await self.disambiguator.disambiguate_query_interactive(
                spec, conversation_context
            )

            if not confirmed_spec or not confirmed_spec.confirmed:
                return {"status": "cancelled", "message": "Research cancelled by user"}

            # PHASE 3: Deep Research Execution
            print(f"\n[PHASE 3] Deep Research Execution")
            print(f"   > Executing deep research with confirmed specification...")

            research_result = await self._execute_deep_research(
                confirmed_spec, conversation_context
            )

            if "error" in research_result:
                return research_result

            # PHASE 4: Report Generation
            print(f"\n[PHASE 4] Report Generation")
            print(f"   > Generating comprehensive report with conversation context...")

            final_report = await self._generate_spec_aware_report(
                confirmed_spec, research_result, conversation_context
            )

            print(f"   > Comprehensive report generated")

            # Save to conversation memory
            self.memory_manager.add_turn(
                conversation_id=self.current_conversation_id,
                user_input=query,
                agent_response=final_report,
                pokemon_mentioned=self._extract_pokemon_from_spec(confirmed_spec),
                query_type=confirmed_spec.query_type.value,
                context_used={
                    "specification": confirmed_spec.to_dict(),
                    "research_type": "deep_research_with_spec",
                },
            )

            return {
                "query": query,
                "conversation_id": self.current_conversation_id,
                "specification": confirmed_spec.to_dict(),
                "research_result": research_result,
                "final_report": final_report,
                "status": "completed",
            }

        except Exception as e:
            print(f"   ERROR: {e}")
            return {"error": str(e)}

    async def _execute_deep_research(
        self, spec: TaskSpecification, conversation_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute deep research based on confirmed specification."""

        query_type = spec.query_type
        fields = spec.fields

        if query_type.value == "competitive_analysis":
            return await self._execute_competitive_analysis(
                fields, conversation_context
            )

        elif query_type.value == "team_building":
            return await self._execute_team_building(fields, conversation_context)

        elif query_type.value == "pokemon_comparison":
            return await self._execute_pokemon_comparison(fields, conversation_context)

        elif query_type.value == "moveset_analysis":
            return await self._execute_moveset_analysis(fields, conversation_context)

        elif query_type.value == "meta_analysis":
            return await self._execute_meta_analysis(fields, conversation_context)

        else:
            return await self._execute_general_research(fields, conversation_context)

    async def _execute_competitive_analysis(
        self, fields: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute competitive analysis research."""

        pokemon_name = fields.get("target_pokemon")
        format_name = fields.get("format", "OU")
        analysis_depth = fields.get("analysis_depth", "detailed")
        include_counters = fields.get("include_counters", True)

        print(f"   > Analyzing {pokemon_name.title()} for {format_name} format")

        # Get Pokemon data
        pokemon_data = await self.pokeapi_client.get_pokemon(pokemon_name.lower())
        if not pokemon_data:
            return {"error": f"Pokemon {pokemon_name} not found"}

        print(f"   > Retrieved {pokemon_data.name.title()} data")
        print(
            f"     - Base stats: {[stat['base_stat'] for stat in pokemon_data.stats]}"
        )
        print(f"     - Types: {[t['type']['name'] for t in pokemon_data.types]}")

        # Generate comprehensive analysis
        analysis_prompt = f"""
        Perform a {analysis_depth} competitive analysis of {pokemon_data.name.title()} for {format_name} format.
        
        Pokemon Data:
        - Base Stats: {self._format_stats(pokemon_data.stats)}
        - Types: {', '.join([t['type']['name'] for t in pokemon_data.types])}
        - Abilities: {', '.join([a['ability']['name'] for a in pokemon_data.abilities])}
        
        Analysis Requirements:
        - Format: {format_name}
        - Depth: {analysis_depth}
        - Include Counters: {include_counters}
        
        Provide comprehensive analysis including:
        1. Tier placement and viability in {format_name}
        2. Role analysis and team positioning
        3. Stat analysis and optimization
        4. Common movesets and items
        5. Strengths and advantages
        6. Weaknesses and vulnerabilities
        {"7. Counter Pokemon and strategies" if include_counters else ""}
        8. Team synergies and partners
        9. Usage recommendations
        
        Be specific to {format_name} format and provide actionable insights.
        """

        response = await self.llm_client.chat_completion(
            messages=[{"role": "user", "content": analysis_prompt}], temperature=0.3
        )

        analysis = response["choices"][0]["message"]["content"]

        return {
            "type": "competitive_analysis",
            "pokemon": pokemon_data.name.title(),
            "format": format_name,
            "pokemon_data": pokemon_data,
            "analysis": analysis,
            "specification": fields,
        }

    async def _execute_team_building(
        self, fields: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute team building research."""

        core_pokemon = fields.get("core_pokemon")
        format_name = fields.get("format", "OU")
        team_style = fields.get("team_style", "balanced")
        include_legendaries = fields.get("include_legendaries", False)
        hazard_support = fields.get("hazard_support", True)
        hazard_removal = fields.get("hazard_removal", True)

        print(
            f"   > Building {team_style} team around {core_pokemon.title()} for {format_name}"
        )

        # Get core Pokemon data
        pokemon_data = await self.pokeapi_client.get_pokemon(core_pokemon.lower())
        if not pokemon_data:
            return {"error": f"Pokemon {core_pokemon} not found"}

        print(f"   > Core Pokemon: {pokemon_data.name.title()}")

        # Generate team building analysis
        team_prompt = f"""
        Build a competitive {team_style} team around {pokemon_data.name.title()} for {format_name} format.
        
        Core Pokemon: {pokemon_data.name.title()}
        - Types: {', '.join([t['type']['name'] for t in pokemon_data.types])}
        - Base Stats: {self._format_stats(pokemon_data.stats)}
        
        Team Requirements:
        - Format: {format_name}
        - Style: {team_style}
        - Include Legendaries: {include_legendaries}
        - Hazard Support: {hazard_support}
        - Hazard Removal: {hazard_removal}
        
        Provide comprehensive team building analysis:
        1. Core Pokemon role and set
        2. Team composition (5 additional Pokemon)
        3. Role distribution and synergies
        4. Movesets for each team member
        5. Items and abilities
        6. Team strategy and win conditions
        7. Common threats and how to handle them
        8. Alternative options and substitutions
        
        Focus on {format_name} viability and {team_style} playstyle.
        """

        response = await self.llm_client.chat_completion(
            messages=[{"role": "user", "content": team_prompt}], temperature=0.4
        )

        team_analysis = response["choices"][0]["message"]["content"]

        return {
            "type": "team_building",
            "core_pokemon": pokemon_data.name.title(),
            "format": format_name,
            "team_style": team_style,
            "pokemon_data": pokemon_data,
            "team_analysis": team_analysis,
            "specification": fields,
        }

    async def _execute_pokemon_comparison(
        self, fields: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute Pokemon comparison research."""

        pokemon_list = fields.get("pokemon_list", [])
        comparison_aspects = fields.get("comparison_aspects", "all")
        format_name = fields.get("format", "general")

        print(f"   > Comparing {', '.join([p.title() for p in pokemon_list])}")

        # Get data for all Pokemon
        pokemon_data = {}
        for pokemon_name in pokemon_list:
            data = await self.pokeapi_client.get_pokemon(pokemon_name.lower())
            if data:
                pokemon_data[pokemon_name] = data
                print(f"   > Retrieved {data.name.title()} data")

        if len(pokemon_data) < 2:
            return {"error": "Need at least 2 Pokemon for comparison"}

        # Generate comparison analysis
        comparison_prompt = f"""
        Compare these Pokemon: {', '.join([data.name.title() for data in pokemon_data.values()])}
        
        Pokemon Data:
        """

        for name, data in pokemon_data.items():
            comparison_prompt += f"""
        
        {data.name.title()}:
        - Types: {', '.join([t['type']['name'] for t in data.types])}
        - Base Stats: {self._format_stats(data.stats)}
        - Abilities: {', '.join([a['ability']['name'] for a in data.abilities])}
        """

        comparison_prompt += f"""
        
        Comparison Requirements:
        - Aspects: {comparison_aspects}
        - Format Context: {format_name}
        
        Provide detailed comparison covering:
        1. Stat comparison and analysis
        2. Type advantages and disadvantages
        3. Role comparison in competitive play
        4. Movepool and versatility comparison
        5. Competitive viability ranking
        6. Situational advantages
        7. Team building considerations
        8. Recommendation for different use cases
        
        {"Focus on " + format_name + " format context." if format_name != "general" else "Provide general competitive context."}
        """

        response = await self.llm_client.chat_completion(
            messages=[{"role": "user", "content": comparison_prompt}], temperature=0.3
        )

        comparison_analysis = response["choices"][0]["message"]["content"]

        return {
            "type": "pokemon_comparison",
            "pokemon_compared": list(pokemon_data.keys()),
            "comparison_aspects": comparison_aspects,
            "format": format_name,
            "pokemon_data": pokemon_data,
            "comparison_analysis": comparison_analysis,
            "specification": fields,
        }

    async def _execute_moveset_analysis(
        self, fields: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute moveset analysis research."""

        pokemon_name = fields.get("target_pokemon")
        format_name = fields.get("format", "OU")
        moveset_types = fields.get("moveset_types", "all")

        print(f"   > Analyzing {pokemon_name.title()} movesets for {format_name}")

        # Get Pokemon data
        pokemon_data = await self.pokeapi_client.get_pokemon(pokemon_name.lower())
        if not pokemon_data:
            return {"error": f"Pokemon {pokemon_name} not found"}

        print(f"   > Retrieved {pokemon_data.name.title()} data")

        # Generate moveset analysis
        moveset_prompt = f"""
        Analyze movesets for {pokemon_data.name.title()} in {format_name} format.
        
        Pokemon: {pokemon_data.name.title()}
        - Types: {', '.join([t['type']['name'] for t in pokemon_data.types])}
        - Base Stats: {self._format_stats(pokemon_data.stats)}
        - Abilities: {', '.join([a['ability']['name'] for a in pokemon_data.abilities])}
        
        Analysis Requirements:
        - Format: {format_name}
        - Moveset Types: {moveset_types}
        
        Provide comprehensive moveset analysis:
        1. Standard competitive movesets
        2. Alternative and niche sets
        3. Move selection rationale
        4. Item choices and synergies
        5. EV spreads and nature options
        6. Ability selection
        7. Role-specific adaptations
        8. Situational move options
        9. Set comparisons and recommendations
        
        Focus on {format_name} viability and current meta relevance.
        """

        response = await self.llm_client.chat_completion(
            messages=[{"role": "user", "content": moveset_prompt}], temperature=0.3
        )

        moveset_analysis = response["choices"][0]["message"]["content"]

        return {
            "type": "moveset_analysis",
            "pokemon": pokemon_data.name.title(),
            "format": format_name,
            "moveset_types": moveset_types,
            "pokemon_data": pokemon_data,
            "moveset_analysis": moveset_analysis,
            "specification": fields,
        }

    async def _execute_meta_analysis(
        self, fields: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute meta analysis research."""

        format_name = fields.get("format", "OU")
        analysis_focus = fields.get("analysis_focus", "comprehensive")
        time_period = fields.get("time_period", "current")

        print(f"   > Analyzing {format_name} meta - {analysis_focus} focus")

        # Generate meta analysis
        meta_prompt = f"""
        Analyze the competitive meta for {format_name} format.
        
        Analysis Requirements:
        - Format: {format_name}
        - Focus: {analysis_focus}
        - Time Period: {time_period}
        
        Provide comprehensive meta analysis:
        1. Current tier rankings and usage
        2. Dominant strategies and archetypes
        3. Meta threats and centralizing Pokemon
        4. Team building trends
        5. Common cores and synergies
        6. Emerging strategies
        7. Meta shifts and adaptations
        8. Predictions and recommendations
        
        Focus on {analysis_focus} aspects of the {format_name} meta.
        """

        response = await self.llm_client.chat_completion(
            messages=[{"role": "user", "content": meta_prompt}], temperature=0.4
        )

        meta_analysis = response["choices"][0]["message"]["content"]

        return {
            "type": "meta_analysis",
            "format": format_name,
            "analysis_focus": analysis_focus,
            "time_period": time_period,
            "meta_analysis": meta_analysis,
            "specification": fields,
        }

    async def _execute_general_research(
        self, fields: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute general Pokemon research."""

        topic = fields.get("topic", "")
        detail_level = fields.get("detail_level", "detailed")

        print(f"   > Researching: {topic}")

        # Generate general research
        research_prompt = f"""
        Research this Pokemon topic: {topic}
        
        Requirements:
        - Detail Level: {detail_level}
        - Provide comprehensive information
        - Include practical applications
        - Cover competitive relevance where applicable
        
        Provide thorough research covering all relevant aspects of the topic.
        """

        response = await self.llm_client.chat_completion(
            messages=[{"role": "user", "content": research_prompt}], temperature=0.4
        )

        research_result = response["choices"][0]["message"]["content"]

        return {
            "type": "general_research",
            "topic": topic,
            "detail_level": detail_level,
            "research_result": research_result,
            "specification": fields,
        }

    async def _generate_spec_aware_report(
        self,
        spec: TaskSpecification,
        research_result: Dict[str, Any],
        conversation_context: Dict[str, Any],
    ) -> str:
        """Generate final report that's aware of the specification and context."""

        context_text = ""
        if conversation_context.get("mentioned_pokemon"):
            context_text = f"\nConversation context: Building on previous discussion of {', '.join(conversation_context['mentioned_pokemon'])}"

        report_prompt = f"""
        Create a comprehensive research report based on this specification and research:
        
        Original Query: "{spec.original_query}"
        Research Type: {spec.query_type.value.replace('_', ' ').title()}
        Specification: {spec.fields}{context_text}
        
        Research Results:
        {research_result.get('analysis', research_result.get('team_analysis', research_result.get('comparison_analysis', research_result.get('moveset_analysis', research_result.get('meta_analysis', research_result.get('research_result', ''))))))}
        
        Generate a professional research report with:
        
        # {research_result.get('type', 'Pokemon Research').replace('_', ' ').title()} Report
        
        ## Executive Summary
        ## Research Specification
        ## Detailed Analysis
        ## Key Findings
        ## Practical Recommendations
        ## Conclusion
        
        Make it comprehensive, well-structured, and actionable.
        Reference the specification requirements throughout the report.
        """

        response = await self.llm_client.chat_completion(
            messages=[{"role": "user", "content": report_prompt}], temperature=0.2
        )

        return response["choices"][0]["message"]["content"]

    def _extract_pokemon_from_spec(self, spec: TaskSpecification) -> List[str]:
        """Extract Pokemon names from specification."""
        pokemon_list = []

        if "target_pokemon" in spec.fields:
            pokemon_list.append(spec.fields["target_pokemon"])

        if "core_pokemon" in spec.fields:
            pokemon_list.append(spec.fields["core_pokemon"])

        if "pokemon_list" in spec.fields:
            pokemon_list.extend(spec.fields["pokemon_list"])

        return pokemon_list

    def _format_stats(self, stats: List[Dict]) -> str:
        """Format Pokemon stats for display."""
        stat_names = {
            "hp": "HP",
            "attack": "Attack",
            "defense": "Defense",
            "special-attack": "Sp. Attack",
            "special-defense": "Sp. Defense",
            "speed": "Speed",
        }

        formatted = []
        for stat in stats:
            name = stat_names.get(stat["stat"]["name"], stat["stat"]["name"])
            value = stat["base_stat"]
            formatted.append(f"{name}: {value}")

        return ", ".join(formatted)

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get current conversation summary."""
        if not self.current_conversation_id:
            return {}

        return self.memory_manager.get_conversation_summary(
            self.current_conversation_id
        )

    def clear_current_session(self) -> None:
        """Clear current conversation session."""
        if self.current_conversation_id:
            self.memory_manager.clear_conversation(self.current_conversation_id)
            self.current_conversation_id = None


async def main():
    """Main interactive function with clean formatting."""

    # Load environment
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found")
        return

    researcher = CleanPokemonResearcher(api_key)

    print("POKEMON DEEP RESEARCH AGENT - PROFESSIONAL MODE")
    print("=" * 80)
    print("Features:")
    print("  * Query Analysis & Specification Confirmation")
    print("  * Conversation Memory & Follow-up Questions")
    print("  * Deep Research with Context Awareness")
    print("  * Professional Report Generation")
    print()
    print("Research Types:")
    print("  * Competitive Analysis - Single Pokemon viability analysis")
    print("  * Team Building - Build teams around core Pokemon")
    print("  * Pokemon Comparison - Compare multiple Pokemon")
    print("  * Moveset Analysis - Analyze movesets and sets")
    print("  * Meta Analysis - Analyze competitive meta and trends")
    print("  * General Research - Any Pokemon-related topic")
    print()
    print("Commands:")
    print("  * 'new session' - Start fresh conversation")
    print("  * 'summary' - Show conversation summary")
    print("  * 'clear' - Clear current session")
    print("  * 'quit' - Exit")
    print("=" * 80)

    # Start initial session
    session_id = researcher.start_new_session()
    print(f"Started new conversation session: {session_id[:8]}...")

    while True:
        try:
            query = input("\nYour Pokemon research question: ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                print("Thank you for using Pokemon Deep Research Agent!")
                break
            elif query.lower() in ["new session", "new", "reset"]:
                session_id = researcher.start_new_session()
                print(f"Started new conversation session: {session_id[:8]}...")
                continue
            elif query.lower() in ["summary", "sum"]:
                summary = researcher.get_conversation_summary()
                if summary:
                    print(f"\nConversation Summary:")
                    print(f"  Session: {summary['conversation_id'][:8]}...")
                    print(f"  Turns: {summary['total_turns']}")
                    print(
                        f"  Pokemon discussed: {', '.join(summary['mentioned_pokemon'])}"
                    )
                    print(f"  Themes: {', '.join(summary['themes'])}")
                else:
                    print("No active conversation.")
                continue
            elif query.lower() in ["clear", "cls"]:
                researcher.clear_current_session()
                print("Cleared current session.")
                continue

            if not query:
                print("Please enter a research question!")
                continue

            result = await researcher.process_query_with_spec_confirmation(query)

            if result.get("status") == "cancelled":
                print(f"\n{result['message']}")
            elif "error" in result:
                print(f"\nERROR: {result['error']}")
            else:
                print(f"\n" + "=" * 80)
                print("DEEP RESEARCH COMPLETE")
                print("=" * 80)
                print(result["final_report"])
                print("=" * 80)
                print(f"Status: {result['status']}")
                print(
                    f"Research Type: {result['specification']['query_type'].replace('_', ' ').title()}"
                )

        except KeyboardInterrupt:
            print("\nThank you for using Pokemon Deep Research Agent!")
            break
        except Exception as e:
            print(f"\nSYSTEM ERROR: {e}")


if __name__ == "__main__":
    asyncio.run(main())
