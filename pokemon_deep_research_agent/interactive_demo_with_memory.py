#!/usr/bin/env python3
"""
Interactive Pokemon Deep Research Agent Demo with Conversation Memory
Supports follow-up questions and maintains conversation history within sessions
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
from src.tools.llm_client import LLMClient
from src.tools.pokeapi_client import PokéAPIClient

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class MemoryEnabledPokemonResearcher:
    """Pokemon researcher with conversation memory and context awareness."""

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

        # Initialize conversation memory
        self.memory_manager = ConversationMemoryManager()
        self.current_conversation_id = None

    def start_new_session(self) -> str:
        """Start a new conversation session."""
        self.current_conversation_id = self.memory_manager.start_new_conversation()
        return self.current_conversation_id

    async def process_pokemon_query_with_memory(self, query: str) -> Dict[str, Any]:
        """Process Pokemon query with conversation memory and context."""

        if not self.current_conversation_id:
            self.start_new_session()

        print(f"\n🚀 **POKEMON DEEP RESEARCH AGENT - MEMORY-ENABLED MODE**")
        print("=" * 75)
        print(f"📋 **User Query:** {query}")

        # Get conversation context
        context_prompt = self.memory_manager.format_context_for_llm(
            self.current_conversation_id, max_turns=3
        )

        if context_prompt:
            print(
                f"🧠 **Using Conversation Context:** Previous discussion about {', '.join(self.memory_manager.get_persistent_context(self.current_conversation_id).get('mentioned_pokemon', []))}"
            )

        print()

        # Step 1: Context-Aware Query Analysis
        print(f"🧠 **STEP 1: Context-Aware Query Analysis Agent**")
        print(f"   Analyzing query with conversation context...")

        analysis_prompt = f"""
        {context_prompt}
        
        Analyze this Pokemon query considering the conversation context:
        Query: "{query}"
        
        Determine:
        1. Is this a follow-up question or new topic?
        2. What Pokemon are mentioned or referenced?
        3. What type of analysis is needed?
        4. How does this relate to previous discussion?
        5. Research strategy considering context
        
        Provide a clear analysis plan.
        """

        try:
            analysis_response = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": analysis_prompt}], temperature=0.3
            )
            query_analysis = analysis_response["choices"][0]["message"]["content"]

            print(f"   ✅ Context-aware analysis completed")
            print(f"   📊 Analysis: {query_analysis[:200]}...")

            # Extract query type and Pokemon
            query_type = self._extract_query_type(query, query_analysis)
            pokemon_names = await self._extract_pokemon_names_with_context(
                query, query_analysis
            )

            # Step 2: Smart Data Collection
            print(f"\n🔍 **STEP 2: Context-Aware Data Collection Agent**")

            # Check if we need new data or can use context
            persistent_context = self.memory_manager.get_persistent_context(
                self.current_conversation_id
            )
            previously_discussed = persistent_context.get("mentioned_pokemon", [])

            if not pokemon_names and previously_discussed:
                print(
                    f"   🔄 No new Pokemon mentioned, using context from: {', '.join(previously_discussed)}"
                )
                pokemon_names = previously_discussed[:3]  # Use up to 3 most recent

            if not pokemon_names:
                print(
                    f"   ⚠️  No specific Pokemon mentioned, using general research approach"
                )
                return await self._handle_general_query_with_memory(
                    query, query_analysis
                )

            print(
                f"   🎯 Target Pokemon: {', '.join([p.title() for p in pokemon_names])}"
            )

            # Collect data (with caching awareness)
            pokemon_data = {}
            for pokemon_name in pokemon_names:
                print(f"   📡 Retrieving {pokemon_name} data from PokéAPI...")
                data = await self.pokeapi_client.get_pokemon(pokemon_name.lower())
                if data:
                    pokemon_data[pokemon_name] = data
                    print(f"   ✅ Retrieved data for {data.name.title()}")
                    print(
                        f"      📊 Base stats: {[stat['base_stat'] for stat in data.stats]}"
                    )
                    print(f"      🏷️ Types: {[t['type']['name'] for t in data.types]}")
                else:
                    print(f"   ❌ Could not find data for {pokemon_name}")

            if not pokemon_data:
                return {"error": "No valid Pokemon data found"}

            # Step 3: Context-Aware Analysis
            print(f"\n🧠 **STEP 3: Context-Aware Pokemon Analysis Agent**")
            print(f"   Performing analysis with conversation memory...")

            analysis = await self._analyze_pokemon_data_with_context(
                pokemon_data, query, query_analysis, context_prompt
            )

            print(f"   ✅ Context-aware analysis completed")

            # Step 4: Memory-Enhanced Report Generation
            print(f"\n📝 **STEP 4: Memory-Enhanced Report Generation Agent**")
            print(f"   Synthesizing report with conversation continuity...")

            final_report = await self._generate_contextual_report(
                pokemon_data, analysis, query, context_prompt
            )

            print(f"   ✅ Memory-enhanced report generated")

            # Save to conversation memory
            self.memory_manager.add_turn(
                conversation_id=self.current_conversation_id,
                user_input=query,
                agent_response=final_report,
                pokemon_mentioned=list(pokemon_data.keys()),
                query_type=query_type,
                context_used={"previous_pokemon": previously_discussed},
            )

            return {
                "query": query,
                "conversation_id": self.current_conversation_id,
                "pokemon_analyzed": list(pokemon_data.keys()),
                "query_analysis": query_analysis,
                "detailed_analysis": analysis,
                "final_report": final_report,
                "context_used": bool(context_prompt),
                "status": "completed",
            }

        except Exception as e:
            print(f"   ❌ Error during processing: {e}")
            return {"error": str(e)}

    def _extract_query_type(self, query: str, analysis: str) -> str:
        """Extract query type from query and analysis."""
        query_lower = query.lower()
        analysis_lower = analysis.lower()

        if any(
            word in query_lower for word in ["compare", "vs", "versus", "difference"]
        ):
            return "comparison"
        elif any(word in query_lower for word in ["team", "build", "synergy"]):
            return "team_building"
        elif any(word in query_lower for word in ["competitive", "tier", "meta"]):
            return "competitive"
        elif any(word in query_lower for word in ["moveset", "moves", "attacks"]):
            return "movesets"
        elif any(word in query_lower for word in ["stats", "base stats"]):
            return "stats"
        else:
            return "general"

    async def _extract_pokemon_names_with_context(
        self, query: str, analysis: str
    ) -> List[str]:
        """Extract Pokemon names considering context and analysis."""

        # Enhanced Pokemon name extraction using LLM
        extraction_prompt = f"""
        Extract Pokemon names from this query and analysis:
        
        Query: "{query}"
        Analysis: "{analysis}"
        
        List only the specific Pokemon names mentioned. If pronouns like "it", "them", "this Pokemon" are used, 
        indicate that context is needed.
        
        Format: Just list the Pokemon names, one per line. If context needed, write "CONTEXT_NEEDED".
        """

        try:
            response = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": extraction_prompt}],
                temperature=0.1,
            )

            result = response["choices"][0]["message"]["content"].strip()

            if "CONTEXT_NEEDED" in result:
                return []  # Will use context from memory

            # Parse Pokemon names
            lines = [
                line.strip().lower() for line in result.split("\n") if line.strip()
            ]
            return [name for name in lines if name and not name.startswith("#")]

        except Exception:
            # Fallback to simple extraction
            return await self._simple_pokemon_extraction(query)

    async def _simple_pokemon_extraction(self, query: str) -> List[str]:
        """Simple Pokemon name extraction fallback."""
        common_pokemon = [
            "pikachu",
            "charizard",
            "blastoise",
            "venusaur",
            "mewtwo",
            "mew",
            "garchomp",
            "dragonite",
            "tyranitar",
            "metagross",
            "salamence",
            "lucario",
            "gengar",
            "alakazam",
            "machamp",
            "golem",
            "lapras",
            "snorlax",
            "gyarados",
            "vaporeon",
            "jolteon",
            "flareon",
            "espeon",
            "umbreon",
            "leafeon",
            "glaceon",
            "sylveon",
            "eevee",
            "raichu",
        ]

        query_lower = query.lower()
        return [pokemon for pokemon in common_pokemon if pokemon in query_lower]

    async def _handle_general_query_with_memory(
        self, query: str, analysis: str
    ) -> Dict[str, Any]:
        """Handle general queries with conversation memory."""

        print(f"   🔍 Processing general query with conversation context...")

        context_prompt = self.memory_manager.format_context_for_llm(
            self.current_conversation_id, max_turns=3
        )

        general_prompt = f"""
        {context_prompt}
        
        Answer this Pokemon query considering the conversation context:
        Query: "{query}"
        
        Analysis context: {analysis}
        
        Provide a detailed, contextual response that:
        1. References previous discussion if relevant
        2. Addresses all aspects of the current query
        3. Includes specific examples and data
        4. Maintains conversation continuity
        """

        response = await self.llm_client.chat_completion(
            messages=[{"role": "user", "content": general_prompt}], temperature=0.4
        )

        result = response["choices"][0]["message"]["content"]

        # Save to memory
        self.memory_manager.add_turn(
            conversation_id=self.current_conversation_id,
            user_input=query,
            agent_response=result,
            pokemon_mentioned=[],
            query_type="general",
            context_used={"has_context": bool(context_prompt)},
        )

        return {
            "query": query,
            "conversation_id": self.current_conversation_id,
            "query_type": "general",
            "analysis": analysis,
            "result": result,
            "context_used": bool(context_prompt),
            "status": "completed",
        }

    async def _analyze_pokemon_data_with_context(
        self,
        pokemon_data: Dict[str, Any],
        query: str,
        query_analysis: str,
        context_prompt: str,
    ) -> str:
        """Analyze Pokemon data with conversation context."""

        analysis_prompt = f"""
        {context_prompt}
        
        Perform contextual analysis for: "{query}"
        
        Query Analysis: {query_analysis}
        
        Pokemon Data Available:
        """

        for name, data in pokemon_data.items():
            analysis_prompt += f"""
        
        {data.name.title()}:
        - Base Stats: {self._format_stats(data.stats)}
        - Types: {', '.join([t['type']['name'] for t in data.types])}
        - Abilities: {', '.join([a['ability']['name'] for a in data.abilities])}
        """

        analysis_prompt += """
        
        Provide comprehensive analysis that:
        1. Considers previous conversation context
        2. Addresses the specific query
        3. Builds on previous discussion if relevant
        4. Includes competitive analysis, strengths/weaknesses
        5. Provides specific insights and recommendations
        
        Be detailed and maintain conversation continuity.
        """

        response = await self.llm_client.chat_completion(
            messages=[{"role": "user", "content": analysis_prompt}], temperature=0.3
        )

        return response["choices"][0]["message"]["content"]

    async def _generate_contextual_report(
        self,
        pokemon_data: Dict[str, Any],
        analysis: str,
        query: str,
        context_prompt: str,
    ) -> str:
        """Generate report with conversation context."""

        pokemon_names = [data.name.title() for data in pokemon_data.values()]
        title = (
            f"{' & '.join(pokemon_names)} Analysis"
            if len(pokemon_names) > 1
            else f"{pokemon_names[0]} Analysis"
        )

        report_prompt = f"""
        {context_prompt}
        
        Create a comprehensive research report answering: "{query}"
        
        Title: {title}
        
        Based on this contextual analysis:
        {analysis}
        
        Format as a professional report that:
        1. References previous discussion if relevant
        2. Maintains conversation continuity
        3. Provides comprehensive analysis
        4. Includes actionable recommendations
        
        Structure:
        # {title}
        
        ## Executive Summary
        ## Detailed Analysis  
        ## Key Findings
        ## Recommendations
        ## Follow-up Suggestions
        
        Make it comprehensive and contextually aware.
        """

        response = await self.llm_client.chat_completion(
            messages=[{"role": "user", "content": report_prompt}], temperature=0.2
        )

        return response["choices"][0]["message"]["content"]

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
    """Main interactive function with memory."""

    # Load environment
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found")
        return

    researcher = MemoryEnabledPokemonResearcher(api_key)

    print("🧠 **POKEMON DEEP RESEARCH AGENT - MEMORY-ENABLED MODE**")
    print("=" * 70)
    print("🎯 **NEW FEATURE: Conversation Memory & Follow-up Questions!**")
    print()
    print("Features:")
    print("• 🧠 Remembers previous questions and context")
    print("• 🔄 Supports follow-up questions")
    print("• 📚 Maintains conversation history")
    print("• 🎯 Context-aware responses")
    print()
    print("Try these examples:")
    print("1. 'Tell me about Garchomp's competitive viability'")
    print("2. 'How does it compare to Dragonite?' (follow-up)")
    print("3. 'What about team synergies?' (another follow-up)")
    print()
    print("Commands:")
    print("• 'new session' - Start fresh conversation")
    print("• 'summary' - Show conversation summary")
    print("• 'clear' - Clear current session")
    print("• 'quit' - Exit")
    print("=" * 70)

    # Start initial session
    session_id = researcher.start_new_session()
    print(f"🆕 Started new conversation session: {session_id[:8]}...")

    while True:
        try:
            query = input("\n🔍 Your Pokemon question: ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                print("👋 Thanks for using Pokemon Deep Research Agent!")
                break
            elif query.lower() in ["new session", "new", "reset"]:
                session_id = researcher.start_new_session()
                print(f"🆕 Started new conversation session: {session_id[:8]}...")
                continue
            elif query.lower() in ["summary", "sum"]:
                summary = researcher.get_conversation_summary()
                if summary:
                    print(f"\n📊 **Conversation Summary:**")
                    print(f"   Session: {summary['conversation_id'][:8]}...")
                    print(f"   Turns: {summary['total_turns']}")
                    print(
                        f"   Pokemon discussed: {', '.join(summary['mentioned_pokemon'])}"
                    )
                    print(f"   Themes: {', '.join(summary['themes'])}")
                else:
                    print("No active conversation.")
                continue
            elif query.lower() in ["clear", "cls"]:
                researcher.clear_current_session()
                print("🗑️ Cleared current session.")
                continue

            if not query:
                print("Please enter a question!")
                continue

            result = await researcher.process_pokemon_query_with_memory(query)

            if "error" in result:
                print(f"\n❌ **Error:** {result['error']}")
            else:
                print(f"\n🎉 **RESEARCH COMPLETE!**")
                print("=" * 75)

                if result.get("query_type") == "general":
                    print(result["result"])
                else:
                    print(result["final_report"])

                print("=" * 75)
                print(f"✅ **Status:** {result['status']}")
                if result.get("context_used"):
                    print(f"🧠 **Used conversation context for better response**")

        except KeyboardInterrupt:
            print("\n👋 Thanks for using Pokemon Deep Research Agent!")
            break
        except Exception as e:
            print(f"\n❌ **System Error:** {e}")


if __name__ == "__main__":
    asyncio.run(main())
