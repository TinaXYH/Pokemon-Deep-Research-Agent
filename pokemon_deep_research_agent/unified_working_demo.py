#!/usr/bin/env python3
"""
Unified Pokemon Deep Research Agent - Complete Working Demo
Combines conversation memory, specification confirmation, and real Pokemon research
"""

import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import our research components
from src.tools.llm_client import LLMClient
from src.tools.pokeapi_client import PokéAPIClient

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ConversationMemory:
    """Simple conversation memory for this demo"""

    def __init__(self):
        self.conversations = {}
        self.current_conversation_id = None

    def start_new_conversation(self):
        """Start a new conversation"""
        conv_id = str(uuid.uuid4())[:8]
        self.current_conversation_id = conv_id
        self.conversations[conv_id] = {
            "turns": [],
            "pokemon_discussed": set(),
            "research_context": {},
            "started_at": datetime.now().isoformat(),
        }
        return conv_id

    def add_turn(
        self, user_query: str, agent_response: str, pokemon_names: List[str] = None
    ):
        """Add a conversation turn"""
        if not self.current_conversation_id:
            self.start_new_conversation()

        conv = self.conversations[self.current_conversation_id]
        conv["turns"].append(
            {
                "user": user_query,
                "agent": agent_response,
                "timestamp": datetime.now().isoformat(),
            }
        )

        if pokemon_names:
            conv["pokemon_discussed"].update(pokemon_names)

    def get_context(self) -> Dict[str, Any]:
        """Get current conversation context"""
        if (
            not self.current_conversation_id
            or self.current_conversation_id not in self.conversations
        ):
            return {}

        conv = self.conversations[self.current_conversation_id]
        return {
            "previous_turns": conv["turns"][-3:],  # Last 3 turns
            "pokemon_discussed": list(conv["pokemon_discussed"]),
            "turn_count": len(conv["turns"]),
        }


class SpecificationManager:
    """Handles query classification and specification generation"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def classify_and_generate_spec(
        self, query: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Classify query and generate specification"""

        # Build context-aware prompt
        context_info = ""
        if context.get("pokemon_discussed"):
            context_info = f"Previously discussed Pokemon: {', '.join(context['pokemon_discussed'])}"

        if context.get("previous_turns"):
            recent_turns = context["previous_turns"][-2:]  # Last 2 turns
            context_info += "\nRecent conversation:\n"
            for turn in recent_turns:
                context_info += (
                    f"User: {turn['user']}\nAgent: {turn['agent'][:100]}...\n"
                )

        classification_prompt = f"""
        Analyze this Pokemon query and classify it. Consider the conversation context.
        
        Query: "{query}"
        
        Context: {context_info}
        
        Classify the query type and extract key information:
        1. Research Type: competitive_analysis, team_building, pokemon_comparison, moveset_analysis, meta_analysis, or general_research
        2. Pokemon Names: Extract any Pokemon names mentioned (or referenced by "it", "them", etc.)
        3. Format: OU, UU, RU, etc. (default: OU)
        4. Specific Focus: What specific aspect to focus on
        
        If this is a follow-up question using pronouns like "it", "them", "this Pokemon", 
        resolve them using the context of previously discussed Pokemon.
        
        Return a JSON object with: research_type, pokemon_names, format, focus, is_followup
        """

        try:
            response = await self.llm_client.chat_completion(
                [{"role": "user", "content": classification_prompt}]
            )
            response_text = response["choices"][0]["message"]["content"]

            # Try to extract JSON from response
            import re

            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                spec = json.loads(json_match.group())
            else:
                # Fallback specification
                spec = {
                    "research_type": "general_research",
                    "pokemon_names": [],
                    "format": "OU",
                    "focus": query,
                    "is_followup": len(context.get("previous_turns", [])) > 0,
                }
        except Exception as e:
            logger.warning(f"Failed to parse specification: {e}")
            spec = {
                "research_type": "general_research",
                "pokemon_names": [],
                "format": "OU",
                "focus": query,
                "is_followup": len(context.get("previous_turns", [])) > 0,
            }

        return spec


class UnifiedResearchAgent:
    """Unified research agent that combines all functionality"""

    def __init__(self):
        # Load environment variables
        from dotenv import load_dotenv

        load_dotenv()

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.llm_client = LLMClient(api_key)
        self.pokeapi_client = PokéAPIClient()
        self.memory = ConversationMemory()
        self.spec_manager = SpecificationManager(self.llm_client)

    async def process_query(self, query: str) -> str:
        """Process a query with full workflow"""

        print(f"\n{'='*80}")
        print(f"📋 User Query: {query}")

        # Get conversation context
        context = self.memory.get_context()

        if context.get("pokemon_discussed"):
            print(
                f"🧠 Conversation Context: Previously discussed {', '.join(list(context['pokemon_discussed'])[:3])}"
            )

        # Phase 1: Query Analysis & Specification
        print(f"\n🔍 PHASE 1: Query Analysis & Specification Generation")
        print(f"   Analyzing query intent and generating research specification...")

        spec = await self.spec_manager.classify_and_generate_spec(query, context)

        print(f"   ✅ Research Type: {spec['research_type']}")
        print(f"   ✅ Pokemon: {spec.get('pokemon_names', 'None specified')}")
        print(f"   ✅ Format: {spec.get('format', 'OU')}")

        # Phase 2: Specification Confirmation (simplified for demo)
        print(f"\n⚙️  PHASE 2: Specification Confirmation")
        print(f"   Research specification confirmed automatically")
        print(f"   Focus: {spec.get('focus', query)}")

        # Phase 3: Data Collection
        print(f"\n📊 PHASE 3: Data Collection")
        print(f"   Collecting Pokemon data and competitive information...")

        pokemon_data = {}
        if spec.get("pokemon_names"):
            for pokemon_name in spec["pokemon_names"]:
                try:
                    print(f"   🔍 Fetching data for {pokemon_name}...")
                    data = await self.pokeapi_client.get_pokemon(pokemon_name.lower())
                    pokemon_data[pokemon_name] = (
                        data.model_dump() if hasattr(data, "model_dump") else data
                    )
                    print(
                        f"   ✅ Retrieved {pokemon_name} data (ID: {getattr(data, 'id', 'N/A')})"
                    )
                except Exception as e:
                    print(f"   ⚠️  Could not fetch {pokemon_name}: {e}")

        # Phase 4: Deep Analysis
        print(f"\n🧠 PHASE 4: Deep Analysis")
        print(f"   Performing competitive analysis and strategic evaluation...")

        analysis_prompt = self._build_analysis_prompt(
            spec, pokemon_data, context, query
        )
        analysis_response = await self.llm_client.chat_completion(
            [{"role": "user", "content": analysis_prompt}]
        )
        analysis = analysis_response["choices"][0]["message"]["content"]

        print(f"   ✅ Analysis completed ({len(analysis)} characters)")

        # Phase 5: Report Generation
        print(f"\n📝 PHASE 5: Report Generation")
        print(f"   Generating comprehensive research report...")

        report_prompt = self._build_report_prompt(
            spec, pokemon_data, analysis, context, query
        )
        report_response = await self.llm_client.chat_completion(
            [{"role": "user", "content": report_prompt}]
        )
        final_report = report_response["choices"][0]["message"]["content"]

        # Update conversation memory
        pokemon_names = spec.get("pokemon_names", [])
        self.memory.add_turn(query, final_report, pokemon_names)

        print(f"   ✅ Report generated and saved to conversation memory")
        print(f"\n{'='*80}")

        return final_report

    def _build_analysis_prompt(
        self, spec: Dict, pokemon_data: Dict, context: Dict, original_query: str
    ) -> str:
        """Build analysis prompt"""

        context_str = ""
        if context.get("previous_turns"):
            context_str = "Previous conversation context:\n"
            for turn in context["previous_turns"][-2:]:
                context_str += (
                    f"User: {turn['user']}\nAgent: {turn['agent'][:200]}...\n\n"
                )

        pokemon_info = ""
        if pokemon_data:
            pokemon_info = "Pokemon data available:\n"
            for name, data in pokemon_data.items():
                pokemon_info += f"- {name}: Type {data.get('types', [])}, Stats: {data.get('stats', {})}\n"

        prompt = f"""
        You are a Pokemon competitive analysis expert. Analyze the following research request:
        
        Original Query: "{original_query}"
        Research Type: {spec.get('research_type', 'general')}
        Pokemon Focus: {spec.get('pokemon_names', [])}
        Format: {spec.get('format', 'OU')}
        
        {context_str}
        
        {pokemon_info}
        
        Provide a detailed competitive analysis focusing on:
        - Statistical analysis and competitive viability
        - Role analysis and team positioning  
        - Strengths and weaknesses
        - Counter strategies and matchups
        - Team synergies and recommendations
        
        If this is a follow-up question, build upon the previous discussion.
        Keep the analysis professional, detailed, and actionable.
        """

        return prompt

    def _build_report_prompt(
        self,
        spec: Dict,
        pokemon_data: Dict,
        analysis: str,
        context: Dict,
        original_query: str,
    ) -> str:
        """Build final report prompt"""

        context_str = ""
        if context.get("previous_turns"):
            context_str = "This is a follow-up to previous discussion. Build upon previous context naturally."

        prompt = f"""
        Generate a comprehensive Pokemon research report based on the analysis.
        
        Original Query: "{original_query}"
        Research Type: {spec.get('research_type', 'general')}
        
        {context_str}
        
        Analysis Results: {analysis}
        
        Format the report as a professional research document with:
        1. Executive Summary
        2. Detailed Analysis
        3. Strategic Recommendations
        4. Practical Applications
        5. Conclusion
        
        Use clear markdown formatting for readability.
        Make the response comprehensive but focused on the user's specific question.
        If this is a follow-up, reference previous discussion naturally.
        """

        return prompt

    def start_new_conversation(self):
        """Start a new conversation"""
        conv_id = self.memory.start_new_conversation()
        print(f"🆕 Started new conversation session: {conv_id}")
        return conv_id

    def get_conversation_summary(self):
        """Get conversation summary"""
        context = self.memory.get_context()
        if not context.get("previous_turns"):
            return "No conversation history yet."

        summary = f"Conversation Summary:\n"
        summary += f"- Total turns: {context.get('turn_count', 0)}\n"
        summary += (
            f"- Pokemon discussed: {', '.join(context.get('pokemon_discussed', []))}\n"
        )
        summary += f"- Recent topics: {[turn['user'] for turn in context['previous_turns'][-3:]]}\n"

        return summary


async def main():
    """Main interactive loop"""

    print("🧠 POKEMON DEEP RESEARCH AGENT - UNIFIED WORKING DEMO")
    print("=" * 80)
    print("🎯 Features:")
    print("• Real conversation memory and context")
    print("• Specification confirmation workflow")
    print("• Live PokéAPI data integration")
    print("• Context-aware follow-up questions")
    print("• Professional research reports")
    print()
    print("Commands:")
    print("• 'new session' - Start fresh conversation")
    print("• 'summary' - Show conversation summary")
    print("• 'quit' - Exit")
    print("=" * 80)

    try:
        agent = UnifiedResearchAgent()
        agent.start_new_conversation()

        while True:
            try:
                query = input("\n🔍 Your Pokemon question: ").strip()

                if not query:
                    continue

                if query.lower() == "quit":
                    print("👋 Goodbye!")
                    break

                if query.lower() == "new session":
                    agent.start_new_conversation()
                    continue

                if query.lower() == "summary":
                    print(agent.get_conversation_summary())
                    continue

                # Process the query
                response = await agent.process_query(query)

                print("\n📋 RESEARCH REPORT:")
                print("-" * 80)
                print(response)
                print("-" * 80)
                print("✅ Research completed! Ask a follow-up question or new topic.")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error processing query: {e}")
                logger.error(f"Query processing error: {e}")

    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        logger.error(f"Initialization error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
