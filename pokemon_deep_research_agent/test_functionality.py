#!/usr/bin/env python3
"""
Comprehensive functionality test for Pokemon Deep Research Agent
Tests specification confirmation and interaction modes
"""

import asyncio
import os
import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.tools.llm_client import LLMClient
from src.tools.pokeapi_client import PokéAPIClient
from src.core.specification_manager import SpecificationManager
from src.core.query_disambiguator import QueryDisambiguator
from src.core.conversation_memory import ConversationMemoryManager
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class FunctionalityTester:
    """Comprehensive functionality tester for Pokemon Deep Research Agent."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.setup_clients()
    
    def setup_clients(self):
        """Setup all required clients and managers."""
        self.llm_client = LLMClient(
            api_key=self.api_key,
            model="gpt-4.1-mini",
            max_tokens=2000,
            temperature=0.3
        )
        
        self.pokeapi_client = PokéAPIClient(
            base_url="https://pokeapi.co/api/v2/",
            cache_dir="data/cache",
            rate_limit_delay=0.1,
            max_concurrent_requests=10
        )
        
        self.memory_manager = ConversationMemoryManager()
        self.spec_manager = SpecificationManager(self.llm_client)
        self.disambiguator = QueryDisambiguator(self.spec_manager, self.llm_client)
    
    async def test_basic_functionality(self):
        """Test basic system functionality."""
        print("=" * 60)
        print("TESTING BASIC FUNCTIONALITY")
        print("=" * 60)
        
        # Test 1: LLM Client
        print("\n[TEST 1] LLM Client Connection")
        try:
            response = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": "Say 'Hello, Pokemon!'"}],
                temperature=0.1
            )
            content = response['choices'][0]['message']['content']
            print(f"   SUCCESS: {content}")
        except Exception as e:
            print(f"   FAILED: {e}")
            return False
        
        # Test 2: PokéAPI Client
        print("\n[TEST 2] PokéAPI Client Connection")
        try:
            pokemon_data = await self.pokeapi_client.get_pokemon("pikachu")
            if pokemon_data:
                print(f"   SUCCESS: Retrieved {pokemon_data.name} data")
                print(f"   Types: {[t['type']['name'] for t in pokemon_data.types]}")
            else:
                print("   FAILED: No data returned")
                return False
        except Exception as e:
            print(f"   FAILED: {e}")
            return False
        
        # Test 3: Memory Manager
        print("\n[TEST 3] Conversation Memory Manager")
        try:
            conv_id = self.memory_manager.start_new_conversation()
            self.memory_manager.add_turn(
                conversation_id=conv_id,
                user_input="Test query",
                agent_response="Test response",
                pokemon_mentioned=["pikachu"],
                query_type="test"
            )
            history = self.memory_manager.get_conversation_history(conv_id)
            print(f"   SUCCESS: Created conversation with {len(history)} turns")
            self.memory_manager.clear_conversation(conv_id)
        except Exception as e:
            print(f"   FAILED: {e}")
            return False
        
        print("\n[RESULT] Basic functionality tests PASSED")
        return True
    
    async def test_specification_workflow(self):
        """Test specification confirmation workflow."""
        print("\n" + "=" * 60)
        print("TESTING SPECIFICATION WORKFLOW")
        print("=" * 60)
        
        test_queries = [
            "Tell me about Garchomp's competitive viability",
            "Build me a team around Pikachu",
            "Compare Charizard and Blastoise"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n[TEST {i}] Query: \"{query}\"")
            
            try:
                # Step 1: Generate specification
                start_time = time.time()
                spec = await self.spec_manager.analyze_query_and_generate_spec(query)
                gen_time = time.time() - start_time
                
                print(f"   Query Classification: {spec.query_type.value} ({gen_time:.2f}s)")
                print(f"   Extracted Fields: {spec.fields}")
                
                # Step 2: Check missing fields
                missing_required = self.spec_manager.get_missing_required_fields(spec)
                optional_fields = self.spec_manager.get_optional_fields_to_confirm(spec)
                
                print(f"   Missing Required: {[f.name for f in missing_required]}")
                print(f"   Optional Fields: {[f.name for f in optional_fields]}")
                
                # Step 3: Apply defaults
                self.spec_manager.apply_defaults_to_spec(spec)
                print(f"   After Defaults: {spec.fields}")
                
                print(f"   SUCCESS: Specification workflow completed")
                
            except Exception as e:
                print(f"   FAILED: {e}")
                return False
        
        print("\n[RESULT] Specification workflow tests PASSED")
        return True
    
    async def test_context_awareness(self):
        """Test context-aware functionality."""
        print("\n" + "=" * 60)
        print("TESTING CONTEXT AWARENESS")
        print("=" * 60)
        
        # Create conversation context
        conversation_context = {
            "mentioned_pokemon": ["garchomp", "dragonite"],
            "query_types": ["competitive"],
            "themes": ["competitive", "stats"]
        }
        
        follow_up_queries = [
            "How does it compare to Salamence?",
            "What about movesets?",
            "Can they work together on a team?"
        ]
        
        for i, query in enumerate(follow_up_queries, 1):
            print(f"\n[TEST {i}] Follow-up: \"{query}\"")
            print(f"   Context: {conversation_context}")
            
            try:
                # Generate spec with context
                spec = await self.spec_manager.analyze_query_and_generate_spec(
                    query, conversation_context
                )
                
                print(f"   Classification: {spec.query_type.value}")
                print(f"   Initial Fields: {spec.fields}")
                
                # Apply auto-completion
                spec = await self.disambiguator.auto_complete_from_context(
                    spec, conversation_context
                )
                
                print(f"   After Auto-completion: {spec.fields}")
                print(f"   SUCCESS: Context awareness working")
                
            except Exception as e:
                print(f"   FAILED: {e}")
                return False
        
        print("\n[RESULT] Context awareness tests PASSED")
        return True
    
    async def test_research_execution(self):
        """Test actual research execution with timeout protection."""
        print("\n" + "=" * 60)
        print("TESTING RESEARCH EXECUTION")
        print("=" * 60)
        
        # Simple research test
        print("\n[TEST 1] Simple Pokemon Analysis")
        try:
            # Create a simple specification
            from src.core.specification_manager import TaskSpecification, QueryType
            
            spec = TaskSpecification(
                query_type=QueryType.COMPETITIVE_ANALYSIS,
                original_query="Tell me about Pikachu's competitive viability",
                fields={
                    "target_pokemon": "pikachu",
                    "format": "OU",
                    "analysis_depth": "basic",
                    "include_counters": False
                },
                confirmed=True
            )
            
            # Get Pokemon data
            start_time = time.time()
            pokemon_data = await self.pokeapi_client.get_pokemon("pikachu")
            data_time = time.time() - start_time
            
            if pokemon_data:
                print(f"   Pokemon Data Retrieved: {pokemon_data.name} ({data_time:.2f}s)")
                print(f"   Types: {[t['type']['name'] for t in pokemon_data.types]}")
                print(f"   Base Stats: {[stat['base_stat'] for stat in pokemon_data.stats]}")
            else:
                print("   FAILED: No Pokemon data")
                return False
            
            # Generate analysis (with timeout)
            analysis_prompt = f"""
            Provide a brief competitive analysis of {pokemon_data.name.title()} for OU format.
            
            Pokemon: {pokemon_data.name.title()}
            Types: {', '.join([t['type']['name'] for t in pokemon_data.types])}
            
            Provide a 2-3 sentence analysis focusing on viability and role.
            """
            
            start_time = time.time()
            response = await asyncio.wait_for(
                self.llm_client.chat_completion(
                    messages=[{"role": "user", "content": analysis_prompt}],
                    temperature=0.3
                ),
                timeout=30.0  # 30 second timeout
            )
            analysis_time = time.time() - start_time
            
            analysis = response['choices'][0]['message']['content']
            print(f"   Analysis Generated ({analysis_time:.2f}s):")
            print(f"   {analysis[:200]}...")
            
            print(f"   SUCCESS: Research execution completed")
            
        except asyncio.TimeoutError:
            print("   FAILED: Research execution timed out")
            return False
        except Exception as e:
            print(f"   FAILED: {e}")
            return False
        
        print("\n[RESULT] Research execution tests PASSED")
        return True
    
    async def test_interaction_mode_simulation(self):
        """Test interaction mode with simulated input."""
        print("\n" + "=" * 60)
        print("TESTING INTERACTION MODE SIMULATION")
        print("=" * 60)
        
        # Simulate a complete interaction workflow
        print("\n[TEST 1] Complete Interaction Simulation")
        
        try:
            # Start conversation
            conv_id = self.memory_manager.start_new_conversation()
            print(f"   Started conversation: {conv_id[:8]}...")
            
            # Query 1: Initial query
            query1 = "Tell me about Garchomp's competitive viability"
            print(f"   Query 1: {query1}")
            
            # Generate and confirm specification
            spec1 = await self.spec_manager.analyze_query_and_generate_spec(query1)
            self.spec_manager.apply_defaults_to_spec(spec1)
            spec1.confirmed = True
            
            print(f"   Spec 1: {spec1.query_type.value} - {spec1.fields}")
            
            # Simulate research result
            result1 = "Garchomp is a top-tier Dragon/Ground type Pokemon in OU format..."
            
            # Add to memory
            self.memory_manager.add_turn(
                conversation_id=conv_id,
                user_input=query1,
                agent_response=result1,
                pokemon_mentioned=["garchomp"],
                query_type=spec1.query_type.value
            )
            
            # Query 2: Follow-up query
            query2 = "How does it compare to Dragonite?"
            print(f"   Query 2: {query2}")
            
            # Get context and generate spec
            context = self.memory_manager.get_persistent_context(conv_id)
            spec2 = await self.spec_manager.analyze_query_and_generate_spec(query2, context)
            spec2 = await self.disambiguator.auto_complete_from_context(spec2, context)
            self.spec_manager.apply_defaults_to_spec(spec2)
            spec2.confirmed = True
            
            print(f"   Spec 2: {spec2.query_type.value} - {spec2.fields}")
            print(f"   Context Used: {context}")
            
            # Simulate research result
            result2 = "Comparing Garchomp to Dragonite, both are powerful Dragon types..."
            
            # Add to memory
            self.memory_manager.add_turn(
                conversation_id=conv_id,
                user_input=query2,
                agent_response=result2,
                pokemon_mentioned=["dragonite"],
                query_type=spec2.query_type.value
            )
            
            # Check conversation summary
            summary = self.memory_manager.get_conversation_summary(conv_id)
            print(f"   Conversation Summary:")
            print(f"     Total Turns: {summary['total_turns']}")
            print(f"     Pokemon Discussed: {summary['mentioned_pokemon']}")
            print(f"     Query Types: {summary['query_types']}")
            
            # Cleanup
            self.memory_manager.clear_conversation(conv_id)
            
            print(f"   SUCCESS: Interaction mode simulation completed")
            
        except Exception as e:
            print(f"   FAILED: {e}")
            return False
        
        print("\n[RESULT] Interaction mode simulation PASSED")
        return True
    
    async def run_all_tests(self):
        """Run all functionality tests."""
        print("POKEMON DEEP RESEARCH AGENT - FUNCTIONALITY TESTING")
        print("=" * 60)
        
        tests = [
            ("Basic Functionality", self.test_basic_functionality),
            ("Specification Workflow", self.test_specification_workflow),
            ("Context Awareness", self.test_context_awareness),
            ("Research Execution", self.test_research_execution),
            ("Interaction Mode Simulation", self.test_interaction_mode_simulation)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\nRunning {test_name}...")
            try:
                if await test_func():
                    passed += 1
                    print(f"✓ {test_name} PASSED")
                else:
                    print(f"✗ {test_name} FAILED")
            except Exception as e:
                print(f"✗ {test_name} FAILED with exception: {e}")
        
        print("\n" + "=" * 60)
        print(f"TESTING COMPLETE: {passed}/{total} tests passed")
        print("=" * 60)
        
        if passed == total:
            print("ALL TESTS PASSED - System is working correctly!")
            return True
        else:
            print("SOME TESTS FAILED - System needs fixes")
            return False

async def main():
    """Main test function."""
    # Load environment
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found")
        return
    
    tester = FunctionalityTester(api_key)
    success = await tester.run_all_tests()
    
    if success:
        print("\nSYSTEM READY FOR PRODUCTION!")
    else:
        print("\nSYSTEM NEEDS FIXES BEFORE PRODUCTION")

if __name__ == "__main__":
    asyncio.run(main())

