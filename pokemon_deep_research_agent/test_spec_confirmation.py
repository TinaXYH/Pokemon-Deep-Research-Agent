#!/usr/bin/env python3
"""
Test script for specification confirmation workflow
Tests the complete query disambiguation and spec confirmation process
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import logging

from src.core.conversation_memory import ConversationMemoryManager
from src.core.query_disambiguator import QueryDisambiguator
from src.core.specification_manager import QueryType, SpecificationManager
from src.tools.llm_client import LLMClient

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


async def test_specification_workflow():
    """Test the complete specification confirmation workflow."""

    print("🔬 **TESTING SPECIFICATION CONFIRMATION WORKFLOW**")
    print("=" * 70)

    # Load environment
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found")
        return

    # Initialize components
    llm_client = LLMClient(
        api_key=api_key, model="gpt-4.1-mini", max_tokens=2000, temperature=0.3
    )

    spec_manager = SpecificationManager(llm_client)
    disambiguator = QueryDisambiguator(spec_manager, llm_client)
    memory_manager = ConversationMemoryManager()

    # Test cases
    test_queries = [
        {
            "query": "Tell me about Garchomp's competitive viability",
            "expected_type": QueryType.COMPETITIVE_ANALYSIS,
            "description": "Competitive analysis query",
        },
        {
            "query": "Build me a team around Pikachu",
            "expected_type": QueryType.TEAM_BUILDING,
            "description": "Team building query",
        },
        {
            "query": "Compare Charizard and Blastoise",
            "expected_type": QueryType.POKEMON_COMPARISON,
            "description": "Pokemon comparison query",
        },
        {
            "query": "What movesets does Mewtwo use?",
            "expected_type": QueryType.MOVESET_ANALYSIS,
            "description": "Moveset analysis query",
        },
        {
            "query": "What's the current OU meta like?",
            "expected_type": QueryType.META_ANALYSIS,
            "description": "Meta analysis query",
        },
    ]

    for i, test_case in enumerate(test_queries, 1):
        print(f"\n📝 **TEST {i}: {test_case['description']}**")
        print(f"Query: \"{test_case['query']}\"")

        try:
            # Step 1: Query analysis and spec generation
            print(f"   🔍 Analyzing query and generating specification...")

            spec = await spec_manager.analyze_query_and_generate_spec(
                test_case["query"]
            )

            print(f"   ✅ Query classified as: {spec.query_type.value}")
            print(f"   📊 Expected: {test_case['expected_type'].value}")

            if spec.query_type == test_case["expected_type"]:
                print(f"   ✅ Classification correct!")
            else:
                print(f"   ⚠️ Classification mismatch")

            # Step 2: Show extracted fields
            print(f"   📋 Extracted fields: {spec.fields}")

            # Step 3: Check missing required fields
            missing_required = spec_manager.get_missing_required_fields(spec)
            optional_fields = spec_manager.get_optional_fields_to_confirm(spec)

            print(
                f"   ❓ Missing required fields: {[f.name for f in missing_required]}"
            )
            print(
                f"   ⚙️ Optional fields to confirm: {[f.name for f in optional_fields]}"
            )

            # Step 4: Show specification summary
            print(f"   📄 Specification summary:")
            summary = spec_manager.format_specification_summary(spec)
            print("   " + summary.replace("\n", "\n   "))

            # Step 5: Apply defaults
            spec_manager.apply_defaults_to_spec(spec)
            print(f"   📌 Applied defaults: {spec.fields}")

            print(f"   ✅ Test {i} completed successfully")

        except Exception as e:
            print(f"   ❌ Test {i} failed: {e}")

    print(f"\n🎉 **SPECIFICATION WORKFLOW TESTS COMPLETED!**")


async def test_context_auto_completion():
    """Test auto-completion from conversation context."""

    print(f"\n🧠 **TESTING CONTEXT AUTO-COMPLETION**")
    print("=" * 70)

    # Load environment
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found")
        return

    # Initialize components
    llm_client = LLMClient(
        api_key=api_key, model="gpt-4.1-mini", max_tokens=2000, temperature=0.3
    )

    spec_manager = SpecificationManager(llm_client)
    disambiguator = QueryDisambiguator(spec_manager, llm_client)
    memory_manager = ConversationMemoryManager()

    # Create conversation context
    conversation_context = {
        "mentioned_pokemon": ["garchomp", "dragonite"],
        "query_types": ["competitive"],
        "themes": ["competitive", "stats"],
    }

    # Test follow-up queries
    follow_up_queries = [
        {
            "query": "How does it compare to Salamence?",
            "description": "Pronoun reference follow-up",
        },
        {"query": "What about movesets?", "description": "Implicit topic follow-up"},
        {
            "query": "Can they work together on a team?",
            "description": "Multiple Pokemon reference",
        },
    ]

    for i, test_case in enumerate(follow_up_queries, 1):
        print(f"\n📝 **CONTEXT TEST {i}: {test_case['description']}**")
        print(f"Query: \"{test_case['query']}\"")
        print(f"Context: {conversation_context}")

        try:
            # Generate specification with context
            spec = await spec_manager.analyze_query_and_generate_spec(
                test_case["query"], conversation_context
            )

            print(f"   🔍 Query classified as: {spec.query_type.value}")
            print(f"   📋 Initial fields: {spec.fields}")

            # Apply auto-completion from context
            spec = await disambiguator.auto_complete_from_context(
                spec, conversation_context
            )

            print(f"   🧠 After context auto-completion: {spec.fields}")
            print(f"   ✅ Context test {i} completed")

        except Exception as e:
            print(f"   ❌ Context test {i} failed: {e}")

    print(f"\n🎉 **CONTEXT AUTO-COMPLETION TESTS COMPLETED!**")


async def test_schema_validation():
    """Test specification schema validation."""

    print(f"\n📋 **TESTING SPECIFICATION SCHEMAS**")
    print("=" * 70)

    # Initialize spec manager
    spec_manager = SpecificationManager()

    # Test all query types have schemas
    for query_type in QueryType:
        print(f"\n📝 **Testing schema for: {query_type.value}**")

        schema = spec_manager.schemas.get(query_type)
        if schema:
            print(f"   ✅ Schema found with {len(schema)} fields")

            # Check required fields
            required_fields = [f for f in schema if f.required]
            optional_fields = [f for f in schema if not f.required]

            print(f"   📌 Required fields: {[f.name for f in required_fields]}")
            print(f"   ⚙️ Optional fields: {[f.name for f in optional_fields]}")

            # Validate field properties
            for field in schema:
                if field.field_type == "select" and not field.options:
                    print(f"   ⚠️ Select field {field.name} missing options")
                elif not field.question:
                    print(f"   ⚠️ Field {field.name} missing question")
                else:
                    print(f"   ✅ Field {field.name} properly configured")
        else:
            print(f"   ❌ No schema found for {query_type.value}")

    print(f"\n🎉 **SCHEMA VALIDATION COMPLETED!**")


async def test_field_extraction():
    """Test field extraction from queries."""

    print(f"\n🔍 **TESTING FIELD EXTRACTION**")
    print("=" * 70)

    # Load environment
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found")
        return

    # Initialize components
    llm_client = LLMClient(
        api_key=api_key, model="gpt-4.1-mini", max_tokens=1000, temperature=0.1
    )

    spec_manager = SpecificationManager(llm_client)

    # Test extraction cases
    extraction_tests = [
        {
            "query": "Tell me about Garchomp's competitive viability in OU",
            "expected_fields": {"target_pokemon": "garchomp", "format": "OU"},
            "query_type": QueryType.COMPETITIVE_ANALYSIS,
        },
        {
            "query": "Build a balanced team around Pikachu for VGC",
            "expected_fields": {
                "core_pokemon": "pikachu",
                "format": "VGC",
                "team_style": "balanced",
            },
            "query_type": QueryType.TEAM_BUILDING,
        },
        {
            "query": "Compare Charizard and Blastoise stats",
            "expected_fields": {
                "pokemon_list": ["charizard", "blastoise"],
                "comparison_aspects": "stats",
            },
            "query_type": QueryType.POKEMON_COMPARISON,
        },
    ]

    for i, test_case in enumerate(extraction_tests, 1):
        print(f"\n📝 **EXTRACTION TEST {i}**")
        print(f"Query: \"{test_case['query']}\"")
        print(f"Expected fields: {test_case['expected_fields']}")

        try:
            # Extract fields
            extracted_fields = await spec_manager._extract_fields_from_query(
                test_case["query"], test_case["query_type"]
            )

            print(f"   🔍 Extracted fields: {extracted_fields}")

            # Check if expected fields were extracted
            matches = 0
            for key, expected_value in test_case["expected_fields"].items():
                if key in extracted_fields:
                    extracted_value = extracted_fields[key]
                    if isinstance(expected_value, list):
                        if set(expected_value).issubset(set(extracted_value)):
                            matches += 1
                            print(f"   ✅ {key}: {extracted_value} (contains expected)")
                        else:
                            print(
                                f"   ⚠️ {key}: {extracted_value} (missing some expected)"
                            )
                    elif str(expected_value).lower() == str(extracted_value).lower():
                        matches += 1
                        print(f"   ✅ {key}: {extracted_value}")
                    else:
                        print(
                            f"   ⚠️ {key}: {extracted_value} (expected {expected_value})"
                        )
                else:
                    print(f"   ❌ {key}: not extracted")

            accuracy = matches / len(test_case["expected_fields"]) * 100
            print(f"   📊 Extraction accuracy: {accuracy:.1f}%")

        except Exception as e:
            print(f"   ❌ Extraction test {i} failed: {e}")

    print(f"\n🎉 **FIELD EXTRACTION TESTS COMPLETED!**")


if __name__ == "__main__":
    asyncio.run(test_specification_workflow())
    asyncio.run(test_context_auto_completion())
    asyncio.run(test_schema_validation())
    asyncio.run(test_field_extraction())
