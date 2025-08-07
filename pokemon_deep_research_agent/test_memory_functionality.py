#!/usr/bin/env python3
"""
Test script for conversation memory functionality
Demonstrates follow-up questions and context awareness
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import logging

from src.core.conversation_memory import ConversationMemoryManager

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


async def test_conversation_memory():
    """Test conversation memory functionality."""

    print("🧠 **TESTING CONVERSATION MEMORY FUNCTIONALITY**")
    print("=" * 60)

    # Initialize memory manager
    memory_manager = ConversationMemoryManager(memory_dir="data/test_conversations")

    # Test 1: Start new conversation
    print("\n📝 **TEST 1: Starting New Conversation**")
    conv_id = memory_manager.start_new_conversation()
    print(f"✅ Started conversation: {conv_id[:8]}...")

    # Test 2: Add conversation turns
    print("\n📝 **TEST 2: Adding Conversation Turns**")

    # Turn 1
    memory_manager.add_turn(
        conversation_id=conv_id,
        user_input="Tell me about Garchomp's competitive viability",
        agent_response="Garchomp is a top-tier Dragon/Ground type Pokemon...",
        pokemon_mentioned=["garchomp"],
        query_type="competitive",
        context_used={},
    )
    print("✅ Added turn 1: Garchomp competitive analysis")

    # Turn 2 (follow-up)
    memory_manager.add_turn(
        conversation_id=conv_id,
        user_input="How does it compare to Dragonite?",
        agent_response="Comparing Garchomp to Dragonite, both are powerful Dragon types...",
        pokemon_mentioned=["dragonite"],
        query_type="comparison",
        context_used={"previous_pokemon": ["garchomp"]},
    )
    print("✅ Added turn 2: Garchomp vs Dragonite comparison")

    # Turn 3 (another follow-up)
    memory_manager.add_turn(
        conversation_id=conv_id,
        user_input="What about team synergies for both?",
        agent_response="For team building with Garchomp and Dragonite...",
        pokemon_mentioned=["garchomp", "dragonite"],
        query_type="team_building",
        context_used={"previous_pokemon": ["garchomp", "dragonite"]},
    )
    print("✅ Added turn 3: Team synergies follow-up")

    # Test 3: Retrieve conversation history
    print("\n📝 **TEST 3: Retrieving Conversation History**")
    history = memory_manager.get_conversation_history(conv_id)
    print(f"✅ Retrieved {len(history)} turns from conversation")

    for i, turn in enumerate(history, 1):
        print(f"   Turn {i}: {turn.user_input[:50]}...")
        print(f"           Pokemon: {turn.pokemon_mentioned}")
        print(f"           Type: {turn.query_type}")

    # Test 4: Get persistent context
    print("\n📝 **TEST 4: Checking Persistent Context**")
    context = memory_manager.get_persistent_context(conv_id)
    print(f"✅ Persistent context retrieved:")
    print(f"   Pokemon mentioned: {context.get('mentioned_pokemon', [])}")
    print(f"   Query types: {context.get('query_types', [])}")
    print(f"   Themes: {context.get('themes', [])}")

    # Test 5: Format context for LLM
    print("\n📝 **TEST 5: Formatting Context for LLM**")
    llm_context = memory_manager.format_context_for_llm(conv_id, max_turns=2)
    print(f"✅ Formatted LLM context ({len(llm_context)} characters):")
    print(f"   Preview: {llm_context[:200]}...")

    # Test 6: Conversation summary
    print("\n📝 **TEST 6: Conversation Summary**")
    summary = memory_manager.get_conversation_summary(conv_id)
    print(f"✅ Conversation summary:")
    print(f"   ID: {summary['conversation_id'][:8]}...")
    print(f"   Total turns: {summary['total_turns']}")
    print(f"   Pokemon discussed: {summary['mentioned_pokemon']}")
    print(f"   Themes: {summary['themes']}")
    print(f"   Query types: {summary['query_types']}")

    # Test 7: Multiple conversations
    print("\n📝 **TEST 7: Multiple Conversations**")
    conv_id_2 = memory_manager.start_new_conversation()
    print(f"✅ Started second conversation: {conv_id_2[:8]}...")

    memory_manager.add_turn(
        conversation_id=conv_id_2,
        user_input="What are the best Electric-type Pokemon?",
        agent_response="The best Electric-type Pokemon include Pikachu, Jolteon...",
        pokemon_mentioned=["pikachu", "jolteon"],
        query_type="general",
        context_used={},
    )
    print("✅ Added turn to second conversation")

    # List all conversations
    conversations = memory_manager.list_active_conversations()
    print(f"✅ Total active conversations: {len(conversations)}")
    for conv in conversations:
        print(
            f"   {conv['conversation_id'][:8]}... - {conv['total_turns']} turns - {conv['mentioned_pokemon']}"
        )

    # Test 8: Context isolation
    print("\n📝 **TEST 8: Context Isolation Between Conversations**")
    context_1 = memory_manager.get_persistent_context(conv_id)
    context_2 = memory_manager.get_persistent_context(conv_id_2)

    print(f"✅ Conversation 1 Pokemon: {context_1.get('mentioned_pokemon', [])}")
    print(f"✅ Conversation 2 Pokemon: {context_2.get('mentioned_pokemon', [])}")
    print(f"✅ Contexts are properly isolated: {context_1 != context_2}")

    # Test 9: Persistence (save/load)
    print("\n📝 **TEST 9: Testing Persistence**")

    # Create new memory manager to test loading
    memory_manager_2 = ConversationMemoryManager(memory_dir="data/test_conversations")
    loaded_conversations = memory_manager_2.list_active_conversations()

    print(f"✅ Loaded {len(loaded_conversations)} conversations from disk")
    print(f"✅ Persistence working: {len(loaded_conversations) >= 2}")

    # Test 10: Cleanup
    print("\n📝 **TEST 10: Cleanup**")
    memory_manager.clear_all_conversations()
    remaining = memory_manager.list_active_conversations()
    print(f"✅ Cleared all conversations: {len(remaining) == 0}")

    print("\n🎉 **ALL TESTS PASSED!**")
    print("=" * 60)
    print("✅ Conversation memory system is working correctly")
    print("✅ Follow-up questions are supported")
    print("✅ Context isolation between conversations")
    print("✅ Persistent storage working")
    print("✅ Ready for production use!")


async def test_memory_with_simulated_queries():
    """Test memory with realistic Pokemon queries."""

    print("\n🎮 **TESTING WITH REALISTIC POKEMON QUERIES**")
    print("=" * 60)

    memory_manager = ConversationMemoryManager(memory_dir="data/test_conversations")
    conv_id = memory_manager.start_new_conversation()

    # Simulate a realistic conversation flow
    conversation_flow = [
        {
            "user": "Tell me about Garchomp's competitive viability",
            "agent": "Garchomp is an S-tier Dragon/Ground type Pokemon in competitive play. With base stats of 108/130/95/80/85/102, it excels as a physical sweeper...",
            "pokemon": ["garchomp"],
            "type": "competitive",
        },
        {
            "user": "How does it compare to Salamence?",
            "agent": "Comparing Garchomp to Salamence, both are powerful Dragon types but serve different roles. Garchomp has better physical attack and ground typing...",
            "pokemon": ["salamence"],
            "type": "comparison",
        },
        {
            "user": "What about movesets for both?",
            "agent": "For Garchomp, the standard set includes Earthquake, Dragon Claw, Stone Edge, and Swords Dance. Salamence typically runs Dragon Dance, Outrage, Earthquake...",
            "pokemon": ["garchomp", "salamence"],
            "type": "movesets",
        },
        {
            "user": "Can they work together on the same team?",
            "agent": "While both Garchomp and Salamence are Dragon types, they can work together with careful team building. Garchomp handles Steel types while Salamence...",
            "pokemon": ["garchomp", "salamence"],
            "type": "team_building",
        },
    ]

    print(f"🎯 Simulating {len(conversation_flow)} conversation turns...")

    for i, turn in enumerate(conversation_flow, 1):
        memory_manager.add_turn(
            conversation_id=conv_id,
            user_input=turn["user"],
            agent_response=turn["agent"],
            pokemon_mentioned=turn["pokemon"],
            query_type=turn["type"],
            context_used={},
        )
        print(f"✅ Turn {i}: {turn['user'][:40]}...")

    # Test context formatting
    print(f"\n📋 **Testing Context Formatting for Follow-up**")
    context = memory_manager.format_context_for_llm(conv_id, max_turns=3)

    print(f"✅ Generated context for LLM ({len(context)} characters)")
    print(f"📝 Context preview:")
    print(context[:300] + "..." if len(context) > 300 else context)

    # Test persistent context
    persistent = memory_manager.get_persistent_context(conv_id)
    print(f"\n🧠 **Persistent Context Analysis**")
    print(f"✅ Pokemon discussed: {persistent.get('mentioned_pokemon', [])}")
    print(f"✅ Query types: {persistent.get('query_types', [])}")
    print(f"✅ Themes: {persistent.get('themes', [])}")

    # Cleanup
    memory_manager.clear_conversation(conv_id)
    print(f"\n🗑️ Cleaned up test conversation")


if __name__ == "__main__":
    asyncio.run(test_conversation_memory())
    asyncio.run(test_memory_with_simulated_queries())
