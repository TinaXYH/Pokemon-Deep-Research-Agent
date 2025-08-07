"""
Conversation Memory Manager for Pokemon Deep Research Agent

Provides conversation-level memory and context management to support
follow-up questions and maintain conversation history within sessions.
"""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""

    turn_id: str
    timestamp: datetime
    user_input: str
    agent_response: str
    pokemon_mentioned: List[str]
    query_type: str
    context_used: Dict[str, Any]


@dataclass
class ConversationContext:
    """Represents the context and state of a conversation."""

    conversation_id: str
    created_at: datetime
    last_updated: datetime
    turns: List[ConversationTurn]
    persistent_context: Dict[str, Any]  # Carries forward between turns

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "conversation_id": self.conversation_id,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "turns": [
                {
                    "turn_id": turn.turn_id,
                    "timestamp": turn.timestamp.isoformat(),
                    "user_input": turn.user_input,
                    "agent_response": turn.agent_response,
                    "pokemon_mentioned": turn.pokemon_mentioned,
                    "query_type": turn.query_type,
                    "context_used": turn.context_used,
                }
                for turn in self.turns
            ],
            "persistent_context": self.persistent_context,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationContext":
        """Create from dictionary."""
        turns = [
            ConversationTurn(
                turn_id=turn_data["turn_id"],
                timestamp=datetime.fromisoformat(turn_data["timestamp"]),
                user_input=turn_data["user_input"],
                agent_response=turn_data["agent_response"],
                pokemon_mentioned=turn_data["pokemon_mentioned"],
                query_type=turn_data["query_type"],
                context_used=turn_data["context_used"],
            )
            for turn_data in data["turns"]
        ]

        return cls(
            conversation_id=data["conversation_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            turns=turns,
            persistent_context=data["persistent_context"],
        )


class ConversationMemoryManager:
    """
    Manages conversation memory and context for Pokemon Deep Research Agent.

    Features:
    - Session-level conversation memory
    - Context persistence across turns
    - Follow-up question support
    - Conversation history management
    - Automatic context summarization
    """

    def __init__(self, memory_dir: str = "data/conversations"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)

        # Active conversations in memory
        self.active_conversations: Dict[str, ConversationContext] = {}

        # Load existing conversations
        self._load_conversations()

    def _load_conversations(self) -> None:
        """Load existing conversations from disk."""
        try:
            for conv_file in self.memory_dir.glob("*.json"):
                with open(conv_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    context = ConversationContext.from_dict(data)
                    self.active_conversations[context.conversation_id] = context

            self.logger.info(f"Loaded {len(self.active_conversations)} conversations")
        except Exception as e:
            self.logger.error(f"Error loading conversations: {e}")

    def _save_conversation(self, conversation_id: str) -> None:
        """Save a conversation to disk."""
        try:
            if conversation_id in self.active_conversations:
                context = self.active_conversations[conversation_id]
                file_path = self.memory_dir / f"{conversation_id}.json"

                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(context.to_dict(), f, indent=2, ensure_ascii=False)

                self.logger.debug(f"Saved conversation {conversation_id}")
        except Exception as e:
            self.logger.error(f"Error saving conversation {conversation_id}: {e}")

    def start_new_conversation(self) -> str:
        """Start a new conversation and return its ID."""
        conversation_id = str(uuid4())

        context = ConversationContext(
            conversation_id=conversation_id,
            created_at=datetime.now(),
            last_updated=datetime.now(),
            turns=[],
            persistent_context={},
        )

        self.active_conversations[conversation_id] = context
        self._save_conversation(conversation_id)

        self.logger.info(f"Started new conversation: {conversation_id}")
        return conversation_id

    def add_turn(
        self,
        conversation_id: str,
        user_input: str,
        agent_response: str,
        pokemon_mentioned: List[str] = None,
        query_type: str = "general",
        context_used: Dict[str, Any] = None,
    ) -> None:
        """Add a new turn to the conversation."""
        if conversation_id not in self.active_conversations:
            raise ValueError(f"Conversation {conversation_id} not found")

        turn = ConversationTurn(
            turn_id=str(uuid4()),
            timestamp=datetime.now(),
            user_input=user_input,
            agent_response=agent_response,
            pokemon_mentioned=pokemon_mentioned or [],
            query_type=query_type,
            context_used=context_used or {},
        )

        context = self.active_conversations[conversation_id]
        context.turns.append(turn)
        context.last_updated = datetime.now()

        # Update persistent context
        self._update_persistent_context(conversation_id, turn)

        self._save_conversation(conversation_id)
        self.logger.debug(f"Added turn to conversation {conversation_id}")

    def _update_persistent_context(
        self, conversation_id: str, turn: ConversationTurn
    ) -> None:
        """Update persistent context based on the new turn."""
        context = self.active_conversations[conversation_id]

        # Track mentioned Pokemon across the conversation
        if "mentioned_pokemon" not in context.persistent_context:
            context.persistent_context["mentioned_pokemon"] = []

        # Convert to set for deduplication, then back to list
        mentioned_set = set(context.persistent_context["mentioned_pokemon"])
        for pokemon in turn.pokemon_mentioned:
            mentioned_set.add(pokemon.lower())

        context.persistent_context["mentioned_pokemon"] = list(mentioned_set)

        # Track query types
        if "query_types" not in context.persistent_context:
            context.persistent_context["query_types"] = []

        if turn.query_type not in context.persistent_context["query_types"]:
            context.persistent_context["query_types"].append(turn.query_type)

        # Track conversation themes
        if "themes" not in context.persistent_context:
            context.persistent_context["themes"] = []

        # Simple theme detection based on keywords
        themes = self._extract_themes(turn.user_input)
        for theme in themes:
            if theme not in context.persistent_context["themes"]:
                context.persistent_context["themes"].append(theme)

    def _extract_themes(self, user_input: str) -> List[str]:
        """Extract themes from user input."""
        themes = []
        input_lower = user_input.lower()

        theme_keywords = {
            "competitive": [
                "competitive",
                "tier",
                "meta",
                "tournament",
                "ranked",
                "ladder",
            ],
            "team_building": ["team", "build", "synergy", "composition", "partner"],
            "comparison": ["compare", "vs", "versus", "better", "difference"],
            "movesets": ["moveset", "moves", "attacks", "abilities"],
            "stats": ["stats", "base stats", "iv", "ev", "nature"],
            "strategy": ["strategy", "tactics", "counter", "weakness"],
        }

        for theme, keywords in theme_keywords.items():
            if any(keyword in input_lower for keyword in keywords):
                themes.append(theme)

        return themes

    def get_conversation_history(
        self, conversation_id: str, max_turns: int = 10
    ) -> List[ConversationTurn]:
        """Get conversation history for context."""
        if conversation_id not in self.active_conversations:
            return []

        context = self.active_conversations[conversation_id]
        return context.turns[-max_turns:] if max_turns > 0 else context.turns

    def get_persistent_context(self, conversation_id: str) -> Dict[str, Any]:
        """Get persistent context for the conversation."""
        if conversation_id not in self.active_conversations:
            return {}

        return self.active_conversations[conversation_id].persistent_context.copy()

    def format_context_for_llm(self, conversation_id: str, max_turns: int = 5) -> str:
        """Format conversation context for LLM prompt."""
        if conversation_id not in self.active_conversations:
            return ""

        context = self.active_conversations[conversation_id]
        recent_turns = context.turns[-max_turns:] if max_turns > 0 else context.turns

        if not recent_turns:
            return ""

        # Format conversation history
        history_text = "Previous conversation context:\n"

        for i, turn in enumerate(recent_turns[:-1], 1):  # Exclude current turn
            history_text += f"\nTurn {i}:\n"
            history_text += f"User: {turn.user_input}\n"
            history_text += f"Assistant: {turn.agent_response[:200]}...\n"

        # Add persistent context
        persistent = context.persistent_context
        if persistent.get("mentioned_pokemon"):
            history_text += (
                f"\nPokemon discussed: {', '.join(persistent['mentioned_pokemon'])}\n"
            )

        if persistent.get("themes"):
            history_text += f"Conversation themes: {', '.join(persistent['themes'])}\n"

        history_text += (
            "\nCurrent query (use above context to provide relevant follow-up):\n"
        )

        return history_text

    def clear_conversation(self, conversation_id: str) -> None:
        """Clear a specific conversation."""
        if conversation_id in self.active_conversations:
            del self.active_conversations[conversation_id]

            # Remove file
            file_path = self.memory_dir / f"{conversation_id}.json"
            if file_path.exists():
                file_path.unlink()

            self.logger.info(f"Cleared conversation {conversation_id}")

    def clear_all_conversations(self) -> None:
        """Clear all conversations."""
        for conversation_id in list(self.active_conversations.keys()):
            self.clear_conversation(conversation_id)

        self.logger.info("Cleared all conversations")

    def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """Get a summary of the conversation."""
        if conversation_id not in self.active_conversations:
            return {}

        context = self.active_conversations[conversation_id]

        return {
            "conversation_id": conversation_id,
            "created_at": context.created_at.isoformat(),
            "last_updated": context.last_updated.isoformat(),
            "total_turns": len(context.turns),
            "mentioned_pokemon": context.persistent_context.get(
                "mentioned_pokemon", []
            ),
            "themes": context.persistent_context.get("themes", []),
            "query_types": context.persistent_context.get("query_types", []),
        }

    def list_active_conversations(self) -> List[Dict[str, Any]]:
        """List all active conversations."""
        return [
            self.get_conversation_summary(conv_id)
            for conv_id in self.active_conversations.keys()
        ]
