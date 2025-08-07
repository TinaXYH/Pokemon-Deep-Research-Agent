"""
Vector-based memory system for Pokemon Deep Research Agent.

This module implements a sophisticated memory system using vector embeddings
for storing and retrieving Pokemon research data, user preferences, and
conversation history.
"""

import asyncio
import json
import logging
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """Represents a single memory entry with vector embedding."""

    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    timestamp: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "access_count": self.access_count,
            "last_accessed": (
                self.last_accessed.isoformat() if self.last_accessed else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            embedding=data["embedding"],
            metadata=data["metadata"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            access_count=data.get("access_count", 0),
            last_accessed=(
                datetime.fromisoformat(data["last_accessed"])
                if data.get("last_accessed")
                else None
            ),
        )


class VectorMemorySystem:
    """
    Advanced vector-based memory system for Pokemon research.

    Features:
    - Vector embeddings for semantic similarity search
    - User preference learning and adaptation
    - Conversation history with context awareness
    - Pokemon research knowledge accumulation
    - Automatic memory consolidation and cleanup
    """

    def __init__(
        self,
        openai_client: AsyncOpenAI,
        memory_dir: str = "data/memory",
        embedding_model: str = "text-embedding-3-small",
        max_memory_entries: int = 10000,
        similarity_threshold: float = 0.7,
    ):
        """
        Initialize the vector memory system.

        Args:
            openai_client: OpenAI client for generating embeddings
            memory_dir: Directory to store memory files
            embedding_model: OpenAI embedding model to use
            max_memory_entries: Maximum number of memory entries to keep
            similarity_threshold: Minimum similarity for memory retrieval
        """
        self.openai_client = openai_client
        self.memory_dir = Path(memory_dir)
        self.embedding_model = embedding_model
        self.max_memory_entries = max_memory_entries
        self.similarity_threshold = similarity_threshold

        # In-memory storage for fast access
        self.memories: Dict[str, MemoryEntry] = {}
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.memory_ids: List[str] = []

        # Memory categories
        self.user_preferences: Dict[str, Any] = {}
        self.conversation_history: List[Dict[str, Any]] = []
        self.pokemon_knowledge: Dict[str, Any] = {}

        # Ensure memory directory exists
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized VectorMemorySystem with {embedding_model}")

    async def initialize(self) -> None:
        """Initialize the memory system and load existing memories."""
        try:
            await self._load_memories()
            await self._load_user_preferences()
            await self._load_conversation_history()
            await self._load_pokemon_knowledge()

            if self.memories:
                await self._rebuild_embeddings_matrix()
                logger.info(f"Loaded {len(self.memories)} memory entries")
            else:
                logger.info("Starting with empty memory system")

        except Exception as e:
            logger.error(f"Error initializing memory system: {e}")
            # Continue with empty memory if loading fails
            self.memories = {}
            self.embeddings_matrix = None
            self.memory_ids = []

    async def store_memory(
        self,
        content: str,
        memory_type: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a new memory with vector embedding.

        Args:
            content: Text content to store
            memory_type: Type of memory (user_preference, conversation, pokemon_data, etc.)
            metadata: Additional metadata to store with the memory

        Returns:
            Memory ID of the stored entry
        """
        try:
            # Generate embedding for the content
            embedding = await self._generate_embedding(content)

            # Create memory entry
            memory_id = f"{memory_type}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            memory_entry = MemoryEntry(
                id=memory_id,
                content=content,
                embedding=embedding,
                metadata=metadata or {},
                timestamp=datetime.now(),
            )

            # Store in memory
            self.memories[memory_id] = memory_entry

            # Update embeddings matrix
            await self._add_to_embeddings_matrix(memory_entry)

            # Handle memory type specific storage
            if memory_type == "user_preference":
                await self._update_user_preferences(content, metadata)
            elif memory_type == "conversation":
                await self._add_to_conversation_history(content, metadata)
            elif memory_type == "pokemon_knowledge":
                await self._update_pokemon_knowledge(content, metadata)

            # Cleanup old memories if needed
            await self._cleanup_old_memories()

            # Persist to disk
            await self._save_memory(memory_entry)

            logger.debug(f"Stored memory: {memory_id}")
            return memory_id

        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            raise

    async def retrieve_similar_memories(
        self,
        query: str,
        memory_type: Optional[str] = None,
        limit: int = 5,
        min_similarity: Optional[float] = None,
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        Retrieve memories similar to the query using vector similarity.

        Args:
            query: Query text to find similar memories for
            memory_type: Filter by memory type (optional)
            limit: Maximum number of memories to return
            min_similarity: Minimum similarity threshold (optional)

        Returns:
            List of (memory_entry, similarity_score) tuples
        """
        try:
            if not self.memories or self.embeddings_matrix is None:
                return []

            # Generate embedding for query
            query_embedding = await self._generate_embedding(query)
            query_vector = np.array(query_embedding).reshape(1, -1)

            # Calculate similarities
            similarities = np.dot(self.embeddings_matrix, query_vector.T).flatten()

            # Get top similar memories
            threshold = min_similarity or self.similarity_threshold
            similar_indices = np.where(similarities >= threshold)[0]

            if len(similar_indices) == 0:
                return []

            # Sort by similarity
            sorted_indices = similar_indices[
                np.argsort(similarities[similar_indices])[::-1]
            ]

            results = []
            for idx in sorted_indices[:limit]:
                memory_id = self.memory_ids[idx]
                memory_entry = self.memories[memory_id]

                # Filter by memory type if specified
                if memory_type and not memory_id.startswith(memory_type):
                    continue

                # Update access statistics
                memory_entry.access_count += 1
                memory_entry.last_accessed = datetime.now()

                results.append((memory_entry, float(similarities[idx])))

            logger.debug(
                f"Retrieved {len(results)} similar memories for query: {query[:50]}..."
            )
            return results

        except Exception as e:
            logger.error(f"Error retrieving similar memories: {e}")
            return []

    async def get_user_preferences(self) -> Dict[str, Any]:
        """Get current user preferences."""
        return self.user_preferences.copy()

    async def update_user_preference(self, key: str, value: Any) -> None:
        """Update a specific user preference."""
        self.user_preferences[key] = value

        # Store as memory for future learning
        content = f"User preference: {key} = {value}"
        await self.store_memory(
            content=content,
            memory_type="user_preference",
            metadata={"preference_key": key, "preference_value": value},
        )

        await self._save_user_preferences()

    async def get_conversation_context(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation history for context."""
        return self.conversation_history[-limit:] if self.conversation_history else []

    async def add_conversation_turn(
        self,
        user_query: str,
        agent_response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a conversation turn to history."""
        conversation_turn = {
            "timestamp": datetime.now().isoformat(),
            "user_query": user_query,
            "agent_response": agent_response,
            "metadata": metadata or {},
        }

        self.conversation_history.append(conversation_turn)

        # Store as memory
        content = f"User: {user_query}\nAgent: {agent_response}"
        await self.store_memory(
            content=content, memory_type="conversation", metadata=conversation_turn
        )

        # Keep only recent conversations in memory
        if len(self.conversation_history) > 100:
            self.conversation_history = self.conversation_history[-50:]

        await self._save_conversation_history()

    async def learn_from_interaction(
        self,
        user_query: str,
        user_feedback: Optional[str] = None,
        successful: bool = True,
    ) -> None:
        """Learn from user interactions to improve future responses."""
        try:
            # Analyze query patterns
            similar_memories = await self.retrieve_similar_memories(
                user_query, memory_type="conversation", limit=3
            )

            # Update user preferences based on query patterns
            if "competitive" in user_query.lower():
                await self.update_user_preference("prefers_competitive_analysis", True)
            elif "beginner" in user_query.lower():
                await self.update_user_preference("experience_level", "beginner")
            elif "team" in user_query.lower():
                await self.update_user_preference("interested_in_team_building", True)

            # Store learning metadata
            learning_data = {
                "query_type": self._classify_query_type(user_query),
                "successful": successful,
                "feedback": user_feedback,
                "similar_queries_count": len(similar_memories),
            }

            await self.store_memory(
                content=f"Learning from interaction: {user_query}",
                memory_type="learning",
                metadata=learning_data,
            )

        except Exception as e:
            logger.error(f"Error learning from interaction: {e}")

    async def get_pokemon_knowledge(
        self, pokemon_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get accumulated knowledge about a specific Pokemon."""
        return self.pokemon_knowledge.get(pokemon_name.lower())

    async def update_pokemon_knowledge(
        self, pokemon_name: str, knowledge: Dict[str, Any]
    ) -> None:
        """Update knowledge about a specific Pokemon."""
        pokemon_key = pokemon_name.lower()

        if pokemon_key not in self.pokemon_knowledge:
            self.pokemon_knowledge[pokemon_key] = {}

        self.pokemon_knowledge[pokemon_key].update(knowledge)

        # Store as memory
        content = f"Pokemon knowledge for {pokemon_name}: {json.dumps(knowledge)}"
        await self.store_memory(
            content=content,
            memory_type="pokemon_knowledge",
            metadata={"pokemon": pokemon_name, "knowledge": knowledge},
        )

        await self._save_pokemon_knowledge()

    async def consolidate_memories(self) -> None:
        """Consolidate and optimize memory storage."""
        try:
            logger.info("Starting memory consolidation...")

            # Remove duplicate memories
            await self._remove_duplicate_memories()

            # Archive old, rarely accessed memories
            await self._archive_old_memories()

            # Rebuild embeddings matrix for efficiency
            await self._rebuild_embeddings_matrix()

            # Save consolidated state
            await self._save_all_memories()

            logger.info(
                f"Memory consolidation complete. {len(self.memories)} memories retained."
            )

        except Exception as e:
            logger.error(f"Error during memory consolidation: {e}")

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI API."""
        try:
            response = await self.openai_client.embeddings.create(
                model=self.embedding_model, input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * 1536  # text-embedding-3-small dimension

    async def _add_to_embeddings_matrix(self, memory_entry: MemoryEntry) -> None:
        """Add new memory embedding to the matrix."""
        try:
            new_embedding = np.array(memory_entry.embedding).reshape(1, -1)

            if self.embeddings_matrix is None:
                self.embeddings_matrix = new_embedding
                self.memory_ids = [memory_entry.id]
            else:
                self.embeddings_matrix = np.vstack(
                    [self.embeddings_matrix, new_embedding]
                )
                self.memory_ids.append(memory_entry.id)

        except Exception as e:
            logger.error(f"Error adding to embeddings matrix: {e}")

    async def _rebuild_embeddings_matrix(self) -> None:
        """Rebuild the embeddings matrix from current memories."""
        try:
            if not self.memories:
                self.embeddings_matrix = None
                self.memory_ids = []
                return

            embeddings = []
            memory_ids = []

            for memory_id, memory_entry in self.memories.items():
                embeddings.append(memory_entry.embedding)
                memory_ids.append(memory_id)

            self.embeddings_matrix = np.array(embeddings)
            self.memory_ids = memory_ids

            logger.debug(f"Rebuilt embeddings matrix with {len(embeddings)} entries")

        except Exception as e:
            logger.error(f"Error rebuilding embeddings matrix: {e}")

    def _classify_query_type(self, query: str) -> str:
        """Classify the type of user query."""
        query_lower = query.lower()

        if any(
            word in query_lower for word in ["compare", "vs", "versus", "difference"]
        ):
            return "comparison"
        elif any(word in query_lower for word in ["team", "build", "composition"]):
            return "team_building"
        elif any(
            word in query_lower for word in ["competitive", "tier", "meta", "viable"]
        ):
            return "competitive_analysis"
        elif any(word in query_lower for word in ["beginner", "new", "start", "learn"]):
            return "beginner_guidance"
        elif any(word in query_lower for word in ["move", "ability", "stats"]):
            return "technical_analysis"
        else:
            return "general"

    async def _update_user_preferences(
        self, content: str, metadata: Dict[str, Any]
    ) -> None:
        """Update user preferences based on memory content."""
        if "preference_key" in metadata and "preference_value" in metadata:
            self.user_preferences[metadata["preference_key"]] = metadata[
                "preference_value"
            ]

    async def _add_to_conversation_history(
        self, content: str, metadata: Dict[str, Any]
    ) -> None:
        """Add to conversation history."""
        if "user_query" in metadata and "agent_response" in metadata:
            self.conversation_history.append(metadata)

    async def _update_pokemon_knowledge(
        self, content: str, metadata: Dict[str, Any]
    ) -> None:
        """Update Pokemon knowledge base."""
        if "pokemon" in metadata and "knowledge" in metadata:
            pokemon_key = metadata["pokemon"].lower()
            if pokemon_key not in self.pokemon_knowledge:
                self.pokemon_knowledge[pokemon_key] = {}
            self.pokemon_knowledge[pokemon_key].update(metadata["knowledge"])

    async def _cleanup_old_memories(self) -> None:
        """Remove old memories if we exceed the limit."""
        if len(self.memories) <= self.max_memory_entries:
            return

        # Sort by last accessed time and access count
        memory_items = list(self.memories.items())
        memory_items.sort(
            key=lambda x: (x[1].last_accessed or x[1].timestamp, x[1].access_count)
        )

        # Remove oldest, least accessed memories
        to_remove = len(self.memories) - self.max_memory_entries
        for i in range(to_remove):
            memory_id = memory_items[i][0]
            del self.memories[memory_id]

        # Rebuild embeddings matrix
        await self._rebuild_embeddings_matrix()

    async def _remove_duplicate_memories(self) -> None:
        """Remove duplicate memories based on high similarity."""
        if len(self.memories) < 2:
            return

        to_remove = set()
        memory_list = list(self.memories.items())

        for i in range(len(memory_list)):
            if memory_list[i][0] in to_remove:
                continue

            for j in range(i + 1, len(memory_list)):
                if memory_list[j][0] in to_remove:
                    continue

                # Calculate similarity between embeddings
                emb1 = np.array(memory_list[i][1].embedding)
                emb2 = np.array(memory_list[j][1].embedding)
                similarity = np.dot(emb1, emb2) / (
                    np.linalg.norm(emb1) * np.linalg.norm(emb2)
                )

                if similarity > 0.95:  # Very high similarity threshold for duplicates
                    # Keep the one with higher access count
                    if memory_list[i][1].access_count >= memory_list[j][1].access_count:
                        to_remove.add(memory_list[j][0])
                    else:
                        to_remove.add(memory_list[i][0])

        # Remove duplicates
        for memory_id in to_remove:
            del self.memories[memory_id]

        if to_remove:
            logger.info(f"Removed {len(to_remove)} duplicate memories")

    async def _archive_old_memories(self) -> None:
        """Archive old, rarely accessed memories."""
        cutoff_date = datetime.now() - timedelta(days=30)
        to_archive = []

        for memory_id, memory_entry in self.memories.items():
            if memory_entry.timestamp < cutoff_date and memory_entry.access_count < 2:
                to_archive.append(memory_id)

        if to_archive:
            # Save to archive file
            archive_file = self.memory_dir / "archived_memories.json"
            archived_data = []

            if archive_file.exists():
                with open(archive_file, "r") as f:
                    archived_data = json.load(f)

            for memory_id in to_archive:
                archived_data.append(self.memories[memory_id].to_dict())
                del self.memories[memory_id]

            with open(archive_file, "w") as f:
                json.dump(archived_data, f, indent=2)

            logger.info(f"Archived {len(to_archive)} old memories")

    # File I/O methods
    async def _save_memory(self, memory_entry: MemoryEntry) -> None:
        """Save a single memory entry to disk."""
        try:
            memory_file = self.memory_dir / f"{memory_entry.id}.json"
            with open(memory_file, "w") as f:
                json.dump(memory_entry.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving memory {memory_entry.id}: {e}")

    async def _load_memories(self) -> None:
        """Load all memory entries from disk."""
        try:
            for memory_file in self.memory_dir.glob("*.json"):
                if memory_file.name.startswith(
                    (
                        "user_preferences",
                        "conversation_history",
                        "pokemon_knowledge",
                        "archived",
                    )
                ):
                    continue

                try:
                    with open(memory_file, "r") as f:
                        data = json.load(f)

                    memory_entry = MemoryEntry.from_dict(data)
                    self.memories[memory_entry.id] = memory_entry

                except Exception as e:
                    logger.warning(f"Error loading memory file {memory_file}: {e}")

        except Exception as e:
            logger.error(f"Error loading memories: {e}")

    async def _save_all_memories(self) -> None:
        """Save all memories to disk."""
        for memory_entry in self.memories.values():
            await self._save_memory(memory_entry)

    async def _save_user_preferences(self) -> None:
        """Save user preferences to disk."""
        try:
            prefs_file = self.memory_dir / "user_preferences.json"
            with open(prefs_file, "w") as f:
                json.dump(self.user_preferences, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving user preferences: {e}")

    async def _load_user_preferences(self) -> None:
        """Load user preferences from disk."""
        try:
            prefs_file = self.memory_dir / "user_preferences.json"
            if prefs_file.exists():
                with open(prefs_file, "r") as f:
                    self.user_preferences = json.load(f)
        except Exception as e:
            logger.error(f"Error loading user preferences: {e}")

    async def _save_conversation_history(self) -> None:
        """Save conversation history to disk."""
        try:
            conv_file = self.memory_dir / "conversation_history.json"
            with open(conv_file, "w") as f:
                json.dump(self.conversation_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving conversation history: {e}")

    async def _load_conversation_history(self) -> None:
        """Load conversation history from disk."""
        try:
            conv_file = self.memory_dir / "conversation_history.json"
            if conv_file.exists():
                with open(conv_file, "r") as f:
                    self.conversation_history = json.load(f)
        except Exception as e:
            logger.error(f"Error loading conversation history: {e}")

    async def _save_pokemon_knowledge(self) -> None:
        """Save Pokemon knowledge to disk."""
        try:
            knowledge_file = self.memory_dir / "pokemon_knowledge.json"
            with open(knowledge_file, "w") as f:
                json.dump(self.pokemon_knowledge, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving Pokemon knowledge: {e}")

    async def _load_pokemon_knowledge(self) -> None:
        """Load Pokemon knowledge from disk."""
        try:
            knowledge_file = self.memory_dir / "pokemon_knowledge.json"
            if knowledge_file.exists():
                with open(knowledge_file, "r") as f:
                    self.pokemon_knowledge = json.load(f)
        except Exception as e:
            logger.error(f"Error loading Pokemon knowledge: {e}")

    async def close(self) -> None:
        """Clean up resources and save final state."""
        try:
            await self._save_all_memories()
            await self._save_user_preferences()
            await self._save_conversation_history()
            await self._save_pokemon_knowledge()
            logger.info("Vector memory system closed successfully")
        except Exception as e:
            logger.error(f"Error closing memory system: {e}")


# Convenience functions for easy integration
async def create_vector_memory_system(
    openai_client: AsyncOpenAI, config: Dict[str, Any]
) -> VectorMemorySystem:
    """Create and initialize a vector memory system."""
    memory_system = VectorMemorySystem(
        openai_client=openai_client,
        memory_dir=config.get("memory_dir", "data/memory"),
        embedding_model=config.get("embedding_model", "text-embedding-3-small"),
        max_memory_entries=config.get("max_memory_entries", 10000),
        similarity_threshold=config.get("similarity_threshold", 0.7),
    )

    await memory_system.initialize()
    return memory_system
