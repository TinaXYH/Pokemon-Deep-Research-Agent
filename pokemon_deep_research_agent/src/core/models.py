"""
Core data models for the Pokémon Deep Research Agent system.

This module defines the fundamental data structures used throughout the system
for tasks, messages, agent communication, and research results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class TaskStatus(Enum):
    """Status of a task in the system."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Priority levels for tasks."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class MessageType(Enum):
    """Types of messages in the system."""

    TASK_ASSIGNMENT = "task_assignment"
    TASK_RESULT = "task_result"
    QUERY_CLARIFICATION = "query_clarification"
    STATUS_UPDATE = "status_update"
    ERROR_REPORT = "error_report"
    COORDINATION = "coordination"


class AgentType(Enum):
    """Types of agents in the system."""

    COORDINATOR = "coordinator"
    TASK_MANAGER = "task_manager"
    POKEAPI_RESEARCHER = "pokeapi_researcher"
    BATTLE_STRATEGIST = "battle_strategist"
    POKEMON_ANALYST = "pokemon_analyst"
    REPORT_GENERATOR = "report_generator"


@dataclass
class Task:
    """Represents a task in the multi-agent system."""

    id: UUID = field(default_factory=uuid4)
    title: str = ""
    description: str = ""
    task_type: str = ""
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    dependencies: List[UUID] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    def update_status(
        self, status: TaskStatus, error_message: Optional[str] = None
    ) -> None:
        """Update task status and timestamp."""
        self.status = status
        self.updated_at = datetime.now()
        if status == TaskStatus.COMPLETED:
            self.completed_at = datetime.now()
        if error_message:
            self.error_message = error_message


@dataclass
class Message:
    """Represents a message between agents."""

    id: UUID = field(default_factory=uuid4)
    sender: str = ""
    recipient: str = ""
    message_type: MessageType = MessageType.COORDINATION
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    task_id: Optional[UUID] = None
    requires_response: bool = False
    response_to: Optional[UUID] = None


class PokemonData(BaseModel):
    """Represents Pokémon data from PokéAPI."""

    id: int
    name: str
    height: int
    weight: int
    base_experience: Optional[int] = None
    types: List[Dict[str, Any]] = Field(default_factory=list)
    abilities: List[Dict[str, Any]] = Field(default_factory=list)
    stats: List[Dict[str, Any]] = Field(default_factory=list)
    moves: List[Dict[str, Any]] = Field(default_factory=list)
    sprites: Dict[str, Any] = Field(default_factory=dict)
    species: Dict[str, Any] = Field(default_factory=dict)


class MoveData(BaseModel):
    """Represents move data from PokéAPI."""

    id: int
    name: str
    accuracy: Optional[int] = None
    power: Optional[int] = None
    pp: int
    priority: int
    damage_class: Dict[str, Any] = Field(default_factory=dict)
    type: Dict[str, Any] = Field(default_factory=dict)
    effect_entries: List[Dict[str, Any]] = Field(default_factory=list)
    target: Dict[str, Any] = Field(default_factory=dict)


class TypeData(BaseModel):
    """Represents type data from PokéAPI."""

    id: int
    name: str
    damage_relations: Dict[str, Any] = Field(default_factory=dict)
    pokemon: List[Dict[str, Any]] = Field(default_factory=list)
    moves: List[Dict[str, Any]] = Field(default_factory=list)


@dataclass
class ResearchQuery:
    """Represents a user research query."""

    id: UUID = field(default_factory=uuid4)
    original_query: str = ""
    clarified_query: str = ""
    query_type: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    status: TaskStatus = TaskStatus.PENDING


@dataclass
class ResearchResult:
    """Represents the result of a research operation."""

    query_id: UUID
    agent_id: str
    result_type: str
    data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    sources: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentCapability:
    """Represents an agent's capability."""

    name: str
    description: str
    input_types: List[str] = field(default_factory=list)
    output_types: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class AgentConfig:
    """Configuration for an agent."""

    agent_id: str
    agent_type: AgentType
    name: str
    description: str
    capabilities: List[AgentCapability] = field(default_factory=list)
    max_concurrent_tasks: int = 5
    retry_attempts: int = 3
    timeout_seconds: int = 300
    config: Dict[str, Any] = field(default_factory=dict)


class TeamRecommendation(BaseModel):
    """Represents a team building recommendation."""

    pokemon_list: List[PokemonData]
    team_analysis: Dict[str, Any] = Field(default_factory=dict)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    strategy_notes: str = ""
    confidence_score: float = 0.0
    alternative_options: List[Dict[str, Any]] = Field(default_factory=list)


class BattleAnalysis(BaseModel):
    """Represents a battle strategy analysis."""

    pokemon_name: str
    recommended_moves: List[str] = Field(default_factory=list)
    item_suggestions: List[str] = Field(default_factory=list)
    ev_spread: Dict[str, int] = Field(default_factory=dict)
    nature_recommendation: str = ""
    role_description: str = ""
    matchup_analysis: Dict[str, Any] = Field(default_factory=dict)
    usage_statistics: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System performance and health metrics."""

    total_queries_processed: int = 0
    average_response_time: float = 0.0
    api_calls_made: int = 0
    cache_hit_rate: float = 0.0
    active_agents: int = 0
    failed_tasks: int = 0
    uptime_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
