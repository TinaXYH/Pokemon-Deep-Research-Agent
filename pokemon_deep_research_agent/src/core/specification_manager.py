"""
Specification Manager for Pokemon Deep Research Agent

Implements dynamic query disambiguation and specification confirmation
following ChatGPT's deep research system pattern.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class QueryType(Enum):
    """Types of Pokemon queries that require different specifications."""

    COMPETITIVE_ANALYSIS = "competitive_analysis"
    TEAM_BUILDING = "team_building"
    POKEMON_COMPARISON = "pokemon_comparison"
    MOVESET_ANALYSIS = "moveset_analysis"
    META_ANALYSIS = "meta_analysis"
    TYPE_ANALYSIS = "type_analysis"
    GENERAL_INFO = "general_info"
    BATTLE_STRATEGY = "battle_strategy"


@dataclass
class SpecificationField:
    """Represents a field in a specification schema."""

    name: str
    description: str
    field_type: str  # "select", "boolean", "text", "pokemon_list"
    required: bool
    default_value: Any = None
    options: List[str] = None  # For select fields
    question: str = ""  # Natural language question to ask user


@dataclass
class TaskSpecification:
    """Complete specification for a Pokemon research task."""

    query_type: QueryType
    original_query: str
    fields: Dict[str, Any]
    confirmed: bool = False
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query_type": self.query_type.value,
            "original_query": self.original_query,
            "fields": self.fields,
            "confirmed": self.confirmed,
            "created_at": self.created_at.isoformat(),
        }


class SpecificationManager:
    """
    Manages specification generation, confirmation, and disambiguation.

    Features:
    - Dynamic specification schema based on query type
    - Interactive confirmation workflow
    - Query disambiguation and field elicitation
    - Integration with conversation memory
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)

        # Define specification schemas for different query types
        self.schemas = self._initialize_schemas()

    def _initialize_schemas(self) -> Dict[QueryType, List[SpecificationField]]:
        """Initialize specification schemas for different query types."""

        schemas = {}

        # Competitive Analysis Schema
        schemas[QueryType.COMPETITIVE_ANALYSIS] = [
            SpecificationField(
                name="target_pokemon",
                description="Pokemon to analyze",
                field_type="text",
                required=True,
                question="Which Pokemon would you like me to analyze?",
            ),
            SpecificationField(
                name="format",
                description="Competitive format",
                field_type="select",
                required=True,
                options=[
                    "OU",
                    "UU",
                    "RU",
                    "NU",
                    "PU",
                    "VGC",
                    "National Dex",
                    "Little Cup",
                ],
                default_value="OU",
                question="What competitive format should I focus on?",
            ),
            SpecificationField(
                name="analysis_depth",
                description="Depth of analysis",
                field_type="select",
                required=False,
                options=["basic", "detailed", "comprehensive"],
                default_value="detailed",
                question="How detailed should the analysis be?",
            ),
            SpecificationField(
                name="include_counters",
                description="Include counter analysis",
                field_type="boolean",
                required=False,
                default_value=True,
                question="Should I include counter Pokemon and strategies?",
            ),
        ]

        # Team Building Schema
        schemas[QueryType.TEAM_BUILDING] = [
            SpecificationField(
                name="core_pokemon",
                description="Core Pokemon for the team",
                field_type="text",
                required=True,
                question="Which Pokemon should be the core of the team?",
            ),
            SpecificationField(
                name="format",
                description="Competitive format",
                field_type="select",
                required=True,
                options=["OU", "UU", "RU", "VGC", "National Dex", "Little Cup"],
                default_value="OU",
                question="What competitive format is this team for?",
            ),
            SpecificationField(
                name="team_style",
                description="Team playstyle",
                field_type="select",
                required=True,
                options=[
                    "hyper_offense",
                    "balanced_offense",
                    "balanced",
                    "bulky_offense",
                    "stall",
                ],
                default_value="balanced",
                question="What playstyle do you prefer?",
            ),
            SpecificationField(
                name="include_legendaries",
                description="Allow legendary Pokemon",
                field_type="boolean",
                required=False,
                default_value=False,
                question="Should the team include legendary Pokemon?",
            ),
            SpecificationField(
                name="hazard_support",
                description="Include hazard support",
                field_type="boolean",
                required=False,
                default_value=True,
                question="Do you want hazard setters on the team?",
            ),
            SpecificationField(
                name="hazard_removal",
                description="Include hazard removal",
                field_type="boolean",
                required=False,
                default_value=True,
                question="Do you want hazard removal on the team?",
            ),
        ]

        # Pokemon Comparison Schema
        schemas[QueryType.POKEMON_COMPARISON] = [
            SpecificationField(
                name="pokemon_list",
                description="Pokemon to compare",
                field_type="pokemon_list",
                required=True,
                question="Which Pokemon would you like me to compare?",
            ),
            SpecificationField(
                name="comparison_aspects",
                description="Aspects to compare",
                field_type="select",
                required=True,
                options=["stats", "competitive_viability", "movesets", "roles", "all"],
                default_value="all",
                question="What aspects should I compare?",
            ),
            SpecificationField(
                name="format",
                description="Competitive format context",
                field_type="select",
                required=False,
                options=["OU", "UU", "RU", "VGC", "National Dex", "general"],
                default_value="general",
                question="In what competitive context should I compare them?",
            ),
        ]

        # Moveset Analysis Schema
        schemas[QueryType.MOVESET_ANALYSIS] = [
            SpecificationField(
                name="target_pokemon",
                description="Pokemon for moveset analysis",
                field_type="text",
                required=True,
                question="Which Pokemon's movesets would you like me to analyze?",
            ),
            SpecificationField(
                name="format",
                description="Competitive format",
                field_type="select",
                required=True,
                options=["OU", "UU", "RU", "VGC", "National Dex"],
                default_value="OU",
                question="For which competitive format?",
            ),
            SpecificationField(
                name="moveset_types",
                description="Types of movesets to analyze",
                field_type="select",
                required=False,
                options=["standard", "alternative", "niche", "all"],
                default_value="all",
                question="Which types of movesets should I cover?",
            ),
        ]

        # Meta Analysis Schema
        schemas[QueryType.META_ANALYSIS] = [
            SpecificationField(
                name="format",
                description="Competitive format",
                field_type="select",
                required=True,
                options=["OU", "UU", "RU", "VGC", "National Dex"],
                default_value="OU",
                question="Which competitive format's meta should I analyze?",
            ),
            SpecificationField(
                name="analysis_focus",
                description="Focus of meta analysis",
                field_type="select",
                required=True,
                options=[
                    "tier_list",
                    "trends",
                    "threats",
                    "team_archetypes",
                    "comprehensive",
                ],
                default_value="comprehensive",
                question="What aspect of the meta should I focus on?",
            ),
            SpecificationField(
                name="time_period",
                description="Time period for analysis",
                field_type="select",
                required=False,
                options=["current", "recent_changes", "historical"],
                default_value="current",
                question="What time period should I analyze?",
            ),
        ]

        # General Info Schema (minimal requirements)
        schemas[QueryType.GENERAL_INFO] = [
            SpecificationField(
                name="topic",
                description="Information topic",
                field_type="text",
                required=True,
                question="What specific information are you looking for?",
            ),
            SpecificationField(
                name="detail_level",
                description="Level of detail",
                field_type="select",
                required=False,
                options=["basic", "detailed", "comprehensive"],
                default_value="detailed",
                question="How detailed should the information be?",
            ),
        ]

        return schemas

    async def analyze_query_and_generate_spec(
        self, query: str, conversation_context: Dict[str, Any] = None
    ) -> TaskSpecification:
        """
        Analyze user query and generate initial specification.

        Args:
            query: User's natural language query
            conversation_context: Previous conversation context

        Returns:
            TaskSpecification with initial fields filled
        """

        # Step 1: Determine query type
        query_type = await self._classify_query_type(query, conversation_context)

        # Step 2: Extract known information from query
        extracted_fields = await self._extract_fields_from_query(
            query, query_type, conversation_context
        )

        # Step 3: Create specification with extracted fields
        spec = TaskSpecification(
            query_type=query_type, original_query=query, fields=extracted_fields
        )

        self.logger.info(f"Generated specification for query type: {query_type.value}")
        return spec

    async def _classify_query_type(
        self, query: str, context: Dict[str, Any] = None
    ) -> QueryType:
        """Classify the query type using LLM or rule-based approach."""

        if self.llm_client:
            return await self._llm_classify_query_type(query, context)
        else:
            return self._rule_based_classify_query_type(query)

    async def _llm_classify_query_type(
        self, query: str, context: Dict[str, Any] = None
    ) -> QueryType:
        """Use LLM to classify query type."""

        context_text = ""
        if context and context.get("mentioned_pokemon"):
            context_text = f"\nConversation context: Previously discussed {', '.join(context['mentioned_pokemon'])}"

        classification_prompt = f"""
        Classify this Pokemon query into one of these categories:
        
        Query: "{query}"{context_text}
        
        Categories:
        1. competitive_analysis - Analyzing a single Pokemon's competitive viability
        2. team_building - Building a team around specific Pokemon or strategy
        3. pokemon_comparison - Comparing multiple Pokemon
        4. moveset_analysis - Analyzing movesets for specific Pokemon
        5. meta_analysis - Analyzing competitive meta, tiers, or trends
        6. type_analysis - Analyzing Pokemon types, effectiveness, etc.
        7. battle_strategy - Battle tactics and strategies
        8. general_info - General Pokemon information
        
        Respond with just the category name.
        """

        try:
            response = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": classification_prompt}],
                temperature=0.1,
            )

            classification = (
                response["choices"][0]["message"]["content"].strip().lower()
            )

            # Map response to QueryType
            type_mapping = {
                "competitive_analysis": QueryType.COMPETITIVE_ANALYSIS,
                "team_building": QueryType.TEAM_BUILDING,
                "pokemon_comparison": QueryType.POKEMON_COMPARISON,
                "moveset_analysis": QueryType.MOVESET_ANALYSIS,
                "meta_analysis": QueryType.META_ANALYSIS,
                "type_analysis": QueryType.TYPE_ANALYSIS,
                "battle_strategy": QueryType.BATTLE_STRATEGY,
                "general_info": QueryType.GENERAL_INFO,
            }

            return type_mapping.get(classification, QueryType.GENERAL_INFO)

        except Exception as e:
            self.logger.error(f"Error in LLM query classification: {e}")
            return self._rule_based_classify_query_type(query)

    def _rule_based_classify_query_type(self, query: str) -> QueryType:
        """Fallback rule-based query classification."""

        query_lower = query.lower()

        # Team building keywords
        if any(
            word in query_lower for word in ["team", "build", "composition", "synergy"]
        ):
            return QueryType.TEAM_BUILDING

        # Comparison keywords
        elif any(
            word in query_lower
            for word in ["compare", "vs", "versus", "better", "difference"]
        ):
            return QueryType.POKEMON_COMPARISON

        # Moveset keywords
        elif any(
            word in query_lower for word in ["moveset", "moves", "attacks", "abilities"]
        ):
            return QueryType.MOVESET_ANALYSIS

        # Meta analysis keywords
        elif any(
            word in query_lower
            for word in ["meta", "tier", "trends", "ladder", "usage"]
        ):
            return QueryType.META_ANALYSIS

        # Competitive analysis keywords
        elif any(
            word in query_lower
            for word in [
                "competitive",
                "viability",
                "analysis",
                "strengths",
                "weaknesses",
            ]
        ):
            return QueryType.COMPETITIVE_ANALYSIS

        # Battle strategy keywords
        elif any(
            word in query_lower
            for word in ["strategy", "tactics", "counter", "how to beat"]
        ):
            return QueryType.BATTLE_STRATEGY

        # Default to general info
        else:
            return QueryType.GENERAL_INFO

    async def _extract_fields_from_query(
        self, query: str, query_type: QueryType, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Extract known field values from the query."""

        if self.llm_client:
            return await self._llm_extract_fields(query, query_type, context)
        else:
            return self._rule_based_extract_fields(query, query_type, context)

    async def _llm_extract_fields(
        self, query: str, query_type: QueryType, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Use LLM to extract field values from query."""

        schema = self.schemas.get(query_type, [])
        field_descriptions = {field.name: field.description for field in schema}

        context_text = ""
        if context and context.get("mentioned_pokemon"):
            context_text = f"\nConversation context: Previously discussed {', '.join(context['mentioned_pokemon'])}"

        extraction_prompt = f"""
        Extract information from this Pokemon query for the following fields:
        
        Query: "{query}"{context_text}
        Query Type: {query_type.value}
        
        Fields to extract:
        {json.dumps(field_descriptions, indent=2)}
        
        Extract any explicitly mentioned or clearly implied values.
        If a field is not mentioned or unclear, don't include it.
        
        Respond with a JSON object containing only the fields you can extract.
        """

        try:
            response = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": extraction_prompt}],
                temperature=0.1,
            )

            content = response["choices"][0]["message"]["content"].strip()

            # Try to parse JSON response
            if content.startswith("{") and content.endswith("}"):
                return json.loads(content)
            else:
                # Fallback to rule-based extraction
                return self._rule_based_extract_fields(query, query_type, context)

        except Exception as e:
            self.logger.error(f"Error in LLM field extraction: {e}")
            return self._rule_based_extract_fields(query, query_type, context)

    def _rule_based_extract_fields(
        self, query: str, query_type: QueryType, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Fallback rule-based field extraction."""

        extracted = {}
        query_lower = query.lower()

        # Extract Pokemon names (simple approach)
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
        ]

        mentioned_pokemon = [
            pokemon for pokemon in common_pokemon if pokemon in query_lower
        ]

        # Extract based on query type
        if query_type == QueryType.COMPETITIVE_ANALYSIS and mentioned_pokemon:
            extracted["target_pokemon"] = mentioned_pokemon[0]

        elif query_type == QueryType.TEAM_BUILDING and mentioned_pokemon:
            extracted["core_pokemon"] = mentioned_pokemon[0]

        elif query_type == QueryType.POKEMON_COMPARISON and len(mentioned_pokemon) >= 2:
            extracted["pokemon_list"] = mentioned_pokemon[:3]  # Limit to 3

        elif query_type == QueryType.MOVESET_ANALYSIS and mentioned_pokemon:
            extracted["target_pokemon"] = mentioned_pokemon[0]

        # Extract format if mentioned
        formats = ["ou", "uu", "ru", "nu", "pu", "vgc", "national dex", "little cup"]
        for format_name in formats:
            if format_name in query_lower:
                extracted["format"] = format_name.upper().replace(" ", "_")
                break

        return extracted

    def get_missing_required_fields(
        self, spec: TaskSpecification
    ) -> List[SpecificationField]:
        """Get list of required fields that are missing from specification."""

        schema = self.schemas.get(spec.query_type, [])
        missing_fields = []

        for field in schema:
            if field.required and field.name not in spec.fields:
                missing_fields.append(field)

        return missing_fields

    def get_optional_fields_to_confirm(
        self, spec: TaskSpecification
    ) -> List[SpecificationField]:
        """Get list of optional fields that should be confirmed with user."""

        schema = self.schemas.get(spec.query_type, [])
        optional_fields = []

        for field in schema:
            if not field.required and field.name not in spec.fields:
                optional_fields.append(field)

        return optional_fields

    def format_specification_summary(self, spec: TaskSpecification) -> str:
        """Format specification for user confirmation."""

        summary = f"📋 **Research Specification for: {spec.query_type.value.replace('_', ' ').title()}**\n\n"
        summary += f"**Original Query:** {spec.original_query}\n\n"
        summary += "**Specification:**\n"

        schema = self.schemas.get(spec.query_type, [])
        field_map = {field.name: field for field in schema}

        for field_name, value in spec.fields.items():
            field = field_map.get(field_name)
            if field:
                display_name = field.description
                if isinstance(value, list):
                    value_str = ", ".join(str(v) for v in value)
                else:
                    value_str = str(value)
                summary += f"  • {display_name}: {value_str}\n"

        return summary

    def apply_defaults_to_spec(self, spec: TaskSpecification) -> None:
        """Apply default values to missing optional fields."""

        schema = self.schemas.get(spec.query_type, [])

        for field in schema:
            if field.name not in spec.fields and field.default_value is not None:
                spec.fields[field.name] = field.default_value
