"""
OpenAI LLM client wrapper for the Pokémon Deep Research Agent system.

This module provides a robust interface to OpenAI's API with features like:
- Token usage tracking and optimization
- Response caching for repeated queries
- Error handling and retry logic
- Structured output parsing
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import openai
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.models import ResearchQuery, ResearchResult


class LLMClient:
    """
    OpenAI API client with advanced features for research agents.

    Features:
    - Token usage tracking and optimization
    - Response caching for efficiency
    - Structured output parsing
    - Error handling and retry logic
    - Multiple model support
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4.1-mini",
        max_tokens: int = 4000,
        temperature: float = 0.7,
        cache_responses: bool = True,
        base_url: Optional[str] = None,
    ):
        # Use environment base URL if available
        import os

        base_url = (
            base_url or os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")
        )

        if base_url:
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = AsyncOpenAI(api_key=api_key)

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cache_responses = cache_responses

        # Response cache
        self.response_cache: Dict[str, Dict[str, Any]] = {}

        # Usage tracking
        self.total_tokens_used = 0
        self.total_requests = 0
        self.total_cost = 0.0

        # Model pricing (tokens per dollar, approximate)
        self.model_pricing = {
            "gpt-4.1-mini": {"input": 0.15, "output": 0.60},  # per 1M tokens
            "gpt-4.1-nano": {"input": 0.075, "output": 0.30},  # per 1M tokens
            "gemini-2.5-flash": {"input": 0.075, "output": 0.30},  # per 1M tokens
            "gpt-4": {"input": 0.03, "output": 0.06},  # per 1K tokens (legacy)
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},  # per 1K tokens (legacy)
            "gpt-3.5-turbo": {
                "input": 0.0015,
                "output": 0.002,
            },  # per 1K tokens (legacy)
        }

        self.logger = logging.getLogger(__name__)

    def _get_cache_key(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate cache key for a request."""
        # Create a hash of the messages and parameters
        cache_data = {
            "messages": messages,
            "model": kwargs.get("model", self.model),
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        return str(hash(json.dumps(cache_data, sort_keys=True)))

    def _calculate_cost(self, usage: Dict[str, int]) -> float:
        """Calculate approximate cost of API usage."""
        if self.model not in self.model_pricing:
            return 0.0

        pricing = self.model_pricing[self.model]
        input_cost = (usage.get("prompt_tokens", 0) / 1000) * pricing["input"]
        output_cost = (usage.get("completion_tokens", 0) / 1000) * pricing["output"]

        return input_cost + output_cost

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Make a chat completion request with caching and error handling.
        """
        # Use defaults if not specified
        model = model or self.model
        temperature = temperature or self.temperature
        max_tokens = max_tokens or self.max_tokens

        # Check cache first
        if self.cache_responses:
            cache_key = self._get_cache_key(
                messages, model=model, temperature=temperature, max_tokens=max_tokens
            )
            if cache_key in self.response_cache:
                self.logger.debug("Using cached response")
                return self.response_cache[cache_key]

        # Prepare request parameters
        request_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format:
            request_params["response_format"] = response_format

        if tools:
            request_params["tools"] = tools
            if tool_choice:
                request_params["tool_choice"] = tool_choice

        try:
            # Make the API request
            response = await self.client.chat.completions.create(**request_params)

            # Update usage statistics
            if hasattr(response, "usage") and response.usage:
                usage = response.usage.model_dump()
                self.total_tokens_used += usage.get("total_tokens", 0)
                self.total_cost += self._calculate_cost(usage)

            self.total_requests += 1

            # Convert to dict for easier handling
            response_dict = response.model_dump()

            # Cache the response
            if self.cache_responses and cache_key:
                self.response_cache[cache_key] = response_dict

            self.logger.debug(
                f"API request completed. Tokens used: {response.usage.total_tokens if response.usage else 'unknown'}"
            )

            return response_dict

        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise

    async def analyze_pokemon_query(self, query: str) -> Dict[str, Any]:
        """Analyze a Pokemon research query to determine intent and parameters."""

        system_prompt = """You are an expert Pokemon analyst. Analyze user queries about Pokemon and extract key information.

For the query, identify:
1. Query type (team_building, individual_analysis, comparison, battle_strategy, location_info, etc.)
2. Specific Pokemon mentioned (if any)
3. Game/generation context (if specified)
4. Battle format (singles, doubles, etc.) if relevant
5. Difficulty level or target audience (beginner, competitive, etc.)

Respond in a structured format but use natural language, not JSON."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze this Pokemon query: {query}"},
        ]

        response = await self.chat_completion(
            messages=messages,
            temperature=0.3,  # Lower temperature for more consistent analysis
        )

        # Parse the natural language response into structured data
        analysis_text = response["choices"][0]["message"]["content"].lower()

        # Extract query type
        query_type = "general"
        if "team" in analysis_text or "build" in analysis_text:
            query_type = "team_building"
        elif "compare" in analysis_text or "vs" in analysis_text:
            query_type = "comparison"
        elif "battle" in analysis_text or "strategy" in analysis_text:
            query_type = "battle_strategy"
        elif "individual" in analysis_text or "about" in query.lower():
            query_type = "individual_analysis"

        # Extract Pokemon mentioned (simple keyword matching)
        pokemon_mentioned = []
        common_pokemon = [
            "pikachu",
            "charizard",
            "blastoise",
            "venusaur",
            "garchomp",
            "dragonite",
            "alakazam",
            "gengar",
            "lucario",
            "mewtwo",
        ]
        for pokemon in common_pokemon:
            if pokemon in query.lower():
                pokemon_mentioned.append(pokemon)

        return {
            "query_type": query_type,
            "pokemon_mentioned": pokemon_mentioned,
            "game_context": "general",
            "battle_format": (
                "singles"
                if "singles" in analysis_text
                else "doubles" if "doubles" in analysis_text else "unknown"
            ),
            "difficulty_level": (
                "competitive"
                if "competitive" in analysis_text
                else "beginner" if "beginner" in analysis_text else "intermediate"
            ),
            "constraints": [],
            "analysis_text": response["choices"][0]["message"]["content"],
        }

    async def generate_clarification_questions(
        self, query: str, analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate clarification questions to better understand user intent."""

        system_prompt = """You are a helpful Pokemon research assistant. Based on the user's query and initial analysis, generate 2-3 clarifying questions that would help provide better, more targeted research results.

Focus on:
- Specific game versions or generations if not mentioned
- Battle format preferences (singles/doubles) for competitive queries
- Skill level or experience (beginner/intermediate/advanced)
- Specific constraints or preferences
- Any ambiguous terms that could be interpreted multiple ways

Keep questions concise and relevant. List each question on a separate line starting with "Q:"."""

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Original query: {query}\n\nAnalysis: {json.dumps(analysis, indent=2)}\n\nGenerate clarification questions:",
            },
        ]

        response = await self.chat_completion(messages=messages, temperature=0.5)

        # Parse questions from the response
        response_text = response["choices"][0]["message"]["content"]
        questions = []
        for line in response_text.split("\n"):
            line = line.strip()
            if line.startswith("Q:"):
                questions.append(line[2:].strip())
            elif line.startswith("-") or line.startswith("•"):
                questions.append(line[1:].strip())

        return questions[:3]  # Return max 3 questions

    async def synthesize_research_findings(
        self, query: ResearchQuery, findings: List[ResearchResult]
    ) -> Dict[str, Any]:
        """Synthesize multiple research findings into a comprehensive analysis."""

        system_prompt = """You are an expert Pokemon researcher tasked with synthesizing research findings into a comprehensive, well-structured analysis.

Your analysis should:
1. Directly answer the user's original question
2. Provide detailed explanations and reasoning
3. Include specific data and statistics where relevant
4. Offer practical recommendations
5. Highlight any important caveats or considerations
6. Structure the response in a clear, logical manner

Use the research findings provided to create an authoritative, helpful response."""

        # Prepare findings summary
        findings_text = ""
        for i, finding in enumerate(findings, 1):
            findings_text += f"\n--- Finding {i} (from {finding.agent_id}) ---\n"
            findings_text += f"Type: {finding.result_type}\n"
            findings_text += f"Data: {json.dumps(finding.data, indent=2)}\n"
            findings_text += f"Confidence: {finding.confidence}\n"
            findings_text += f"Sources: {', '.join(finding.sources)}\n"

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Original Query: {query.clarified_query or query.original_query}\n\nQuery Type: {query.query_type}\n\nResearch Findings:{findings_text}\n\nPlease synthesize these findings into a comprehensive analysis:",
            },
        ]

        response = await self.chat_completion(
            messages=messages,
            temperature=0.6,
            max_tokens=6000,  # Allow for longer synthesis
        )

        return {
            "synthesis": response["choices"][0]["message"]["content"],
            "confidence": min([f.confidence for f in findings]) if findings else 0.0,
            "sources_used": list(
                set([source for f in findings for source in f.sources])
            ),
            "findings_count": len(findings),
        }

    async def generate_team_recommendation(
        self, pokemon_data: List[Dict[str, Any]], analysis_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate team building recommendations based on Pokemon data."""

        system_prompt = """You are a competitive Pokemon team building expert. Based on the provided Pokemon data and context, create detailed team recommendations.

Your recommendations should include:
1. Specific Pokemon selections with justifications
2. Role assignments (sweeper, tank, support, etc.)
3. Movesets and item suggestions
4. Team synergies and strategies
5. Potential weaknesses and how to address them
6. Alternative options for flexibility

Be specific and provide actionable advice."""

        context_text = json.dumps(analysis_context, indent=2)
        pokemon_text = json.dumps(pokemon_data, indent=2)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context: {context_text}\n\nPokemon Data: {pokemon_text}\n\nGenerate team recommendations:",
            },
        ]

        response = await self.chat_completion(
            messages=messages, temperature=0.7, max_tokens=5000
        )

        return {
            "recommendations": response["choices"][0]["message"]["content"],
            "context_used": analysis_context,
            "pokemon_analyzed": len(pokemon_data),
        }

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get API usage statistics."""
        return {
            "total_requests": self.total_requests,
            "total_tokens_used": self.total_tokens_used,
            "estimated_cost": round(self.total_cost, 4),
            "cache_size": len(self.response_cache),
            "average_tokens_per_request": (
                self.total_tokens_used / self.total_requests
                if self.total_requests > 0
                else 0
            ),
        }

    def clear_cache(self) -> None:
        """Clear the response cache."""
        cache_size = len(self.response_cache)
        self.response_cache.clear()
        self.logger.info(f"Cleared {cache_size} cached responses")
