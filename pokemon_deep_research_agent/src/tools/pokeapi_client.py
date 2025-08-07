"""
PokéAPI client with intelligent caching and comprehensive error handling.

This module provides a robust interface to the PokéAPI with features like:
- Multi-level caching (memory and file-based)
- Rate limiting and retry logic
- Data validation and normalization
- Comprehensive endpoint coverage
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.models import MoveData, PokemonData, TypeData


class PokéAPICache:
    """Multi-level caching system for PokéAPI responses."""

    def __init__(self, cache_dir: str = "data/cache", max_memory_items: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Memory cache
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        self.max_memory_items = max_memory_items

        # Cache settings
        self.default_ttl = timedelta(hours=24)  # 24 hour default TTL
        self.pokemon_ttl = timedelta(days=7)  # Pokemon data rarely changes
        self.move_ttl = timedelta(days=7)  # Move data rarely changes
        self.type_ttl = timedelta(days=30)  # Type data almost never changes

        self.logger = logging.getLogger(__name__)

    def _get_cache_file(self, key: str) -> Path:
        """Get the cache file path for a given key."""
        # Create a safe filename from the key
        safe_key = key.replace("/", "_").replace("?", "_").replace("&", "_")
        return self.cache_dir / f"{safe_key}.json"

    def _get_ttl_for_endpoint(self, endpoint: str) -> timedelta:
        """Get appropriate TTL based on endpoint type."""
        if "/pokemon/" in endpoint:
            return self.pokemon_ttl
        elif "/move/" in endpoint:
            return self.move_ttl
        elif "/type/" in endpoint:
            return self.type_ttl
        else:
            return self.default_ttl

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get data from cache."""
        # Check memory cache first
        if key in self.memory_cache:
            timestamp = self.cache_timestamps.get(key)
            if timestamp and datetime.now() - timestamp < self._get_ttl_for_endpoint(
                key
            ):
                self.logger.debug(f"Cache hit (memory): {key}")
                return self.memory_cache[key]
            else:
                # Expired, remove from memory cache
                del self.memory_cache[key]
                if key in self.cache_timestamps:
                    del self.cache_timestamps[key]

        # Check file cache
        cache_file = self._get_cache_file(key)
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)

                # Check if file cache is still valid
                cached_time = datetime.fromisoformat(
                    cache_data.get("cached_at", "1970-01-01")
                )
                if datetime.now() - cached_time < self._get_ttl_for_endpoint(key):
                    data = cache_data.get("data")

                    # Store in memory cache if there's room
                    if len(self.memory_cache) < self.max_memory_items:
                        self.memory_cache[key] = data
                        self.cache_timestamps[key] = cached_time

                    self.logger.debug(f"Cache hit (file): {key}")
                    return data
                else:
                    # Expired file cache, remove it
                    cache_file.unlink()

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                self.logger.warning(f"Invalid cache file {cache_file}: {e}")
                cache_file.unlink()

        return None

    async def set(self, key: str, data: Dict[str, Any]) -> None:
        """Store data in cache."""
        now = datetime.now()

        # Store in memory cache
        if len(self.memory_cache) >= self.max_memory_items:
            # Remove oldest item
            oldest_key = min(
                self.cache_timestamps.keys(), key=lambda k: self.cache_timestamps[k]
            )
            del self.memory_cache[oldest_key]
            del self.cache_timestamps[oldest_key]

        self.memory_cache[key] = data
        self.cache_timestamps[key] = now

        # Store in file cache
        cache_file = self._get_cache_file(key)
        cache_data = {"cached_at": now.isoformat(), "data": data}

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            self.logger.debug(f"Cached: {key}")
        except Exception as e:
            self.logger.error(f"Failed to write cache file {cache_file}: {e}")

    def clear_expired(self) -> int:
        """Clear expired cache entries and return count of removed items."""
        removed_count = 0
        now = datetime.now()

        # Clear expired memory cache
        expired_keys = []
        for key, timestamp in self.cache_timestamps.items():
            if now - timestamp >= self._get_ttl_for_endpoint(key):
                expired_keys.append(key)

        for key in expired_keys:
            del self.memory_cache[key]
            del self.cache_timestamps[key]
            removed_count += 1

        # Clear expired file cache
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)

                cached_time = datetime.fromisoformat(
                    cache_data.get("cached_at", "1970-01-01")
                )
                key = cache_file.stem.replace(
                    "_", "/"
                )  # Approximate reverse of safe_key

                if now - cached_time >= self._get_ttl_for_endpoint(key):
                    cache_file.unlink()
                    removed_count += 1

            except Exception as e:
                self.logger.warning(f"Error checking cache file {cache_file}: {e}")

        if removed_count > 0:
            self.logger.info(f"Cleared {removed_count} expired cache entries")

        return removed_count


class PokéAPIClient:
    """
    Comprehensive client for the PokéAPI with advanced features.

    Features:
    - Intelligent caching with TTL
    - Rate limiting and retry logic
    - Data validation and normalization
    - Comprehensive endpoint coverage
    - Batch operations for efficiency
    """

    def __init__(
        self,
        base_url: str = "https://pokeapi.co/api/v2/",
        cache_dir: str = "data/cache",
        rate_limit_delay: float = 0.1,
        max_concurrent_requests: int = 10,
    ):
        self.base_url = base_url
        self.cache = PokéAPICache(cache_dir)
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0.0

        # Request session and rate limiting
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

        # Statistics
        self.requests_made = 0
        self.cache_hits = 0
        self.cache_misses = 0

        self.logger = logging.getLogger(__name__)

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def start(self) -> None:
        """Initialize the HTTP session."""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={"User-Agent": "Pokemon-Deep-Research-Agent/1.0"},
            )

    async def close(self) -> None:
        """Close the HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _make_request(self, endpoint: str) -> Dict[str, Any]:
        """Make a rate-limited HTTP request with retry logic."""
        if not self.session:
            await self.start()

        async with self.semaphore:
            # Rate limiting
            now = time.time()
            time_since_last = now - self.last_request_time
            if time_since_last < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - time_since_last)

            url = urljoin(self.base_url, endpoint)
            self.logger.debug(f"Making request to: {url}")

            async with self.session.get(url) as response:
                self.last_request_time = time.time()
                self.requests_made += 1

                if response.status == 200:
                    data = await response.json()
                    return data
                elif response.status == 404:
                    raise ValueError(f"Resource not found: {endpoint}")
                else:
                    response.raise_for_status()

    async def _get_with_cache(self, endpoint: str) -> Dict[str, Any]:
        """Get data with caching."""
        # Check cache first
        cached_data = await self.cache.get(endpoint)
        if cached_data:
            self.cache_hits += 1
            return cached_data

        # Cache miss, make request
        self.cache_misses += 1
        data = await self._make_request(endpoint)

        # Store in cache
        await self.cache.set(endpoint, data)

        return data

    # Pokemon endpoints
    async def get_pokemon(self, identifier: Union[int, str]) -> PokemonData:
        """Get detailed Pokemon data."""
        endpoint = f"pokemon/{identifier}"
        data = await self._get_with_cache(endpoint)
        return PokemonData(**data)

    async def get_pokemon_species(self, identifier: Union[int, str]) -> Dict[str, Any]:
        """Get Pokemon species data."""
        endpoint = f"pokemon-species/{identifier}"
        return await self._get_with_cache(endpoint)

    async def get_pokemon_list(
        self, limit: int = 1000, offset: int = 0
    ) -> Dict[str, Any]:
        """Get list of all Pokemon."""
        endpoint = f"pokemon?limit={limit}&offset={offset}"
        return await self._get_with_cache(endpoint)

    async def search_pokemon_by_type(self, type_name: str) -> List[Dict[str, Any]]:
        """Get all Pokemon of a specific type."""
        type_data = await self.get_type(type_name)
        return type_data.pokemon

    # Move endpoints
    async def get_move(self, identifier: Union[int, str]) -> MoveData:
        """Get detailed move data."""
        endpoint = f"move/{identifier}"
        data = await self._get_with_cache(endpoint)
        return MoveData(**data)

    async def get_move_list(self, limit: int = 1000, offset: int = 0) -> Dict[str, Any]:
        """Get list of all moves."""
        endpoint = f"move?limit={limit}&offset={offset}"
        return await self._get_with_cache(endpoint)

    # Type endpoints
    async def get_type(self, identifier: Union[int, str]) -> TypeData:
        """Get detailed type data including effectiveness."""
        endpoint = f"type/{identifier}"
        data = await self._get_with_cache(endpoint)
        return TypeData(**data)

    async def get_type_list(self) -> Dict[str, Any]:
        """Get list of all types."""
        endpoint = "type"
        return await self._get_with_cache(endpoint)

    # Ability endpoints
    async def get_ability(self, identifier: Union[int, str]) -> Dict[str, Any]:
        """Get detailed ability data."""
        endpoint = f"ability/{identifier}"
        return await self._get_with_cache(endpoint)

    async def get_ability_list(
        self, limit: int = 1000, offset: int = 0
    ) -> Dict[str, Any]:
        """Get list of all abilities."""
        endpoint = f"ability?limit={limit}&offset={offset}"
        return await self._get_with_cache(endpoint)

    # Item endpoints
    async def get_item(self, identifier: Union[int, str]) -> Dict[str, Any]:
        """Get detailed item data."""
        endpoint = f"item/{identifier}"
        return await self._get_with_cache(endpoint)

    async def get_item_list(self, limit: int = 1000, offset: int = 0) -> Dict[str, Any]:
        """Get list of all items."""
        endpoint = f"item?limit={limit}&offset={offset}"
        return await self._get_with_cache(endpoint)

    # Location endpoints
    async def get_location(self, identifier: Union[int, str]) -> Dict[str, Any]:
        """Get detailed location data."""
        endpoint = f"location/{identifier}"
        return await self._get_with_cache(endpoint)

    async def get_location_area(self, identifier: Union[int, str]) -> Dict[str, Any]:
        """Get location area data with Pokemon encounters."""
        endpoint = f"location-area/{identifier}"
        return await self._get_with_cache(endpoint)

    # Generation endpoints
    async def get_generation(self, identifier: Union[int, str]) -> Dict[str, Any]:
        """Get generation data."""
        endpoint = f"generation/{identifier}"
        return await self._get_with_cache(endpoint)

    async def get_version_group(self, identifier: Union[int, str]) -> Dict[str, Any]:
        """Get version group data."""
        endpoint = f"version-group/{identifier}"
        return await self._get_with_cache(endpoint)

    # Batch operations
    async def get_pokemon_batch(
        self, identifiers: List[Union[int, str]]
    ) -> List[PokemonData]:
        """Get multiple Pokemon data efficiently."""
        tasks = [self.get_pokemon(identifier) for identifier in identifiers]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def get_moves_batch(
        self, identifiers: List[Union[int, str]]
    ) -> List[MoveData]:
        """Get multiple move data efficiently."""
        tasks = [self.get_move(identifier) for identifier in identifiers]
        return await asyncio.gather(*tasks, return_exceptions=True)

    # Utility methods
    async def search_pokemon_by_name_pattern(
        self, pattern: str
    ) -> List[Dict[str, Any]]:
        """Search Pokemon by name pattern."""
        pokemon_list = await self.get_pokemon_list(limit=2000)
        matching_pokemon = []

        for pokemon in pokemon_list.get("results", []):
            if pattern.lower() in pokemon["name"].lower():
                matching_pokemon.append(pokemon)

        return matching_pokemon

    async def get_pokemon_by_generation(self, generation: int) -> List[Dict[str, Any]]:
        """Get all Pokemon from a specific generation."""
        gen_data = await self.get_generation(generation)
        return gen_data.get("pokemon_species", [])

    async def get_type_effectiveness(
        self, attacking_type: str, defending_type: str
    ) -> float:
        """Get type effectiveness multiplier."""
        type_data = await self.get_type(attacking_type)
        damage_relations = type_data.damage_relations

        # Check for super effective
        for type_info in damage_relations.get("double_damage_to", []):
            if type_info["name"] == defending_type:
                return 2.0

        # Check for not very effective
        for type_info in damage_relations.get("half_damage_to", []):
            if type_info["name"] == defending_type:
                return 0.5

        # Check for no effect
        for type_info in damage_relations.get("no_damage_to", []):
            if type_info["name"] == defending_type:
                return 0.0

        # Normal effectiveness
        return 1.0

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "requests_made": self.requests_made,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(hit_rate, 2),
            "memory_cache_size": len(self.cache.memory_cache),
        }

    async def clear_expired_cache(self) -> int:
        """Clear expired cache entries."""
        return self.cache.clear_expired()
