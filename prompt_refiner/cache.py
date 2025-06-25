#!/usr/bin/env python3
"""Cache management for prompt refinement results."""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import CacheConfig


class Cache:
    """Manages caching of prompt refinement results."""

    def __init__(self, cache_config: CacheConfig):
        """Initialize the cache.

        Args:
            cache_config: Cache configuration object
        """
        self.config = cache_config
        self.enabled = cache_config.enabled
        self.ttl_hours = cache_config.ttl_hours
        self.cache_dir = Path(cache_config.location).expanduser()
        self._cache = {}

        # Load cache from disk if enabled
        if self.enabled and self.cache_dir.exists():
            self._load_cache()

    def _get_cache_file(self) -> Path:
        """Get the path to the cache file."""
        return self.cache_dir / "prompt_cache.json"

    def _load_cache(self):
        """Load cache from disk."""
        cache_file = self._get_cache_file()
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    self._cache = json.load(f)
            except (OSError, json.JSONDecodeError):
                # If cache is corrupted, start fresh
                self._cache = {}

    def _save_cache(self):
        """Save cache to disk."""
        if not self.enabled:
            return

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self._get_cache_file()
            with open(cache_file, 'w') as f:
                json.dump(self._cache, f)
        except (OSError, PermissionError):
            # Fail silently on write errors
            pass

    def _generate_key(self, prompt: str, focus_areas: List[str]) -> str:
        """Generate a unique cache key."""
        # Sort focus areas to ensure consistent keys
        sorted_areas = sorted(focus_areas) if focus_areas else []
        content = f"{prompt}:{','.join(sorted_areas)}"
        return hashlib.sha256(content.encode()).hexdigest()


    def set(self, key: str, data: Dict[str, Any]):
        """Set a value in cache."""
        if not self.enabled:
            return

        self._cache[key] = {
            'data': data,
            'timestamp': time.time()
        }
        self._save_cache()

    # Legacy method signatures for compatibility
    def get_cache_key(self, prompt: str, template: str, provider: str) -> str:
        """Generate a unique cache key for the given parameters."""
        content = f"{prompt}:{template}:{provider}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, *args) -> Optional[Dict[str, Any]]:
        """Get a value from cache. Supports both new and legacy signatures."""
        if len(args) == 1:
            # New signature: get(key)
            return self._get_by_key(args[0])
        elif len(args) == 3:
            # Legacy signature: get(prompt, template, provider)
            if not self.enabled:
                return None
            cache_key = self.get_cache_key(*args)
            return self._get_by_key(cache_key)
        else:
            raise ValueError("Invalid number of arguments")

    def _get_by_key(self, key: str) -> Optional[Dict[str, Any]]:
        """Internal method to get by key."""
        if not self.enabled:
            return None

        if key not in self._cache:
            return None

        entry = self._cache[key]

        # Check TTL
        if self.ttl_hours >= 0:  # Include 0 for instant expiry
            age_hours = (time.time() - entry['timestamp']) / 3600
            if age_hours >= self.ttl_hours:
                # Expired
                del self._cache[key]
                self._save_cache()
                return None

        return entry['data']

    def save(self, prompt: str, template: str, provider: str, result: Dict[str, str]):
        """Store a result in cache using legacy parameters."""
        if not self.enabled:
            return

        cache_key = self.get_cache_key(prompt, template, provider)
        self.set(cache_key, result)

    def clear(self) -> int:
        """Clear all cache entries and return count of cleared entries."""
        count = len(self._cache)
        self._cache = {}
        if self.enabled:
            self._save_cache()
        return count
