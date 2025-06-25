#!/usr/bin/env python3
"""Cache management for prompt refinement results."""

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional


class Cache:
    """Manages caching of prompt refinement results."""

    def __init__(self, cache_dir: Path, enabled: bool, ttl_hours: int):
        """Initialize the cache.

        Args:
            cache_dir: Directory to store cache files
            enabled: Whether caching is enabled
            ttl_hours: Time-to-live for cache entries in hours
        """
        self.cache_dir = cache_dir
        self.enabled = enabled
        self.ttl_hours = ttl_hours

        # Create cache directory if caching is enabled
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_key(self, prompt: str, template: str, provider: str) -> str:
        """Generate a unique cache key for the given parameters.

        Args:
            prompt: The original prompt
            template: The template being used
            provider: The LLM provider (claude/ollama)

        Returns:
            SHA256 hash of the combined parameters
        """
        content = f"{prompt}:{template}:{provider}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, prompt: str, template: str, provider: str) -> Optional[Dict[str, str]]:
        """Retrieve a cached result if available and not expired.

        Args:
            prompt: The original prompt
            template: The template being used
            provider: The LLM provider

        Returns:
            Cached result with 'from_cache': True added, or None if not found/expired
        """
        if not self.enabled:
            return None

        cache_key = self.get_cache_key(prompt, template, provider)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file) as f:
                cached = json.load(f)

            # Check TTL
            cached_time = datetime.fromisoformat(cached['timestamp'])
            if datetime.now() - cached_time > timedelta(hours=self.ttl_hours):
                cache_file.unlink()  # Delete expired cache
                return None

            # Add from_cache indicator
            result = cached['result']
            result['from_cache'] = True
            return result

        except (json.JSONDecodeError, KeyError, ValueError):
            # If cache file is corrupted, remove it
            cache_file.unlink()
            return None

    def save(self, prompt: str, template: str, provider: str, result: Dict[str, str]):
        """Save a result to cache.

        Args:
            prompt: The original prompt
            template: The template being used
            provider: The LLM provider
            result: The refinement result to cache
        """
        if not self.enabled:
            return

        cache_key = self.get_cache_key(prompt, template, provider)
        cache_file = self.cache_dir / f"{cache_key}.json"

        with open(cache_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'prompt': prompt,
                'template': template,
                'provider': provider,
                'result': result
            }, f, indent=2)

    def clear(self) -> int:
        """Clear all cache files.

        Returns:
            Number of cache files deleted
        """
        if not self.cache_dir.exists():
            return 0

        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1

        return count
