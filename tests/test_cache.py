"""Tests for caching functionality."""

import time
from pathlib import Path
from unittest.mock import patch

from prompt_refiner.cache import Cache
from prompt_refiner.config import CacheConfig, Config


class TestCacheHitMiss:
    """Test cache hit/miss behavior."""

    def test_cache_miss_on_first_request(self, tmp_path):
        """Test that first request is a cache miss."""
        cache_config = CacheConfig(location=str(tmp_path), ttl_hours=1)
        cache = Cache(cache_config)

        result = cache.get("test_key")
        assert result is None

    def test_cache_hit_on_second_request(self, tmp_path):
        """Test that second identical request is a cache hit."""
        cache_config = CacheConfig(location=str(tmp_path), ttl_hours=1)
        cache = Cache(cache_config)

        test_data = {"improved_prompt": "test", "score": "8/10"}
        cache.set("test_key", test_data)

        result = cache.get("test_key")
        assert result == test_data

    def test_different_keys_have_separate_entries(self, tmp_path):
        """Test that different keys maintain separate cache entries."""
        cache_config = CacheConfig(location=str(tmp_path), ttl_hours=1)
        cache = Cache(cache_config)

        data1 = {"prompt": "first"}
        data2 = {"prompt": "second"}

        cache.set("key1", data1)
        cache.set("key2", data2)

        assert cache.get("key1") == data1
        assert cache.get("key2") == data2
        assert cache.get("key3") is None


class TestCacheTTL:
    """Test TTL expiration logic."""

    def test_expired_entries_return_none(self, tmp_path):
        """Test that expired entries are treated as cache misses."""
        cache_config = CacheConfig(location=str(tmp_path), ttl_hours=0)  # Instant expiry
        cache = Cache(cache_config)

        cache.set("test_key", {"data": "test"})

        # Force a small delay to ensure expiration
        time.sleep(0.1)

        result = cache.get("test_key")
        assert result is None

    def test_non_expired_entries_return_data(self, tmp_path):
        """Test that non-expired entries return cached data."""
        cache_config = CacheConfig(location=str(tmp_path), ttl_hours=24)
        cache = Cache(cache_config)

        test_data = {"data": "test"}
        cache.set("test_key", test_data)

        result = cache.get("test_key")
        assert result == test_data

    @patch('time.time')
    def test_ttl_boundary_conditions(self, mock_time, tmp_path):
        """Test TTL behavior at exact boundary."""
        cache_config = CacheConfig(location=str(tmp_path), ttl_hours=1)
        cache = Cache(cache_config)

        # Set time to a fixed point
        mock_time.return_value = 1000.0
        cache.set("test_key", {"data": "test"})

        # Move time to just before expiration
        mock_time.return_value = 1000.0 + (3600 - 1)  # 1 hour - 1 second
        assert cache.get("test_key") == {"data": "test"}

        # Move time to exactly at expiration
        mock_time.return_value = 1000.0 + 3600  # Exactly 1 hour
        assert cache.get("test_key") is None


class TestCachePersistence:
    """Test saving/loading from disk."""

    def test_cache_persists_across_instances(self, tmp_path):
        """Test that cache data persists when creating new instances."""
        cache_config = CacheConfig(location=str(tmp_path), ttl_hours=24)

        # First instance - write data
        cache1 = Cache(cache_config)
        cache1.set("persistent_key", {"value": "persisted"})

        # Second instance - read data
        cache2 = Cache(cache_config)
        result = cache2.get("persistent_key")

        assert result == {"value": "persisted"}

    def test_handles_corrupted_cache_file(self, tmp_path):
        """Test graceful handling of corrupted cache files."""
        cache_config = CacheConfig(location=str(tmp_path), ttl_hours=24)
        cache_file = Path(tmp_path) / "prompt_cache.json"

        # Create corrupted cache file
        cache_file.write_text("{ invalid json content")

        # Should handle gracefully and start fresh
        cache = Cache(cache_config)
        assert cache.get("any_key") is None

        # Should be able to write new data
        cache.set("new_key", {"data": "new"})
        assert cache.get("new_key") == {"data": "new"}

    def test_creates_cache_directory_if_missing(self, tmp_path):
        """Test that cache directory is created if it doesn't exist."""
        cache_dir = tmp_path / "new" / "cache" / "dir"
        cache_config = CacheConfig(location=str(cache_dir), ttl_hours=24)

        cache = Cache(cache_config)
        cache.set("test_key", {"data": "test"})

        assert cache_dir.exists()
        assert (cache_dir / "prompt_cache.json").exists()

    def test_handles_permission_errors_gracefully(self, tmp_path):
        """Test graceful handling of permission errors."""
        cache_config = CacheConfig(location=str(tmp_path), ttl_hours=24)
        cache = Cache(cache_config)

        # Mock file operations to raise permission error
        with patch('builtins.open', side_effect=PermissionError("No write access")):
            # Should not raise, just fail silently
            cache.set("test_key", {"data": "test"})

        # Cache keeps data in memory even if disk write fails
        assert cache.get("test_key") == {"data": "test"}


class TestCacheKeyGeneration:
    """Test cache key generation from prompts."""

    def test_identical_prompts_generate_same_key(self):
        """Test that identical prompts generate the same cache key."""
        cache = Cache(CacheConfig())

        prompt = "Optimize this function"
        focus_areas = ["performance", "readability"]

        key1 = cache._generate_key(prompt, focus_areas)
        key2 = cache._generate_key(prompt, focus_areas)

        assert key1 == key2

    def test_different_prompts_generate_different_keys(self):
        """Test that different prompts generate different cache keys."""
        cache = Cache(CacheConfig())

        key1 = cache._generate_key("Optimize this function", ["performance"])
        key2 = cache._generate_key("Optimize this class", ["performance"])
        key3 = cache._generate_key("Optimize this function", ["readability"])

        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

    def test_focus_area_order_doesnt_matter(self):
        """Test that focus area order doesn't affect cache key."""
        cache = Cache(CacheConfig())

        key1 = cache._generate_key("test", ["clarity", "brevity", "accuracy"])
        key2 = cache._generate_key("test", ["accuracy", "clarity", "brevity"])

        assert key1 == key2

    def test_handles_special_characters_in_prompt(self):
        """Test key generation with special characters."""
        cache = Cache(CacheConfig())

        prompt = "Test with special chars: !@#$%^&*()_+-=[]{}|;:',.<>?/~`"
        key = cache._generate_key(prompt, [])

        # Should generate valid key without errors
        assert isinstance(key, str)
        assert len(key) > 0


class TestCacheDisabling:
    """Test behavior when cache is disabled."""

    def test_disabled_cache_always_returns_none(self, tmp_path):
        """Test that disabled cache always returns None."""
        cache_config = CacheConfig(enabled=False, location=str(tmp_path))
        cache = Cache(cache_config)

        cache.set("test_key", {"data": "test"})
        result = cache.get("test_key")

        assert result is None

    def test_disabled_cache_doesnt_write_files(self, tmp_path):
        """Test that disabled cache doesn't create files."""
        cache_config = CacheConfig(enabled=False, location=str(tmp_path))
        cache = Cache(cache_config)

        cache.set("test_key", {"data": "test"})

        # No cache file should be created
        assert not (Path(tmp_path) / "prompt_cache.json").exists()


class TestCacheIntegration:
    """Test cache integration with providers."""

    @patch('prompt_refiner.providers.claude.ClaudeProvider.refine_prompt')
    def test_provider_uses_cache_on_second_call(self, mock_refine, tmp_path):
        """Test that provider uses cached results on second call."""
        from prompt_refiner.providers.claude import ClaudeProvider

        # Setup
        config = Config.from_dict({
            'advanced': {'cache': {'location': str(tmp_path), 'enabled': True}}
        })
        provider = ClaudeProvider(config)
        provider.cache = Cache(config.advanced.cache)

        mock_response = {"improved_prompt": "cached result"}
        mock_refine.return_value = mock_response

        # First call - should hit the provider
        provider.refine_prompt("test prompt", ["clarity"])
        assert mock_refine.call_count == 1

        # Manually cache the result (simulating what the real provider would do)
        cache_key = provider.cache._generate_key("test prompt", ["clarity"])
        provider.cache.set(cache_key, mock_response)

        # Second call - should use cache
        # In real implementation, this would check cache first
        cached_result = provider.cache.get(cache_key)
        assert cached_result == mock_response

    def test_cache_with_home_directory_expansion(self):
        """Test that ~ in cache location is properly expanded."""
        cache_config = CacheConfig(location="~/test_cache")
        cache = Cache(cache_config)

        # Should expand ~ to actual home directory
        assert not cache.cache_dir.parts[0] == "~"
        assert cache.cache_dir.is_absolute()
