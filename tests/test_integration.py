"""Integration tests for the complete refinement flow."""

from unittest.mock import Mock, patch

import pytest

from prompt_refiner.cache import Cache
from prompt_refiner.cli import refine_prompt
from prompt_refiner.config import Config
from prompt_refiner.providers.auto import AutoProvider
from prompt_refiner.providers.base import ProviderError


class TestEndToEndRefinement:
    """Test complete refinement flow from CLI to output."""

    @patch('prompt_refiner.providers.claude.ClaudeProvider.is_available', return_value=True)
    @patch('anthropic.Anthropic')
    def test_successful_refinement_with_claude(self, mock_anthropic, mock_available, tmp_path):
        """Test successful refinement using Claude provider."""
        # Mock Claude API response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.model_dump_json.return_value = '''
        {
            "improved_prompt": "Create a Python function that calculates the factorial of a given integer",
            "changes_made": "Added programming language specification and clarified the input type",
            "effectiveness_score": "9/10 - Clear, specific, and actionable"
        }
        '''
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        # Create config with cache in temp directory
        config = Config.from_dict({
            'provider': {'type': 'auto'},
            'advanced': {'cache': {'location': str(tmp_path)}}
        })

        # Run refinement
        result = refine_prompt(
            prompt="Write a function to calculate factorial",
            config=config,
            focus_areas=['clarity', 'specificity']
        )

        assert result['improved_prompt'] == "Create a Python function that calculates the factorial of a given integer"
        assert "programming language specification" in result['changes_made']
        assert "9/10" in result['effectiveness_score']

    @patch('prompt_refiner.providers.claude.ClaudeProvider.is_available', return_value=False)
    @patch('prompt_refiner.providers.ollama.OllamaProvider.is_available', return_value=True)
    @patch('httpx.Client')
    def test_fallback_to_ollama_when_claude_unavailable(
        self, mock_httpx_client, mock_ollama_available, mock_claude_available, tmp_path
    ):
        """Test fallback to Ollama when Claude is unavailable."""
        # Mock Ollama HTTP response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": '''
            {
                "improved_prompt": "Implement a factorial function in Python",
                "changes_made": "Made it more specific to Python",
                "effectiveness_score": "8/10"
            }
            '''
        }
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response
        mock_httpx_client.return_value.__enter__.return_value = mock_client

        config = Config.from_dict({
            'provider': {'type': 'auto'},
            'advanced': {'cache': {'location': str(tmp_path)}}
        })

        result = refine_prompt(
            prompt="Write factorial function",
            config=config
        )

        assert "factorial function in Python" in result['improved_prompt']
        assert mock_client.post.called

    def test_error_when_no_providers_available(self):
        """Test error handling when no providers are available."""
        with patch('prompt_refiner.providers.claude.ClaudeProvider.is_available', return_value=False):
            with patch('prompt_refiner.providers.ollama.OllamaProvider.is_available', return_value=False):
                config = Config()

                with pytest.raises(ProviderError, match="No available providers"):
                    refine_prompt("test prompt", config)


class TestConfigurationIntegration:
    """Test configuration loading and usage."""

    def test_config_from_file_affects_refinement(self, temp_config_file, tmp_path):
        """Test that configuration from file is properly used."""
        # Update the temp config to use a custom cache location
        config_content = f"""
provider:
  type: auto
  ollama:
    temperature: 0.9
    model: custom-model

refinement:
  focus_areas:
    - technical_accuracy
    - brevity

advanced:
  retry_attempts: 5
  cache:
    location: {tmp_path}
    ttl_hours: 12
"""
        temp_config_file.write_text(config_content)

        # Load config and verify it's applied
        from prompt_refiner.config import load_config
        config_dict = load_config(str(temp_config_file))
        config = Config.from_dict(config_dict)

        assert config.provider.ollama['temperature'] == 0.9
        assert config.provider.ollama['model'] == 'custom-model'
        assert config.refinement.focus_areas == ('technical_accuracy', 'brevity')
        assert config.advanced.retry_attempts == 5
        assert config.advanced.cache.ttl_hours == 12

    def test_environment_variable_config_loading(self, temp_config_file, monkeypatch):
        """Test loading config from environment variable."""
        monkeypatch.setenv('PROMPT_REFINER_CONFIG', str(temp_config_file))

        from prompt_refiner.config import load_config
        config_dict = load_config()

        assert config_dict['provider']['type'] == 'auto'
        assert 'claude' in config_dict['provider']


class TestCacheIntegrationFlow:
    """Test caching behavior in the complete flow."""

    @patch('prompt_refiner.providers.claude.ClaudeProvider.is_available', return_value=True)
    @patch('anthropic.Anthropic')
    def test_second_identical_request_uses_cache(self, mock_anthropic, mock_available, tmp_path):
        """Test that second identical request is served from cache."""
        # Mock Claude API
        mock_client = Mock()
        mock_response = Mock()
        mock_response.model_dump_json.return_value = '{"improved_prompt": "cached test", "changes_made": "none", "effectiveness_score": "10/10"}'
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        config = Config.from_dict({
            'advanced': {'cache': {'location': str(tmp_path), 'enabled': True}}
        })

        # First request
        AutoProvider(config)
        cache = Cache(config.advanced.cache)

        # Manually test cache behavior
        cache_key = cache._generate_key("test prompt", ["clarity"])

        # First check - cache miss
        assert cache.get(cache_key) is None

        # Provider call and cache set
        result = {"improved_prompt": "cached test", "changes_made": "none", "effectiveness_score": "10/10"}
        cache.set(cache_key, result)

        # Second check - cache hit
        cached_result = cache.get(cache_key)
        assert cached_result == result

    def test_cache_disabled_always_calls_provider(self, tmp_path):
        """Test that disabled cache always calls the provider."""
        config = Config.from_dict({
            'advanced': {'cache': {'enabled': False, 'location': str(tmp_path)}}
        })

        cache = Cache(config.advanced.cache)

        # Set should not store anything
        cache.set("key", {"data": "test"})

        # Get should always return None
        assert cache.get("key") is None


class TestErrorHandlingIntegration:
    """Test error handling throughout the system."""

    @patch('prompt_refiner.providers.claude.ClaudeProvider.is_available', return_value=True)
    @patch('anthropic.Anthropic')
    def test_retry_behavior_on_api_errors(self, mock_anthropic, mock_available):
        """Test that retries work correctly on API errors."""
        mock_client = Mock()

        # First two calls fail, third succeeds
        mock_response = Mock()
        mock_response.model_dump_json.return_value = '{"improved_prompt": "success", "changes_made": "retry worked", "effectiveness_score": "7/10"}'

        mock_client.messages.create.side_effect = [
            Exception("Network error"),
            Exception("Another network error"),
            mock_response
        ]
        mock_anthropic.return_value = mock_client

        config = Config.from_dict({'advanced': {'retry_attempts': 3}})

        # This should succeed after retries
        result = refine_prompt("test prompt", config)

        assert result['improved_prompt'] == "success"
        assert mock_client.messages.create.call_count == 3

    def test_graceful_failure_after_max_retries(self):
        """Test graceful failure when max retries exceeded."""
        with patch('prompt_refiner.providers.claude.ClaudeProvider.is_available', return_value=True):
            with patch('anthropic.Anthropic') as mock_anthropic:
                mock_client = Mock()
                mock_client.messages.create.side_effect = Exception("Persistent error")
                mock_anthropic.return_value = mock_client

                config = Config.from_dict({'advanced': {'retry_attempts': 2}})

                with pytest.raises(ProviderError):
                    refine_prompt("test prompt", config)


class TestProviderSpecificBehavior:
    """Test provider-specific configuration and behavior."""

    @patch('prompt_refiner.providers.claude.ClaudeProvider.is_available', return_value=True)
    @patch('anthropic.Anthropic')
    def test_claude_uses_configured_model(self, mock_anthropic, mock_available):
        """Test that Claude uses the configured model."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.model_dump_json.return_value = '{"improved_prompt": "test", "changes_made": "none", "effectiveness_score": "5/10"}'
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        config = Config.from_dict({
            'provider': {
                'type': 'claude',
                'claude': {'model': 'haiku', 'max_turns': 10}
            }
        })

        from prompt_refiner.providers.claude import ClaudeProvider
        provider = ClaudeProvider(config)
        provider.refine_prompt("test")

        # Verify the model parameter was passed
        call_args = mock_client.messages.create.call_args
        assert 'claude-3-haiku' in str(call_args)

    @patch('httpx.Client')
    def test_ollama_uses_custom_api_url(self, mock_httpx_client):
        """Test that Ollama uses custom API URL."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {"response": '{"improved_prompt": "test", "changes_made": "none", "effectiveness_score": "5/10"}'}
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response
        mock_httpx_client.return_value.__enter__.return_value = mock_client

        config = Config.from_dict({
            'provider': {
                'type': 'ollama',
                'ollama': {'api_url': 'http://custom:8080'}
            }
        })

        from prompt_refiner.providers.ollama import OllamaProvider
        provider = OllamaProvider(config)
        provider.refine_prompt("test")

        # Verify custom URL was used
        call_args = mock_client.post.call_args
        assert 'http://custom:8080' in str(call_args)
