"""Tests for provider modules."""

from unittest.mock import Mock, patch

import pytest

from prompt_refiner.config import Config
from prompt_refiner.providers.auto import AutoProvider
from prompt_refiner.providers.base import ProviderError
from prompt_refiner.providers.claude import ClaudeProvider
from prompt_refiner.providers.ollama import OllamaProvider


class TestAutoProviderSelection:
    """Test auto-selection picks available provider."""

    @patch('prompt_refiner.providers.claude.ClaudeProvider.is_available')
    def test_selects_claude_when_available(self, mock_claude_available):
        """Test that Claude is selected when available."""
        mock_claude_available.return_value = True
        config = Config()
        
        provider = AutoProvider(config)
        selected = provider._select_provider()
        
        assert isinstance(selected, ClaudeProvider)

    @patch('prompt_refiner.providers.claude.ClaudeProvider.is_available')
    @patch('prompt_refiner.providers.ollama.OllamaProvider.is_available')
    def test_selects_ollama_when_claude_unavailable(
        self, mock_ollama_available, mock_claude_available
    ):
        """Test that Ollama is selected when Claude is unavailable."""
        mock_claude_available.return_value = False
        mock_ollama_available.return_value = True
        config = Config()
        
        provider = AutoProvider(config)
        selected = provider._select_provider()
        
        assert isinstance(selected, OllamaProvider)

    @patch('prompt_refiner.providers.claude.ClaudeProvider.is_available')
    @patch('prompt_refiner.providers.ollama.OllamaProvider.is_available')
    def test_raises_error_when_no_providers_available(
        self, mock_ollama_available, mock_claude_available
    ):
        """Test error when no providers are available."""
        mock_claude_available.return_value = False
        mock_ollama_available.return_value = False
        config = Config()
        
        with pytest.raises(ProviderError, match="No available providers"):
            provider = AutoProvider(config)


class TestClaudeProviderResponse:
    """Test Claude provider with mock responses."""

    @patch('anthropic.Anthropic')
    def test_successful_refinement(self, mock_anthropic):
        """Test successful prompt refinement with Claude."""
        # Mock the API response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.model_dump_json.return_value = '{"improved_prompt": "Write a Python function that calculates factorial", "changes_made": "Added specificity", "effectiveness_score": "8/10"}'
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        config = Config()
        provider = ClaudeProvider(config)
        
        result = provider.refine_prompt(
            "Write a function to calculate factorial",
            focus_areas=['clarity']
        )
        
        assert "improved_prompt" in result
        assert result["improved_prompt"] == "Write a Python function that calculates factorial"
        assert "changes_made" in result
        assert "effectiveness_score" in result

    @patch('anthropic.Anthropic')
    def test_handles_api_error(self, mock_anthropic):
        """Test handling of API errors."""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_anthropic.return_value = mock_client
        
        config = Config()
        provider = ClaudeProvider(config)
        
        with pytest.raises(ProviderError, match="Claude API error"):
            provider.refine_prompt("test prompt")

    def test_is_available_with_api_key(self):
        """Test availability check when API key is present."""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            assert ClaudeProvider.is_available() is True

    def test_is_available_without_api_key(self):
        """Test availability check when API key is missing."""
        with patch.dict('os.environ', {}, clear=True):
            assert ClaudeProvider.is_available() is False


class TestOllamaProviderFallback:
    """Test Ollama as fallback when Claude unavailable."""

    @patch('httpx.Client')
    def test_successful_refinement(self, mock_httpx_client):
        """Test successful prompt refinement with Ollama."""
        # Mock the HTTP response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": '{"improved_prompt": "Write a Python function that calculates factorial", "changes_made": "Added language specificity", "effectiveness_score": "7/10"}'
        }
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response
        mock_httpx_client.return_value.__enter__.return_value = mock_client
        
        config = Config()
        provider = OllamaProvider(config)
        
        result = provider.refine_prompt(
            "Write a function to calculate factorial",
            focus_areas=['clarity']
        )
        
        assert "improved_prompt" in result
        assert "changes_made" in result
        assert "effectiveness_score" in result

    @patch('httpx.get')
    def test_is_available_when_server_running(self, mock_get):
        """Test availability when Ollama server is running."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        assert OllamaProvider.is_available() is True

    @patch('httpx.get')
    def test_is_available_when_server_not_running(self, mock_get):
        """Test availability when Ollama server is not running."""
        mock_get.side_effect = Exception("Connection refused")
        
        assert OllamaProvider.is_available() is False


class TestProviderErrorHandling:
    """Test network errors and retries."""

    @patch('anthropic.Anthropic')
    def test_claude_retry_on_transient_error(self, mock_anthropic):
        """Test that Claude retries on transient errors."""
        mock_client = Mock()
        # First call fails, second succeeds
        mock_response = Mock()
        mock_response.model_dump_json.return_value = '{"improved_prompt": "test", "changes_made": "none", "effectiveness_score": "5/10"}'
        mock_client.messages.create.side_effect = [
            Exception("Network error"),
            mock_response
        ]
        mock_anthropic.return_value = mock_client
        
        config = Config.from_dict({'advanced': {'retry_attempts': 2}})
        provider = ClaudeProvider(config)
        
        result = provider.refine_prompt("test prompt")
        
        assert "improved_prompt" in result
        assert mock_client.messages.create.call_count == 2

    @patch('httpx.Client')
    def test_ollama_timeout_handling(self, mock_httpx_client):
        """Test Ollama timeout handling."""
        mock_client = Mock()
        mock_client.post.side_effect = Exception("Request timeout")
        mock_httpx_client.return_value.__enter__.return_value = mock_client
        
        config = Config.from_dict({'advanced': {'timeout_seconds': 5}})
        provider = OllamaProvider(config)
        
        with pytest.raises(ProviderError, match="Failed to connect"):
            provider.refine_prompt("test prompt")

    @patch('anthropic.Anthropic')
    def test_max_retries_exceeded(self, mock_anthropic):
        """Test behavior when max retries are exceeded."""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("Persistent error")
        mock_anthropic.return_value = mock_client
        
        config = Config.from_dict({'advanced': {'retry_attempts': 2}})
        provider = ClaudeProvider(config)
        
        with pytest.raises(ProviderError, match="Claude API error"):
            provider.refine_prompt("test prompt")
        
        # Should have tried initial + 2 retries = 3 total
        assert mock_client.messages.create.call_count == 3