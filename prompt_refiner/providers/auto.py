"""Auto provider that selects the best available provider."""

from typing import Any, Dict, List, Optional

from ..config import Config
from .base import BaseProvider, ProviderError
from .claude import ClaudeProvider
from .ollama import OllamaProvider


class AutoProvider(BaseProvider):
    """Automatically selects the best available provider."""
    
    def __init__(self, config: Config):
        """Initialize auto provider with configuration."""
        super().__init__(config)
        self._provider = self._select_provider()
    
    def _select_provider(self) -> BaseProvider:
        """Select the best available provider."""
        # Check Claude first (preferred)
        if ClaudeProvider.is_available():
            return ClaudeProvider(self.config)
        
        # Fall back to Ollama
        if OllamaProvider.is_available():
            return OllamaProvider(self.config)
        
        raise ProviderError("No available providers found. Please ensure Claude API key is set or Ollama is running.")
    
    def refine_prompt(
        self,
        prompt: str,
        focus_areas: Optional[List[str]] = None,
        template: Optional[str] = None
    ) -> Dict[str, Any]:
        """Refine the prompt using the selected provider."""
        return self._provider.refine_prompt(prompt, focus_areas, template)
    
    @staticmethod
    def is_available() -> bool:
        """Check if any provider is available."""
        return ClaudeProvider.is_available() or OllamaProvider.is_available()