"""Base provider abstraction and registry"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type


class ProviderError(Exception):
    """Base exception for provider-related errors"""
    pass


class BaseProvider(ABC):
    """Abstract base class for LLM providers"""

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def refine_prompt(
        self,
        prompt: str,
        focus_areas: Optional[List[str]] = None,
        template: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Refine a prompt using the provider's LLM.

        Args:
            prompt: The prompt to refine
            focus_areas: Areas to focus on during refinement
            template: Template to use for refinement

        Returns:
            Dict containing the refined prompt and metadata
        """
        pass

    @staticmethod
    @abstractmethod
    def is_available() -> bool:
        """Check if this provider is available for use"""
        pass


class ProviderRegistry:
    """Registry for managing available providers"""

    _providers: Dict[str, Type[BaseProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_class: Type[BaseProvider]):
        """Register a provider class"""
        cls._providers[name] = provider_class

    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseProvider]]:
        """Get a provider class by name"""
        return cls._providers.get(name)

    @classmethod
    def list_available(cls) -> list[str]:
        """List all registered provider names"""
        return list(cls._providers.keys())

    @classmethod
    def detect_available(cls, config: Dict[str, Any]) -> Optional[str]:
        """Detect the first available provider"""
        for name, provider_class in cls._providers.items():
            provider = provider_class(config.get(name, {}))
            if provider.is_available():
                return name
        return None
