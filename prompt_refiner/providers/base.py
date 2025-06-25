"""Base provider abstraction and registry"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Type, Optional
import shutil


class BaseProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    def refine(self, prompt: str, retry_attempts: int, timeout_seconds: int) -> Dict[str, str]:
        """
        Refine a prompt using the provider's LLM.
        
        Args:
            prompt: The refinement prompt to send to the LLM
            retry_attempts: Number of retry attempts on failure
            timeout_seconds: Timeout for the request
            
        Returns:
            Dict containing the refined prompt and metadata
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available for use"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this provider"""
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