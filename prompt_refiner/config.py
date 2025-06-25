"""Configuration management for prompt-refiner"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Optional, Tuple

import yaml


@dataclass(frozen=True)
class CacheConfig:
    enabled: bool = True
    ttl_hours: int = 24
    location: str = '~/.cache/prompt-refiner'


@dataclass(frozen=True)
class ProviderConfig:
    type: str = 'auto'
    claude: Dict[str, Any] = field(default_factory=lambda: MappingProxyType({'model': 'opus'}))
    ollama: Dict[str, Any] = field(default_factory=lambda: MappingProxyType({
        'model': 'llama3.2',
        'api_url': 'http://localhost:11434',
        'temperature': 0.7
    }))


@dataclass(frozen=True)
class RefinementConfig:
    focus_areas: Tuple[str, ...] = ('clarity', 'specificity', 'actionability')
    output: Dict[str, Any] = field(default_factory=lambda: MappingProxyType({
        'include_score': True,
        'include_explanation': True,
        'verbose': False
    }))
    templates: Dict[str, Any] = field(default_factory=lambda: MappingProxyType({
        'default': {'emphasis': 'clarity and actionability'}
    }))


@dataclass(frozen=True)
class AdvancedConfig:
    retry_attempts: int = 2
    timeout_seconds: int = 30
    cache: CacheConfig = field(default_factory=CacheConfig)


@dataclass(frozen=True)
class Config:
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> 'Config':
        """Create Config from dictionary"""
        # Handle None or empty data
        if not data:
            return cls()

        provider_data = data.get('provider', {})
        refinement_data = data.get('refinement', {})
        advanced_data = data.get('advanced', {})
        cache_data = advanced_data.get('cache', {})

        # Handle None values for claude/ollama and merge with defaults
        claude_defaults = {'model': 'opus'}
        claude_data = provider_data.get('claude')
        if claude_data is None:
            claude_data = claude_defaults
        else:
            # Merge with defaults
            claude_data = {**claude_defaults, **claude_data}

        ollama_defaults = {
            'model': 'llama3.2',
            'api_url': 'http://localhost:11434',
            'temperature': 0.7
        }
        ollama_data = provider_data.get('ollama')
        if ollama_data is None:
            ollama_data = ollama_defaults
        else:
            # Merge with defaults
            ollama_data = {**ollama_defaults, **ollama_data}

        return cls(
            provider=ProviderConfig(
                type=provider_data.get('type') or 'auto',
                claude=MappingProxyType(claude_data),
                ollama=MappingProxyType(ollama_data)
            ),
            refinement=RefinementConfig(
                focus_areas=tuple(refinement_data.get('focus_areas', ['clarity', 'specificity', 'actionability'])),
                output=MappingProxyType({
                    **{'include_score': True, 'include_explanation': True, 'verbose': False},
                    **refinement_data.get('output', {})
                }),
                templates=MappingProxyType({
                    **{'default': {'emphasis': 'clarity and actionability'}},
                    **refinement_data.get('templates', {})
                })
            ),
            advanced=AdvancedConfig(
                retry_attempts=advanced_data.get('retry_attempts', 2),
                timeout_seconds=advanced_data.get('timeout_seconds', 30),
                cache=CacheConfig(
                    enabled=cache_data.get('enabled', True),
                    ttl_hours=cache_data.get('ttl_hours', 24),
                    location=cache_data.get('location', '~/.cache/prompt-refiner')
                )
            )
        )


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    # Check environment variable first
    env_config = os.environ.get('PROMPT_REFINER_CONFIG')
    if env_config and Path(env_config).exists():
        config_file = Path(env_config)
    elif config_path and Path(config_path).exists():
        config_file = Path(config_path)
    else:
        # Look for config in various locations
        # Get the script directory (parent of the prompt_refiner package)
        package_dir = Path(__file__).parent.parent
        config_locations = [
            package_dir / "config.yaml",
            Path.home() / ".config" / "prompt-refiner" / "config.yaml",
        ]

        config_file = None
        for loc in config_locations:
            if loc.exists():
                config_file = loc
                break

        if not config_file:
            # Return empty dict to use defaults
            return {}

    with open(config_file) as f:
        data = yaml.safe_load(f)
        return data if data is not None else {}
