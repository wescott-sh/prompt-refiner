"""Tests for configuration module."""

import os
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict

import pytest
import yaml

from prompt_refiner.config import (
    AdvancedConfig,
    CacheConfig,
    Config,
    ProviderConfig,
    RefinementConfig,
    load_config,
)


class TestDefaultConfigCreation:
    """Test default configuration creation using Config.from_dict()."""

    def test_empty_dict_creates_all_defaults(self):
        """Test that an empty dict creates a config with all default values."""
        config = Config.from_dict({})
        
        # Test provider defaults
        assert config.provider.type == 'auto'
        assert config.provider.claude['model'] == 'opus'
        assert config.provider.ollama['model'] == 'llama3.2'
        assert config.provider.ollama['api_url'] == 'http://localhost:11434'
        assert config.provider.ollama['temperature'] == 0.7
        
        # Test refinement defaults
        assert config.refinement.focus_areas == ('clarity', 'specificity', 'actionability')
        assert config.refinement.output['include_score'] is True
        assert config.refinement.output['include_explanation'] is True
        assert config.refinement.output['verbose'] is False
        assert config.refinement.templates['default']['emphasis'] == 'clarity and actionability'
        
        # Test advanced defaults
        assert config.advanced.retry_attempts == 2
        assert config.advanced.timeout_seconds == 30
        assert config.advanced.cache.enabled is True
        assert config.advanced.cache.ttl_hours == 24
        assert config.advanced.cache.location == '~/.cache/prompt-refiner'

    def test_partial_dict_merges_with_defaults(self):
        """Test that partial configuration merges correctly with defaults."""
        config_data = {
            'provider': {
                'type': 'claude'
            },
            'refinement': {
                'focus_areas': ['technical', 'precision']
            }
        }
        config = Config.from_dict(config_data)
        
        # Changed values
        assert config.provider.type == 'claude'
        assert config.refinement.focus_areas == ('technical', 'precision')
        
        # Unchanged defaults
        assert config.provider.claude['model'] == 'opus'
        assert config.advanced.retry_attempts == 2
        assert config.refinement.output['include_score'] is True

    def test_nested_dict_creation(self):
        """Test that nested dictionaries are created properly."""
        config_data = {
            'advanced': {
                'cache': {
                    'enabled': False,
                    'ttl_hours': 48
                }
            }
        }
        config = Config.from_dict(config_data)
        
        assert config.advanced.cache.enabled is False
        assert config.advanced.cache.ttl_hours == 48
        assert config.advanced.cache.location == '~/.cache/prompt-refiner'  # default

    def test_dataclass_instantiation_without_from_dict(self):
        """Test direct dataclass instantiation."""
        cache_config = CacheConfig(enabled=False, ttl_hours=12)
        advanced_config = AdvancedConfig(cache=cache_config)
        config = Config(advanced=advanced_config)
        
        assert config.advanced.cache.enabled is False
        assert config.advanced.cache.ttl_hours == 12
        # Other fields should use defaults
        assert config.provider.type == 'auto'


class TestConfigOverridePrecedence:
    """Test that user overrides take precedence over defaults."""

    def test_complete_override(self):
        """Test complete override of all configuration values."""
        config_data = {
            'provider': {
                'type': 'ollama',
                'claude': {'model': 'sonnet'},
                'ollama': {
                    'model': 'mistral',
                    'api_url': 'http://localhost:8080',
                    'temperature': 0.9
                }
            },
            'refinement': {
                'focus_areas': ['brevity', 'clarity'],
                'output': {
                    'include_score': False,
                    'include_explanation': False,
                    'verbose': True
                },
                'templates': {
                    'custom': {'emphasis': 'conciseness'}
                }
            },
            'advanced': {
                'retry_attempts': 5,
                'timeout_seconds': 60,
                'cache': {
                    'enabled': False,
                    'ttl_hours': 12,
                    'location': '/tmp/cache'
                }
            }
        }
        config = Config.from_dict(config_data)
        
        # All values should be overridden
        assert config.provider.type == 'ollama'
        assert config.provider.claude['model'] == 'sonnet'
        assert config.provider.ollama['model'] == 'mistral'
        assert config.provider.ollama['api_url'] == 'http://localhost:8080'
        assert config.provider.ollama['temperature'] == 0.9
        
        assert config.refinement.focus_areas == ('brevity', 'clarity')
        assert config.refinement.output['include_score'] is False
        assert config.refinement.output['include_explanation'] is False
        assert config.refinement.output['verbose'] is True
        assert config.refinement.templates['custom']['emphasis'] == 'conciseness'
        
        assert config.advanced.retry_attempts == 5
        assert config.advanced.timeout_seconds == 60
        assert config.advanced.cache.enabled is False
        assert config.advanced.cache.ttl_hours == 12
        assert config.advanced.cache.location == '/tmp/cache'

    def test_mixed_override_and_defaults(self):
        """Test that non-overridden values keep their defaults."""
        config_data = {
            'provider': {
                'claude': {'model': 'haiku'}  # only override model, not max_turns
            },
            'refinement': {
                'output': {'verbose': True}  # only override verbose
            }
        }
        config = Config.from_dict(config_data)
        
        # Overridden values
        assert config.provider.claude['model'] == 'haiku'
        assert config.refinement.output['verbose'] is True
        
        # Default values should remain
        assert config.refinement.output['include_score'] is True
        assert config.refinement.output['include_explanation'] is True

    def test_empty_list_override(self):
        """Test that empty lists properly override defaults."""
        config_data = {
            'refinement': {
                'focus_areas': []
            }
        }
        config = Config.from_dict(config_data)
        
        assert config.refinement.focus_areas == ()


class TestConfigImmutability:
    """Test that frozen dataclasses prevent mutations."""

    def test_cannot_modify_top_level_fields(self):
        """Test that top-level Config fields cannot be modified."""
        config = Config()
        
        with pytest.raises(FrozenInstanceError):
            config.provider = ProviderConfig(type='ollama')

    def test_cannot_modify_nested_dataclass_fields(self):
        """Test that nested dataclass fields cannot be modified."""
        config = Config()
        
        with pytest.raises(FrozenInstanceError):
            config.provider.type = 'ollama'
        
        with pytest.raises(FrozenInstanceError):
            config.advanced.cache.enabled = False

    def test_mapping_proxy_prevents_dict_mutations(self):
        """Test that MappingProxyType prevents dictionary mutations."""
        config = Config()
        
        # These should raise TypeError since MappingProxyType is immutable
        with pytest.raises(TypeError):
            config.provider.claude['model'] = 'sonnet'
        
        with pytest.raises(TypeError):
            config.refinement.output['verbose'] = True
        
        with pytest.raises(TypeError):
            del config.refinement.templates['default']

    def test_tuple_immutability(self):
        """Test that tuple fields are immutable."""
        config = Config()
        
        # Can't assign to tuple
        with pytest.raises(FrozenInstanceError):
            config.refinement.focus_areas = ('new', 'areas')
        
        # Can't modify tuple in place (tuples are inherently immutable)
        with pytest.raises(AttributeError):
            config.refinement.focus_areas.append('new_area')


class TestLoadConfig:
    """Test the load_config function."""

    def test_load_from_file_path(self, temp_config_file):
        """Test loading configuration from a specific file path."""
        config_dict = load_config(str(temp_config_file))
        
        assert config_dict['provider']['type'] == 'auto'
        assert config_dict['provider']['ollama']['model'] == 'llama2'
        assert config_dict['refinement']['focus_areas'] == ['clarity', 'actionability']
        assert config_dict['advanced']['retry_attempts'] == 3

    def test_load_from_environment_variable(self, temp_config_file, monkeypatch):
        """Test loading configuration from PROMPT_REFINER_CONFIG env var."""
        monkeypatch.setenv('PROMPT_REFINER_CONFIG', str(temp_config_file))
        
        config_dict = load_config()
        
        assert config_dict['provider']['type'] == 'auto'
        assert config_dict['advanced']['cache']['enabled'] is True

    def test_env_var_takes_precedence(self, temp_config_file, tmp_path, monkeypatch):
        """Test that env var takes precedence over passed path."""
        # Create another config file
        other_config = tmp_path / "other_config.yaml"
        other_config.write_text("""
provider:
  type: claude
""")
        
        monkeypatch.setenv('PROMPT_REFINER_CONFIG', str(temp_config_file))
        
        # Pass different path, but env var should win
        config_dict = load_config(str(other_config))
        
        assert config_dict['provider']['type'] == 'auto'  # from temp_config_file

    def test_returns_empty_dict_when_no_config_found(self, tmp_path, monkeypatch):
        """Test that empty dict is returned when no config file is found."""
        # Set a non-existent path
        nonexistent_path = str(tmp_path / 'nonexistent.yaml')
        monkeypatch.setenv('PROMPT_REFINER_CONFIG', nonexistent_path)
        
        # Explicitly pass the nonexistent path to bypass fallback locations
        config_dict = load_config(nonexistent_path)
        
        assert config_dict == {}

    def test_invalid_yaml_raises_error(self, tmp_path):
        """Test that invalid YAML raises an appropriate error."""
        invalid_config = tmp_path / "invalid.yaml"
        invalid_config.write_text("""
invalid: yaml: content
  bad indentation
""")
        
        with pytest.raises(yaml.YAMLError):
            load_config(str(invalid_config))


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_none_values_in_dict(self):
        """Test handling of None values in configuration dict."""
        config_data = {
            'provider': {
                'type': None,  # Should use default
                'claude': None  # Should use default
            }
        }
        config = Config.from_dict(config_data)
        
        # None should be treated as missing, using defaults
        assert config.provider.type == 'auto'
        assert config.provider.claude['model'] == 'opus'

    def test_extra_keys_are_ignored(self):
        """Test that extra keys in config dict are safely ignored."""
        config_data = {
            'provider': {
                'type': 'auto',
                'unknown_provider': {'model': 'test'},  # Extra key
            },
            'unknown_section': {'key': 'value'},  # Extra section
        }
        config = Config.from_dict(config_data)
        
        # Should create config without errors
        assert config.provider.type == 'auto'
        assert not hasattr(config.provider, 'unknown_provider')
        assert not hasattr(config, 'unknown_section')

    def test_special_characters_in_strings(self):
        """Test handling of special characters in string values."""
        config_data = {
            'provider': {
                'ollama': {
                    'api_url': 'http://localhost:8080/v1/api?key=test&value=123'
                }
            },
            'advanced': {
                'cache': {
                    'location': '~/cache/prompt-refiner/with spaces/and-special_chars'
                }
            }
        }
        config = Config.from_dict(config_data)
        
        assert config.provider.ollama['api_url'] == 'http://localhost:8080/v1/api?key=test&value=123'
        assert config.advanced.cache.location == '~/cache/prompt-refiner/with spaces/and-special_chars'

    def test_numeric_edge_cases(self):
        """Test edge cases for numeric configuration values."""
        config_data = {
            'advanced': {
                'retry_attempts': 0,
                'timeout_seconds': 999999,
                'cache': {
                    'ttl_hours': 0
                }
            }
        }
        config = Config.from_dict(config_data)
        
        assert config.advanced.retry_attempts == 0
        assert config.advanced.timeout_seconds == 999999
        assert config.advanced.cache.ttl_hours == 0

    def test_large_config_dict(self):
        """Test handling of large configuration dictionaries."""
        # Create a large templates section
        large_templates = {f'template_{i}': {'emphasis': f'focus_{i}'} for i in range(100)}
        
        config_data = {
            'refinement': {
                'templates': large_templates
            }
        }
        config = Config.from_dict(config_data)
        
        assert len(config.refinement.templates) == 101  # 100 + default template
        assert config.refinement.templates['template_50']['emphasis'] == 'focus_50'
        assert config.refinement.templates['default']['emphasis'] == 'clarity and actionability'


class TestDataclassDefaults:
    """Test that dataclass field defaults work correctly."""

    def test_cache_config_defaults(self):
        """Test CacheConfig default values."""
        cache = CacheConfig()
        assert cache.enabled is True
        assert cache.ttl_hours == 24
        assert cache.location == '~/.cache/prompt-refiner'

    def test_provider_config_defaults(self):
        """Test ProviderConfig default values."""
        provider = ProviderConfig()
        assert provider.type == 'auto'
        assert isinstance(provider.claude, MappingProxyType)
        assert isinstance(provider.ollama, MappingProxyType)

    def test_refinement_config_defaults(self):
        """Test RefinementConfig default values."""
        refinement = RefinementConfig()
        assert refinement.focus_areas == ('clarity', 'specificity', 'actionability')
        assert isinstance(refinement.output, MappingProxyType)
        assert isinstance(refinement.templates, MappingProxyType)

    def test_advanced_config_defaults(self):
        """Test AdvancedConfig default values."""
        advanced = AdvancedConfig()
        assert advanced.retry_attempts == 2
        assert advanced.timeout_seconds == 30
        assert isinstance(advanced.cache, CacheConfig)


class TestIntegration:
    """Integration tests combining load_config and Config.from_dict."""

    def test_full_workflow(self, temp_config_file):
        """Test complete workflow from file to Config object."""
        # Load from file
        config_dict = load_config(str(temp_config_file))
        
        # Create Config object
        config = Config.from_dict(config_dict)
        
        # Verify values from file
        assert config.provider.type == 'auto'
        assert config.provider.ollama['model'] == 'llama2'
        assert config.refinement.focus_areas == ('clarity', 'actionability')
        assert config.refinement.templates['coding']['emphasis'] == 'technical precision'
        assert config.advanced.retry_attempts == 3
        
        # Verify immutability
        with pytest.raises(FrozenInstanceError):
            config.provider.type = 'claude'

    def test_empty_file_uses_all_defaults(self, tmp_path):
        """Test that empty config file results in all defaults."""
        empty_config = tmp_path / "empty.yaml"
        empty_config.write_text("")
        
        config_dict = load_config(str(empty_config))
        config = Config.from_dict(config_dict)
        
        # Should have all defaults
        assert config.provider.type == 'auto'
        assert config.refinement.focus_areas == ('clarity', 'specificity', 'actionability')
        assert config.advanced.cache.enabled is True