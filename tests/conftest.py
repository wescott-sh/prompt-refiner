"""Shared fixtures for prompt-refiner tests."""

from pathlib import Path
from typing import Dict, Generator

import pytest


@pytest.fixture
def temp_config_file(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary config file for testing."""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text("""
provider:
  type: auto
  claude:
    enabled: true
  ollama:
    api_url: http://localhost:11434
    model: llama2
    temperature: 0.7

refinement:
  focus_areas:
    - clarity
    - actionability
  templates:
    default:
      emphasis: clarity and actionability
    coding:
      emphasis: technical precision
  output:
    include_explanation: true
    include_score: true

advanced:
  retry_attempts: 3
  timeout_seconds: 30
  cache:
    enabled: true
    ttl_hours: 24
    location: ~/.cache/prompt-refiner
""")
    yield config_file


@pytest.fixture
def mock_provider_response() -> Dict[str, str]:
    """Mock response from LLM provider."""
    return {
        "improved_prompt": "Write a Python function that calculates factorial",
        "changes_made": "Added specificity about programming language",
        "effectiveness_score": "8/10 - Clear and actionable"
    }


@pytest.fixture
def sample_prompt() -> str:
    """Sample prompt for testing."""
    return "Write a function to calculate factorial"
