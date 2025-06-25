# prompt-refiner

A modern Python package that automatically improves prompts using Claude Code or Ollama.

## Features

- **Auto-detection**: Automatically uses Claude if available, falls back to Ollama
- **Template support**: Specialized refinement for coding, analysis, and writing tasks
- **Smart caching**: Avoids redundant API calls with intelligent caching
- **Configurable**: YAML-based configuration for easy customization
- **Modular architecture**: Clean separation of providers, config, cache, and UI
- **Easy to extend**: Add new providers or templates with minimal effort

## Installation

```bash
# Install with pip
pip install .

# Or with uv (recommended)
uv pip install .

# For development
uv pip install -e .
```

## Usage

### Basic usage
```bash
# Use the full command
prompt-refiner "your prompt here"

# Or use the short alias
refine "write a function to parse CSV"
```

### With template
```bash
refine --template coding "implement binary search in Python"
```

### Interactive mode
```bash
refine
# Then type your prompt and press Enter twice
```

### Verbose mode
```bash
refine --verbose "analyze sales data for trends"
```

## Architecture

The package follows a clean, modular architecture:

```
prompt_refiner/
├── __main__.py       # Entry point for package execution
├── cli.py            # Command-line interface
├── main.py           # Main application logic
├── config.py         # Configuration management
├── cache.py          # Caching functionality
├── ui.py             # User interface helpers
└── providers/        # LLM provider implementations
    ├── base.py       # Abstract base provider
    ├── claude.py     # Claude integration
    └── ollama.py     # Ollama integration
```

This structure makes it easy to:
- Add new LLM providers by extending the base provider
- Customize the UI without touching core logic
- Manage configuration separately from implementation
- Test components in isolation

## Configuration

Edit `config.yaml` to customize:
- Provider settings (Claude/Ollama)
- Refinement focus areas
- Cache settings (TTL, location)
- Template definitions

Example configuration:
```yaml
providers:
  claude:
    enabled: true
    timeout: 30
  ollama:
    enabled: true
    url: http://localhost:11434
    model: llama3.2

cache:
  ttl_seconds: 86400  # 24 hours
  max_size_mb: 100

templates:
  coding:
    focus_areas:
      - technical_requirements
      - edge_cases
      - performance
```

## Templates

Built-in templates for common use cases:

- **default**: General clarity and actionability
- **coding**: Technical specs, edge cases, and implementation details
- **analysis**: Data sources, metrics, and analytical approach
- **writing**: Tone, audience, format, and style guidelines

## Extending

Add a new provider by implementing the base provider interface:

```python
from prompt_refiner.providers.base import BaseProvider

class MyProvider(BaseProvider):
    def refine(self, prompt: str, template: str = "default") -> str:
        # Your implementation here
        pass
```

## Requirements

- Python 3.8+
- Either Claude Code or Ollama installed
- For Ollama: Service running on localhost:11434

## How it works

1. Takes your initial prompt
2. Analyzes it for missing context and specificity
3. Applies template-specific improvements
4. Returns an enhanced version with clear requirements
5. Caches results to avoid redundant API calls