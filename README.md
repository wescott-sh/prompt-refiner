# prompt-refiner

A tool that automatically improves prompts using Claude Code or Ollama.

## Features

- **Auto-detection**: Automatically uses Claude if available, falls back to Ollama
- **Template support**: Specialized refinement for coding, analysis, and writing tasks
- **Caching**: Saves refined prompts to avoid redundant API calls
- **Configurable**: YAML-based configuration for customization
- **No external dependencies**: Uses only Python standard library

## Installation

```bash
# Clone or download the script
chmod +x prompt_refiner.py
```

## Usage

### Basic usage
```bash
python3 prompt_refiner.py "your prompt here"
```

### With template
```bash
python3 prompt_refiner.py --template coding "write a function to parse CSV"
```

### Interactive mode
```bash
python3 prompt_refiner.py
# Then type your prompt and press Enter twice
```

### Verbose mode
```bash
python3 prompt_refiner.py --verbose "analyze this data"
```

## Configuration

Edit `config.yaml` to customize:
- Provider settings (Claude/Ollama)
- Refinement focus areas
- Cache settings
- Template definitions

## Templates

- **default**: General clarity and actionability
- **coding**: Technical specificity and implementation details
- **analysis**: Data sources and metrics focus
- **writing**: Tone, audience, and format emphasis

## Requirements

- Python 3.6+
- Either Claude Code or Ollama installed
- For Ollama: Service running on localhost:11434

## How it works

1. Takes your vague prompt
2. Analyzes it for missing context and specificity
3. Returns an improved version with clear requirements
4. Caches results for 24 hours