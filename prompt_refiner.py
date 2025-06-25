#!/usr/bin/env python3
# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "pyyaml>=6.0",
#   "rich",
#   "typer",
# ]
# ///
"""
prompt-refiner: A tool to automatically improve prompts using Claude or Ollama

Usage:
    python prompt_refiner.py "your prompt here"
    python prompt_refiner.py --config custom_config.yaml "your prompt"
    python prompt_refiner.py --template coding "write a function"
    
    or interactively:
    python prompt_refiner.py
"""

from prompt_refiner.cli import app

if __name__ == "__main__":
    app()