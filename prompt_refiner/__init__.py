"""prompt-refiner: A tool to automatically improve prompts using Claude or Ollama"""

from prompt_refiner.refinement import PromptRefiner
from prompt_refiner.cli import app

__version__ = "0.1.0"
__all__ = ["PromptRefiner", "app"]