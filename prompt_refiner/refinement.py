"""Core refinement logic for prompt-refiner."""

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Optional

from prompt_refiner.cache import Cache
from prompt_refiner.config import (
    AdvancedConfig,
    CacheConfig,
    Config,
    ProviderConfig,
    load_config,
)


class PromptRefiner:
    def __init__(self, config_path: Optional[str] = None, no_cache: bool = False, provider: Optional[str] = None):
        config_dict = load_config(config_path)
        self.config = Config.from_dict(config_dict)

        # Override provider if specified
        if provider:
            self.config = Config(
                provider=ProviderConfig(
                    type=provider,
                    claude=self.config.provider.claude,
                    ollama=self.config.provider.ollama
                ),
                refinement=self.config.refinement,
                advanced=self.config.advanced
            )

        # Override cache if no_cache is specified
        if no_cache:
            self.config = Config(
                provider=self.config.provider,
                refinement=self.config.refinement,
                advanced=AdvancedConfig(
                    retry_attempts=self.config.advanced.retry_attempts,
                    timeout_seconds=self.config.advanced.timeout_seconds,
                    cache=CacheConfig(
                        enabled=False,
                        ttl_hours=self.config.advanced.cache.ttl_hours,
                        location=self.config.advanced.cache.location
                    )
                )
            )

        self.provider = self._detect_provider()

        # Initialize cache
        self.cache = Cache(self.config.advanced.cache)

    def clear_cache(self) -> int:
        """Clear all cache files and return count of files removed"""
        return self.cache.clear()


    def _detect_provider(self) -> str:
        """Detect available LLM provider"""
        provider_type = self.config.provider.type

        if provider_type == 'auto':
            # Check for Claude first
            if shutil.which('claude'):
                return 'claude'
            # Check for Ollama
            elif self._check_ollama():
                return 'ollama'
            else:
                raise RuntimeError(
                    "No LLM provider found. Install Claude Code or Ollama."
                ) from None
        elif provider_type == 'claude':
            if not shutil.which('claude'):
                raise RuntimeError("Claude Code not found in PATH") from None
            return 'claude'
        elif provider_type == 'ollama':
            if not self._check_ollama():
                raise RuntimeError(
                    f"Ollama not available at {self.config.provider.ollama['api_url']}"
                ) from None
            return 'ollama'
        else:
            raise ValueError(f"Unknown provider type: {provider_type}") from None

    def _check_ollama(self) -> bool:
        """Check if Ollama is running"""
        try:
            url = self.config.provider.ollama['api_url']
            req = urllib.request.Request(f"{url}/api/tags")
            with urllib.request.urlopen(req, timeout=2) as response:
                return response.status == 200
        except Exception:
            return False




    def _build_refinement_prompt(self, original_prompt: str, template: str) -> str:
        """Build the refinement prompt based on template"""
        template_config = self.config.refinement.templates.get(
            template, self.config.refinement.templates['default']
        )

        emphasis = template_config.get('emphasis', 'clarity and actionability')
        focus_areas = ', '.join(self.config.refinement.focus_areas)

        return f"""Analyze and improve this prompt for {emphasis}:

Original prompt: "{original_prompt}"

Return a JSON object with exactly these fields:
{{
    "improved_prompt": "the refined version of the prompt",
    "changes_made": "brief explanation of key improvements",
    "effectiveness_score": "rating from 1-10 with brief justification"
}}

Focus on: {focus_areas}
Ensure the improved prompt is specific, actionable, and unambiguous."""

    def _refine_with_claude(self, prompt: str) -> Dict[str, str]:
        """Use Claude to refine the prompt"""
        for attempt in range(self.config.advanced.retry_attempts):
            try:
                result = subprocess.run(
                    ["claude", "--output-format", "json", "-p", prompt],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=self.config.advanced.timeout_seconds
                )

                # Parse the JSON response
                response_data = json.loads(result.stdout)

                # Extract the actual result from the response structure
                if isinstance(response_data, dict) and "result" in response_data:
                    result_text = response_data["result"]
                    # Remove markdown code block if present
                    if result_text.startswith("```json"):
                        result_text = result_text[7:]
                    if result_text.endswith("```"):
                        result_text = result_text[:-3]
                    return json.loads(result_text.strip())

                return response_data

            except subprocess.TimeoutExpired:
                if attempt < self.config.advanced.retry_attempts - 1:
                    time.sleep(1)
                    continue
                raise RuntimeError("Claude request timed out") from None
            except Exception as e:
                if attempt < self.config.advanced.retry_attempts - 1:
                    time.sleep(1)
                    continue
                raise e

    def _refine_with_ollama(self, prompt: str) -> Dict[str, str]:
        """Use Ollama to refine the prompt"""
        ollama_config = self.config.provider.ollama
        url = f"{ollama_config['api_url']}/api/generate"

        for attempt in range(self.config.advanced.retry_attempts):
            try:
                data = json.dumps({
                    'model': ollama_config['model'],
                    'prompt': prompt + "\n\nRespond with valid JSON only.",
                    'temperature': ollama_config['temperature'],
                    'stream': False,
                    'format': 'json'
                }).encode('utf-8')

                req = urllib.request.Request(
                    url,
                    data=data,
                    headers={'Content-Type': 'application/json'}
                )

                with urllib.request.urlopen(req, timeout=self.config.advanced.timeout_seconds) as response:
                    if response.status != 200:
                        raise RuntimeError(f"Ollama error: {response.status}") from None

                    result = json.loads(response.read().decode('utf-8'))
                    return json.loads(result['response'])

            except urllib.error.URLError as e:
                if attempt < self.config.advanced.retry_attempts - 1:
                    time.sleep(1)
                    continue
                raise RuntimeError("Ollama request failed: " + str(e)) from e
            except Exception as e:
                if attempt < self.config.advanced.retry_attempts - 1:
                    time.sleep(1)
                    continue
                raise e

    def refine_prompt(self, original_prompt: str, template: str = 'default') -> Dict[str, str]:
        """Refine a prompt using the configured provider"""
        # Check cache first
        cached = self.cache.get(original_prompt, template, self.provider)
        if cached:
            return cached

        # Build refinement prompt
        refinement_prompt = self._build_refinement_prompt(original_prompt, template)

        try:
            if self.provider == 'claude':
                result = self._refine_with_claude(refinement_prompt)
            else:  # ollama
                result = self._refine_with_ollama(refinement_prompt)

            # Save to cache
            self.cache.save(original_prompt, template, self.provider, result)

            return result

        except Exception as e:
            return {
                "error": str(e),
                "improved_prompt": original_prompt,
                "changes_made": "No changes due to error",
                "provider": self.provider
            }
