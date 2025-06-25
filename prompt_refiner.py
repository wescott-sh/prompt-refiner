#!/usr/bin/env python3
# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "pyyaml>=6.0",
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

import sys
import json
import yaml
import subprocess
import argparse
import os
from pathlib import Path
from typing import Dict, Optional, Any, FrozenSet, Tuple
from dataclasses import dataclass, field
from types import MappingProxyType
import hashlib
import time
from datetime import datetime, timedelta
import shutil
import urllib.request
import urllib.error


@dataclass(frozen=True)
class CacheConfig:
    enabled: bool = True
    ttl_hours: int = 24
    location: str = '~/.cache/prompt-refiner'


@dataclass(frozen=True)
class ProviderConfig:
    type: str = 'auto'
    claude: Dict[str, Any] = field(default_factory=lambda: MappingProxyType({'model': 'opus', 'max_turns': 5}))
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
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create Config from dictionary"""
        provider_data = data.get('provider', {})
        refinement_data = data.get('refinement', {})
        advanced_data = data.get('advanced', {})
        cache_data = advanced_data.get('cache', {})
        
        return cls(
            provider=ProviderConfig(
                type=provider_data.get('type', 'auto'),
                claude=MappingProxyType(provider_data.get('claude', {'model': 'opus', 'max_turns': 5})),
                ollama=MappingProxyType(provider_data.get('ollama', {
                    'model': 'llama3.2',
                    'api_url': 'http://localhost:11434',
                    'temperature': 0.7
                }))
            ),
            refinement=RefinementConfig(
                focus_areas=tuple(refinement_data.get('focus_areas', ['clarity', 'specificity', 'actionability'])),
                output=MappingProxyType(refinement_data.get('output', {
                    'include_score': True,
                    'include_explanation': True,
                    'verbose': False
                })),
                templates=MappingProxyType(refinement_data.get('templates', {
                    'default': {'emphasis': 'clarity and actionability'}
                }))
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


class PromptRefiner:
    def __init__(self, config_path: Optional[str] = None, no_cache: bool = False, provider: Optional[str] = None):
        config_dict = self._load_config(config_path)
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
        self.cache_dir = Path(os.path.expanduser(
            self.config.advanced.cache.location
        ))
        if self.config.advanced.cache.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def clear_cache(self) -> int:
        """Clear all cache files and return count of files removed"""
        if not self.cache_dir.exists():
            return 0
        
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1
        return count
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        # Check environment variable first
        env_config = os.environ.get('PROMPT_REFINER_CONFIG')
        if env_config and Path(env_config).exists():
            config_file = Path(env_config)
        elif config_path and Path(config_path).exists():
            config_file = Path(config_path)
        else:
            # Look for config in script directory first, then user home
            script_dir = Path(__file__).parent
            config_locations = [
                script_dir / "config.yaml",
                Path.home() / ".config" / "prompt-refiner" / "config.yaml",
            ]
            
            config_file = None
            for loc in config_locations:
                if loc.exists():
                    config_file = loc
                    break
            
            if not config_file:
                # Return default config
                return Config().provider.__dict__  # Return empty dict to use defaults
        
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        # Return empty dict - Config dataclass has all defaults
        return {}
    
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
                )
        elif provider_type == 'claude':
            if not shutil.which('claude'):
                raise RuntimeError("Claude Code not found in PATH")
            return 'claude'
        elif provider_type == 'ollama':
            if not self._check_ollama():
                raise RuntimeError(
                    f"Ollama not available at {self.config.provider.ollama['api_url']}"
                )
            return 'ollama'
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is running"""
        try:
            url = self.config.provider.ollama['api_url']
            req = urllib.request.Request(f"{url}/api/tags")
            with urllib.request.urlopen(req, timeout=2) as response:
                return response.status == 200
        except:
            return False
    
    def _get_cache_key(self, prompt: str, template: str) -> str:
        """Generate cache key for a prompt"""
        content = f"{prompt}:{template}:{self.provider}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _get_from_cache(self, prompt: str, template: str) -> Optional[Dict[str, str]]:
        """Retrieve from cache if available and not expired"""
        if not self.config.advanced.cache.enabled:
            return None
        
        cache_key = self._get_cache_key(prompt, template)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cached = json.load(f)
            
            # Check TTL
            cached_time = datetime.fromisoformat(cached['timestamp'])
            ttl_hours = self.config.advanced.cache.ttl_hours
            if datetime.now() - cached_time > timedelta(hours=ttl_hours):
                cache_file.unlink()  # Delete expired cache
                return None
            
            return cached['result']
        except:
            return None
    
    def _save_to_cache(self, prompt: str, template: str, result: Dict[str, str]):
        """Save result to cache"""
        if not self.config.advanced.cache.enabled:
            return
        
        cache_key = self._get_cache_key(prompt, template)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        with open(cache_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'prompt': prompt,
                'template': template,
                'provider': self.provider,
                'result': result
            }, f, indent=2)
    
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
                raise RuntimeError("Claude request timed out")
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
                        raise RuntimeError(f"Ollama error: {response.status}")
                    
                    result = json.loads(response.read().decode('utf-8'))
                    return json.loads(result['response'])
                
            except urllib.error.URLError as e:
                if attempt < self.config.advanced.retry_attempts - 1:
                    time.sleep(1)
                    continue
                raise RuntimeError("Ollama request failed: " + str(e))
            except Exception as e:
                if attempt < self.config.advanced.retry_attempts - 1:
                    time.sleep(1)
                    continue
                raise e
    
    def refine_prompt(self, original_prompt: str, template: str = 'default') -> Dict[str, str]:
        """Refine a prompt using the configured provider"""
        # Check cache first
        cached = self._get_from_cache(original_prompt, template)
        if cached:
            cached['from_cache'] = True
            return cached
        
        # Build refinement prompt
        refinement_prompt = self._build_refinement_prompt(original_prompt, template)
        
        try:
            if self.provider == 'claude':
                result = self._refine_with_claude(refinement_prompt)
            else:  # ollama
                result = self._refine_with_ollama(refinement_prompt)
            
            # Save to cache
            self._save_to_cache(original_prompt, template, result)
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "improved_prompt": original_prompt,
                "changes_made": "No changes due to error",
                "provider": self.provider
            }


def main():
    parser = argparse.ArgumentParser(
        description="Automatically improve prompts using LLMs"
    )
    parser.add_argument('prompt', nargs='*', help='The prompt to refine')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument(
        '--template', 
        choices=['default', 'coding', 'analysis', 'writing'],
        default='default',
        help='Template to use for refinement'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Show detailed output'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable cache for this run'
    )
    parser.add_argument(
        '--provider',
        choices=['auto', 'claude', 'ollama'],
        help='Choose provider explicitly'
    )
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear all cache files before running'
    )
    
    args = parser.parse_args()
    
    # Initialize refiner
    try:
        refiner = PromptRefiner(
            config_path=args.config,
            no_cache=args.no_cache,
            provider=args.provider
        )
    except Exception as e:
        print(f"❌ Error initializing: {e}")
        sys.exit(1)
    
    # Handle cache clearing if requested
    if args.clear_cache:
        count = refiner.clear_cache()
        print(f"🗑️  Cleared {count} cache file(s)")
        if not args.prompt:
            # If no prompt provided and only clearing cache, exit
            sys.exit(0)
    
    # Get prompt
    if args.prompt:
        original_prompt = " ".join(args.prompt)
    else:
        # Interactive mode
        print("Enter your prompt (press Enter twice to finish):")
        lines = []
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                lines.pop()
                break
            lines.append(line)
        original_prompt = "\n".join(lines).strip()
    
    if not original_prompt:
        print("No prompt provided")
        sys.exit(1)
    
    # Show provider info if verbose
    if args.verbose:
        print(f"\n🔧 Using provider: {refiner.provider}")
    
    print("\n🔍 Analyzing prompt...")
    result = refiner.refine_prompt(original_prompt, args.template)
    
    if "error" in result:
        print(f"\n❌ Error: {result['error']}")
        if 'provider' in result:
            print(f"   Provider: {result['provider']}")
        sys.exit(1)
    
    print("\n📝 Original prompt:")
    print(f"   {original_prompt}")
    
    print("\n✨ Improved prompt:")
    print(f"   {result['improved_prompt']}")
    
    if refiner.config.refinement.output['include_explanation']:
        print("\n🔧 Changes made:")
        print(f"   {result['changes_made']}")
    
    if refiner.config.refinement.output['include_score'] and "effectiveness_score" in result:
        print("\n📊 Effectiveness score:")
        print(f"   {result['effectiveness_score']}")
    
    if result.get('from_cache'):
        print("\n💾 (Retrieved from cache)")


if __name__ == "__main__":
    main()
