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

import sys
import json
import yaml
import subprocess
import os
from pathlib import Path
from typing import Dict, Optional, Any, FrozenSet, Tuple, Annotated
from dataclasses import dataclass, field
from types import MappingProxyType
import hashlib
import time
from datetime import datetime, timedelta
import shutil
import urllib.request
import urllib.error

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.prompt import Prompt
from rich.text import Text
from rich import box


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


# Initialize Typer app and Rich console
app = typer.Typer(
    help="Automatically improve prompts using LLMs",
    rich_markup_mode="rich"
)
console = Console()


@app.command()
def main(
    prompt: Annotated[Optional[str], typer.Argument(help="The prompt to refine")] = None,
    config: Annotated[Optional[str], typer.Option("--config", help="Path to configuration file")] = None,
    template: Annotated[str, typer.Option(
        "--template", 
        help="Template to use for refinement"
    )] = "default",
    verbose: Annotated[bool, typer.Option(
        "--verbose", "-v",
        help="Show detailed output"
    )] = False,
    no_cache: Annotated[bool, typer.Option(
        "--no-cache",
        help="Disable cache for this run"
    )] = False,
    provider: Annotated[Optional[str], typer.Option(
        "--provider",
        help="Choose provider explicitly (auto, claude, ollama)"
    )] = None,
    clear_cache: Annotated[bool, typer.Option(
        "--clear-cache",
        help="Clear all cache files before running"
    )] = False
):
    """Refine prompts using Claude or Ollama for better clarity and effectiveness."""
    
    # Validate template
    valid_templates = ['default', 'coding', 'analysis', 'writing']
    if template not in valid_templates:
        console.print(f"[red]Error: Invalid template '{template}'. Choose from: {', '.join(valid_templates)}[/red]")
        raise typer.Exit(1)
    
    # Validate provider
    if provider and provider not in ['auto', 'claude', 'ollama']:
        console.print("[red]Error: Invalid provider. Choose from: auto, claude, ollama[/red]")
        raise typer.Exit(1)
    
    # Initialize refiner
    try:
        refiner = PromptRefiner(
            config_path=config,
            no_cache=no_cache,
            provider=provider
        )
    except Exception as e:
        error_panel = Panel(
            f"[bold red]Error initializing:[/bold red]\n{str(e)}",
            title="[red]Initialization Error[/red]",
            border_style="red",
            box=box.ROUNDED
        )
        console.print(error_panel)
        raise typer.Exit(1)
    
    # Handle cache clearing if requested
    if clear_cache:
        count = refiner.clear_cache()
        console.print(f"\n[green]✓[/green] Cleared [bold cyan]{count}[/bold cyan] cache file(s)")
        if not prompt:
            # If no prompt provided and only clearing cache, exit
            raise typer.Exit(0)
    
    # Get prompt
    if prompt:
        original_prompt = prompt
    else:
        # Interactive mode
        console.print("\n[bold cyan]Interactive Mode[/bold cyan]")
        console.print("[dim]Enter your prompt (press Enter twice to finish):[/dim]\n")
        lines = []
        while True:
            try:
                line = Prompt.ask("", default="", show_default=False)
                if line == "" and lines and lines[-1] == "":
                    lines.pop()
                    break
                lines.append(line)
            except KeyboardInterrupt:
                console.print("\n[yellow]Cancelled[/yellow]")
                raise typer.Exit(0)
        original_prompt = "\n".join(lines).strip()
    
    if not original_prompt:
        console.print("[red]Error: No prompt provided[/red]")
        raise typer.Exit(1)
    
    # Show provider info if verbose
    if verbose:
        provider_info = Panel(
            f"[bold]Provider:[/bold] [cyan]{refiner.provider}[/cyan]\n"
            f"[bold]Template:[/bold] [cyan]{template}[/cyan]\n"
            f"[bold]Cache:[/bold] [cyan]{'Disabled' if no_cache else 'Enabled'}[/cyan]",
            title="[bold]Configuration[/bold]",
            box=box.ROUNDED
        )
        console.print(provider_info)
    
    console.print()
    
    # Refine the prompt with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Analyzing prompt...[/bold cyan]"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Analyzing", total=None)
        result = refiner.refine_prompt(original_prompt, template)
        progress.update(task, completed=True)
    
    if "error" in result:
        error_msg = f"[bold red]Error:[/bold red] {result['error']}"
        if 'provider' in result:
            error_msg += f"\n[bold]Provider:[/bold] {result['provider']}"
        
        error_panel = Panel(
            error_msg,
            title="[red]Refinement Error[/red]",
            border_style="red",
            box=box.ROUNDED
        )
        console.print(error_panel)
        raise typer.Exit(1)
    
    # Display results
    console.print()
    
    # Original prompt
    original_panel = Panel(
        Text(original_prompt, style="dim"),
        title="[bold]Original Prompt[/bold]",
        border_style="blue",
        box=box.ROUNDED
    )
    console.print(original_panel)
    
    # Improved prompt
    improved_panel = Panel(
        Text(result['improved_prompt'], style="green"),
        title="[bold green]✨ Improved Prompt[/bold green]",
        border_style="green",
        box=box.ROUNDED
    )
    console.print(improved_panel)
    
    if refiner.config.refinement.output['include_explanation']:
        changes_panel = Panel(
            Text(result['changes_made']),
            title="[bold yellow]Changes Made[/bold yellow]",
            border_style="yellow",
            box=box.ROUNDED
        )
        console.print(changes_panel)
    
    if refiner.config.refinement.output['include_score'] and "effectiveness_score" in result:
        score_panel = Panel(
            Text(result['effectiveness_score']),
            title="[bold magenta]Effectiveness Score[/bold magenta]",
            border_style="magenta",
            box=box.ROUNDED
        )
        console.print(score_panel)
    
    if result.get('from_cache'):
        console.print("\n[dim cyan]💾 Retrieved from cache[/dim cyan]")


if __name__ == "__main__":
    app()
