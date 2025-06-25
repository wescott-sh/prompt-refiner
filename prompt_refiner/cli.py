"""Command-line interface for prompt-refiner."""

from typing import Annotated, Dict, List, Optional

import typer

from prompt_refiner.config import Config
from prompt_refiner.refinement import PromptRefiner
from prompt_refiner.ui import UI

# Initialize Typer app
app = typer.Typer(
    help="Automatically improve prompts using LLMs",
    rich_markup_mode="rich"
)


def refine_prompt(
    prompt: str,
    config: Config,
    focus_areas: Optional[List[str]] = None,
    template: str = "default"
) -> Dict[str, str]:
    """Refine a prompt programmatically (for testing)."""
    refiner = PromptRefiner(no_cache=not config.advanced.cache.enabled)
    refiner.config = config
    
    # Get the provider
    from prompt_refiner.providers.auto import AutoProvider
    from prompt_refiner.providers.claude import ClaudeProvider
    from prompt_refiner.providers.ollama import OllamaProvider
    
    if config.provider.type == "auto":
        provider = AutoProvider(config)
    elif config.provider.type == "claude":
        provider = ClaudeProvider(config)
    elif config.provider.type == "ollama":
        provider = OllamaProvider(config)
    else:
        raise ValueError(f"Unknown provider type: {config.provider.type}")
    
    # Refine the prompt
    return provider.refine_prompt(prompt, focus_areas, template)


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

    # Initialize UI
    ui = UI()

    # Validate template
    valid_templates = ['default', 'coding', 'analysis', 'writing']
    if template not in valid_templates:
        ui.print(f"[red]Error: Invalid template '{template}'. Choose from: {', '.join(valid_templates)}[/red]")
        raise typer.Exit(code=1)

    try:
        # Initialize refinement engine
        refiner = PromptRefiner(
            config_path=config,
            no_cache=no_cache,
            provider=provider
        )

        # Clear cache if requested
        if clear_cache:
            ui.clear_cache(refiner.config)
            if prompt is None:  # If just clearing cache
                return

        # Get prompt from argument or interactive input
        if prompt is None:
            prompt = ui.get_prompt()
            if not prompt:
                raise typer.Exit(code=0)

        # Show status
        ui.show_status("🔄 Refining your prompt...")

        # Refine the prompt
        result = refiner.refine(prompt, template=template)

        # Display result
        ui.display_result(result, template, verbose)

    except KeyboardInterrupt:
        ui.print("\n[yellow]Refinement cancelled.[/yellow]")
        raise typer.Exit(code=0)
    except Exception as e:
        ui.print(f"[red]Error: {str(e)}[/red]")
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()