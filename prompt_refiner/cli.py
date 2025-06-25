"""Command-line interface for prompt-refiner."""

from typing import Optional, Annotated

import typer

from prompt_refiner.refinement import PromptRefiner
from prompt_refiner.ui import UI


# Initialize Typer app
app = typer.Typer(
    help="Automatically improve prompts using LLMs",
    rich_markup_mode="rich"
)


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
        raise typer.Exit(1)
    
    # Validate provider
    if provider and provider not in ['auto', 'claude', 'ollama']:
        ui.print("[red]Error: Invalid provider. Choose from: auto, claude, ollama[/red]")
        raise typer.Exit(1)
    
    # Initialize refiner
    try:
        refiner = PromptRefiner(
            config_path=config,
            no_cache=no_cache,
            provider=provider
        )
    except Exception as e:
        ui.show_initialization_error(str(e))
        raise typer.Exit(1)
    
    # Handle cache clearing if requested
    if clear_cache:
        count = refiner.clear_cache()
        ui.show_cache_cleared(count)
        if not prompt:
            # If no prompt provided and only clearing cache, exit
            raise typer.Exit(0)
    
    # Get prompt
    if prompt:
        original_prompt = prompt
    else:
        # Interactive mode
        try:
            original_prompt = ui.prompt_for_input()
        except KeyboardInterrupt:
            raise typer.Exit(0)
    
    if not original_prompt:
        ui.print("[red]Error: No prompt provided[/red]")
        raise typer.Exit(1)
    
    # Show provider info if verbose
    if verbose:
        ui.show_config(refiner.provider, template, not no_cache)
    
    ui.print()
    
    # Refine the prompt with progress indicator
    with ui.show_progress("Analyzing prompt..."):
        result = refiner.refine_prompt(original_prompt, template)
    
    if "error" in result:
        ui.show_refinement_error(result['error'], result.get('provider'))
        raise typer.Exit(1)
    
    # Display results
    ui.show_results(
        original=original_prompt,
        improved=result['improved_prompt'],
        changes=result['changes_made'],
        score=result.get('effectiveness_score'),
        from_cache=result.get('from_cache', False),
        show_explanation=refiner.config.refinement.output['include_explanation'],
        show_score=refiner.config.refinement.output['include_score']
    )


if __name__ == "__main__":
    app()