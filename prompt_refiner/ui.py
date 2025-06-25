"""UI module for prompt-refiner using Rich for terminal output."""

from typing import Optional, ContextManager
from contextlib import contextmanager

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.text import Text
from rich import box


class UI:
    """Handles all terminal UI output using Rich."""
    
    def __init__(self):
        """Initialize UI with a Rich console."""
        self.console = Console()
    
    def show_error(self, error: str, provider: Optional[str] = None) -> None:
        """Display an error message in a styled panel.
        
        Args:
            error: The error message to display
            provider: Optional provider name to include in the error
        """
        error_msg = f"[bold red]Error:[/bold red] {error}"
        if provider:
            error_msg += f"\n[bold]Provider:[/bold] {provider}"
        
        error_panel = Panel(
            error_msg,
            title="[red]Error[/red]",
            border_style="red",
            box=box.ROUNDED
        )
        self.console.print(error_panel)
    
    def show_config(self, provider: str, template: str, cache_enabled: bool) -> None:
        """Display configuration information.
        
        Args:
            provider: The LLM provider being used
            template: The template being used
            cache_enabled: Whether caching is enabled
        """
        provider_info = Panel(
            f"[bold]Provider:[/bold] [cyan]{provider}[/cyan]\n"
            f"[bold]Template:[/bold] [cyan]{template}[/cyan]\n"
            f"[bold]Cache:[/bold] [cyan]{'Enabled' if cache_enabled else 'Disabled'}[/cyan]",
            title="[bold]Configuration[/bold]",
            box=box.ROUNDED
        )
        self.console.print(provider_info)
    
    @contextmanager
    def show_progress(self, message: str) -> ContextManager[Progress]:
        """Display a progress spinner with a message.
        
        Args:
            message: The message to display while processing
            
        Yields:
            Progress: A Rich Progress instance
        """
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold cyan]{message}[/bold cyan]"),
            console=self.console,
            transient=True
        ) as progress:
            task = progress.add_task(message, total=None)
            yield progress
            progress.update(task, completed=True)
    
    def show_results(
        self, 
        original: str, 
        improved: str, 
        changes: str, 
        score: Optional[str], 
        from_cache: bool,
        show_explanation: bool,
        show_score: bool
    ) -> None:
        """Display the refinement results.
        
        Args:
            original: The original prompt
            improved: The improved prompt
            changes: Explanation of changes made
            score: Effectiveness score (optional)
            from_cache: Whether the result was from cache
            show_explanation: Whether to show the explanation
            show_score: Whether to show the score
        """
        self.console.print()
        
        # Original prompt
        original_panel = Panel(
            Text(original, style="dim"),
            title="[bold]Original Prompt[/bold]",
            border_style="blue",
            box=box.ROUNDED
        )
        self.console.print(original_panel)
        
        # Improved prompt
        improved_panel = Panel(
            Text(improved, style="green"),
            title="[bold green]✨ Improved Prompt[/bold green]",
            border_style="green",
            box=box.ROUNDED
        )
        self.console.print(improved_panel)
        
        # Changes explanation
        if show_explanation:
            changes_panel = Panel(
                Text(changes),
                title="[bold yellow]Changes Made[/bold yellow]",
                border_style="yellow",
                box=box.ROUNDED
            )
            self.console.print(changes_panel)
        
        # Effectiveness score
        if show_score and score:
            score_panel = Panel(
                Text(score),
                title="[bold magenta]Effectiveness Score[/bold magenta]",
                border_style="magenta",
                box=box.ROUNDED
            )
            self.console.print(score_panel)
        
        # Cache indicator
        if from_cache:
            self.console.print("\n[dim cyan]💾 Retrieved from cache[/dim cyan]")
    
    def prompt_for_input(self) -> str:
        """Prompt the user for input in interactive mode.
        
        Returns:
            The user's input prompt
        """
        self.console.print("\n[bold cyan]Interactive Mode[/bold cyan]")
        self.console.print("[dim]Enter your prompt (press Enter twice to finish):[/dim]\n")
        
        lines = []
        while True:
            try:
                line = Prompt.ask("", default="", show_default=False)
                if line == "" and lines and lines[-1] == "":
                    lines.pop()
                    break
                lines.append(line)
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Cancelled[/yellow]")
                raise
        
        return "\n".join(lines).strip()
    
    def show_cache_cleared(self, count: int) -> None:
        """Display cache clearing message.
        
        Args:
            count: Number of cache files cleared
        """
        self.console.print(f"\n[green]✓[/green] Cleared [bold cyan]{count}[/bold cyan] cache file(s)")
    
    def show_initialization_error(self, error: str) -> None:
        """Display initialization error in a styled panel.
        
        Args:
            error: The error message to display
        """
        error_panel = Panel(
            f"[bold red]Error initializing:[/bold red]\n{error}",
            title="[red]Initialization Error[/red]",
            border_style="red",
            box=box.ROUNDED
        )
        self.console.print(error_panel)
    
    def show_refinement_error(self, error: str, provider: Optional[str] = None) -> None:
        """Display refinement error in a styled panel.
        
        Args:
            error: The error message to display
            provider: Optional provider name to include
        """
        error_msg = f"[bold red]Error:[/bold red] {error}"
        if provider:
            error_msg += f"\n[bold]Provider:[/bold] {provider}"
        
        error_panel = Panel(
            error_msg,
            title="[red]Refinement Error[/red]",
            border_style="red",
            box=box.ROUNDED
        )
        self.console.print(error_panel)
    
    def print(self, *args, **kwargs) -> None:
        """Direct access to console.print for simple output.
        
        Args:
            *args: Arguments to pass to console.print
            **kwargs: Keyword arguments to pass to console.print
        """
        self.console.print(*args, **kwargs)