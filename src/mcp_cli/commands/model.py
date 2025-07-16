# mcp_cli/commands/model.py
"""
Model-management command for MCP-CLI with enhanced discovery support.

Inside chat / interactive mode
------------------------------
  /model                → show current model & provider  
  /model list           → list ALL models (static + discovered)
  /model <name>         → probe & switch model (auto-rollback on failure)
  /model refresh        → refresh discovery and show new models
"""
from __future__ import annotations
import asyncio
from typing import Any, Dict, List
from rich.table import Table
from rich.panel import Panel

# mcp cli
from mcp_cli.model_manager import ModelManager
from mcp_cli.utils.rich_helpers import get_console
from mcp_cli.utils.async_utils import run_blocking
from mcp_cli.utils.llm_probe import LLMProbe


async def check_local_ollama_models():
    """Check what models are actually running in local Ollama."""
    try:
        import httpx
        
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get("http://localhost:11434/api/tags")
            response.raise_for_status()
            data = response.json()
            
            models = [model_data["name"] for model_data in data.get("models", [])]
            return True, models
            
    except Exception:
        return False, []


# ════════════════════════════════════════════════════════════════════════
# Async implementation (core logic)
# ════════════════════════════════════════════════════════════════════════
async def model_action_async(args: List[str], *, context: Dict[str, Any]) -> None:
    console = get_console()

    # Re-use (or lazily create) a ModelManager kept in context
    model_manager: ModelManager = context.get("model_manager") or ModelManager()
    context.setdefault("model_manager", model_manager)

    provider = model_manager.get_active_provider()
    current_model = model_manager.get_active_model()

    # ── no arguments → just display current state ───────────────────────
    if not args:
        await _print_status_enhanced(console, model_manager, current_model, provider)
        return

    # ── "/model list" helper ────────────────────────────────────────────
    if args[0].lower() == "list":
        await _print_model_list_enhanced(console, model_manager, provider)
        return
    
    # ── "/model refresh" helper ──────────────────────────────────────────
    if args[0].lower() == "refresh":
        await _refresh_models(console, model_manager, provider)
        return

    # ── attempt model switch ────────────────────────────────────────────
    new_model = args[0]
    console.print(f"[dim]Probing model '{new_model}'…[/dim]")

    # Try to switch using ModelManager (which handles discovery)
    try:
        # First try direct validation (may trigger discovery)
        is_valid = model_manager.validate_model_for_provider(provider, new_model)
        
        if is_valid:
            # Test the model works
            async with LLMProbe(model_manager, suppress_logging=True) as probe:
                result = await probe.test_model(new_model)
            
            if result.success:
                # Success - commit the change
                model_manager.set_active_model(new_model)
                context["model"] = new_model
                context["client"] = result.client
                context["model_manager"] = model_manager
                console.print(f"[green]✅ Switched to model:[/green] {new_model}")
                return
            else:
                error_msg = result.error_message or "model test failed"
                console.print(f"[red]❌ Model test failed:[/red] {error_msg}")
        else:
            console.print(f"[red]❌ Model not available:[/red] {new_model}")
            
            # Show available models as suggestion
            available = model_manager.get_available_models(provider)
            if available:
                console.print(f"[yellow]💡 Available models:[/yellow] {', '.join(available[:5])}")
                if len(available) > 5:
                    console.print(f"    ... and {len(available) - 5} more")
    
    except Exception as e:
        console.print(f"[red]❌ Model switch failed:[/red] {str(e)}")
    
    console.print(f"[yellow]Keeping current model:[/yellow] {current_model}")


# ════════════════════════════════════════════════════════════════════════
# Enhanced Helper Functions
# ════════════════════════════════════════════════════════════════════════
async def _print_status_enhanced(console, model_manager: ModelManager, model: str, provider: str) -> None:
    """Enhanced status display with discovery information."""
    console.print(f"Current model: [green]{provider}/{model}[/green]")
    
    # Get model counts
    available_models = model_manager.get_available_models(provider)
    console.print(f"Available models for [cyan]{provider}[/cyan]:")
    
    if not available_models:
        console.print("  [red]No models found[/red]")
        return
    
    # Show models with current model highlighted
    for i, available_model in enumerate(available_models):
        if available_model == model:
            console.print(f"  → [bold green]{available_model}[/bold green]")
        else:
            console.print(f"     {available_model}")
        
        # Limit display to prevent spam
        if i >= 9:  # Show max 10 models
            remaining = len(available_models) - 10
            if remaining > 0:
                console.print(f"     ... and {remaining} more models")
            break
    
    # Show discovery info for Ollama
    if provider.lower() == "ollama":
        await _show_ollama_discovery_status(console, model_manager)
    
    console.print("[dim]/model <name> to switch  |  /model list for full list  |  /model refresh to discover[/dim]")


async def _print_model_list_enhanced(console, model_manager: ModelManager, provider: str) -> None:
    """Enhanced model list that shows ALL models including discovered ones."""
    
    # Get models from ModelManager (includes discovered models)
    available_models = model_manager.get_available_models(provider)
    current_model = model_manager.get_active_model()
    
    if not available_models:
        console.print(f"[red]No models found for provider '{provider}'[/red]")
        return
    
    # Create table
    table = Table(title=f"All Models for Provider '{provider}' ({len(available_models)} total)")
    table.add_column("Status", style="cyan", width=8)
    table.add_column("Model Name", style="green")
    table.add_column("Type", style="yellow", width=12)
    table.add_column("Info", style="blue")
    
    # For Ollama, get local model info
    local_models = []
    if provider.lower() == "ollama":
        ollama_running, local_model_names = await check_local_ollama_models()
        if ollama_running:
            local_models = local_model_names
    
    # Categorize models
    static_models = set()
    try:
        # Get static models from provider config
        provider_info = model_manager.get_provider_info(provider)
        static_models = set(provider_info.get("models", []))
    except Exception:
        pass
    
    # Add models to table
    for model_name in available_models:
        # Status
        if model_name == current_model:
            status = "→"
            status_style = "bold green"
        else:
            status = " "
            status_style = "dim"
        
        # Type determination
        if model_name in static_models:
            model_type = "Static"
            info = "Config"
        elif model_name in local_models:
            model_type = "Local"
            info = "Ollama"
        else:
            model_type = "Discovered"
            info = "Available"
        
        # Add special indicators
        if ":latest" in model_name:
            info += " (latest)"
        elif "embed" in model_name.lower():
            info += " (embedding)"
        
        table.add_row(
            f"[{status_style}]{status}[/{status_style}]",
            model_name,
            model_type,
            info
        )
    
    console.print(table)
    
    # Show discovery summary for Ollama
    if provider.lower() == "ollama":
        await _show_ollama_discovery_summary(console, available_models, local_models, static_models)
    
    console.print(f"\n[dim]💡 Use '/model <name>' to switch to any model[/dim]")


async def _refresh_models(console, model_manager: ModelManager, provider: str) -> None:
    """Refresh model discovery and show results."""
    
    console.print(f"🔄 Refreshing model discovery for [cyan]{provider}[/cyan]...")
    
    # Get current count
    before_models = model_manager.get_available_models(provider)
    before_count = len(before_models)
    
    try:
        # Force refresh discovery
        success = model_manager.refresh_discovery(provider)
        
        if success:
            # Get new count
            after_models = model_manager.get_available_models(provider)
            after_count = len(after_models)
            
            console.print(f"✅ Refresh completed")
            console.print(f"   Models before: {before_count}")
            console.print(f"   Models after: {after_count}")
            
            if after_count > before_count:
                new_count = after_count - before_count
                console.print(f"   🎉 Discovered {new_count} new models!")
                
                # Show new models
                new_models = [m for m in after_models if m not in before_models]
                for new_model in new_models[:5]:  # Show first 5 new models
                    console.print(f"      • {new_model}")
                if len(new_models) > 5:
                    console.print(f"      ... and {len(new_models) - 5} more")
            else:
                console.print(f"   ℹ️  No new models discovered")
        else:
            console.print(f"❌ Refresh failed")
    
    except Exception as e:
        console.print(f"❌ Refresh error: {e}")


async def _show_ollama_discovery_status(console, model_manager: ModelManager) -> None:
    """Show Ollama discovery status in status view."""
    
    try:
        # Check local Ollama
        ollama_running, local_models = await check_local_ollama_models()
        
        if ollama_running:
            available_models = model_manager.get_available_models("ollama")
            discovery_status = model_manager.get_discovery_status()
            
            console.print(f"\n[dim]Ollama: {len(local_models)} local, {len(available_models)} accessible"
                         f" | Discovery: {'✅' if discovery_status.get('ollama_enabled') else '❌'}[/dim]")
        else:
            console.print(f"\n[dim]Ollama: Not running | Use 'ollama serve' to start[/dim]")
    
    except Exception:
        pass


async def _show_ollama_discovery_summary(console, available_models: List[str], local_models: List[str], static_models: set) -> None:
    """Show Ollama discovery summary in list view."""
    
    if not local_models:
        return
    
    # Calculate stats
    local_set = set(local_models)
    available_set = set(available_models)
    static_count = len(static_models)
    local_count = len(local_models)
    accessible_count = len(available_models)
    
    # Models that are local but not accessible
    missing_count = len(local_set - available_set)
    
    # Create summary panel
    summary_lines = [
        f"📊 Ollama Summary:",
        f"   • Static models: {static_count}",
        f"   • Local models: {local_count}",
        f"   • Accessible: {accessible_count}",
    ]
    
    if missing_count > 0:
        summary_lines.append(f"   • Not accessible: {missing_count}")
        summary_lines.append(f"   💡 Use '/model refresh' to discover more models")
    else:
        summary_lines.append(f"   ✅ All local models are accessible!")
    
    console.print(Panel("\n".join(summary_lines), title="Discovery Status", border_style="blue"))


# ════════════════════════════════════════════════════════════════════════
# Sync wrapper for non-async code-paths
# ════════════════════════════════════════════════════════════════════════
def model_action(args: List[str], *, context: Dict[str, Any]) -> None:
    """Thin synchronous facade around *model_action_async*."""
    run_blocking(model_action_async(args, context=context))