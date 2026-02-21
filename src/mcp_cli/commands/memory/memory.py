# src/mcp_cli/commands/memory/memory.py
"""
Unified memory command — visualize AI Virtual Memory state.
"""

from __future__ import annotations

import json
from typing import Any

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandMode,
    CommandParameter,
    CommandResult,
)
from chuk_term.ui import output, format_table


class MemoryCommand(UnifiedCommand):
    """View AI virtual memory state."""

    @property
    def name(self) -> str:
        return "memory"

    @property
    def aliases(self) -> list[str]:
        return ["vm", "mem"]

    @property
    def description(self) -> str:
        return "View AI virtual memory state"

    @property
    def help_text(self) -> str:
        return """
View AI virtual memory state (requires --vm flag).

Usage:
  /memory             - Summary dashboard (mode, pages, utilization, metrics)
  /memory pages       - Table of all memory pages
  /memory page <id>   - Detailed view of a specific page
  /memory stats       - Full debug dump of all VM subsystem stats

Aliases: /vm, /mem

Examples:
  /vm                 - Quick overview of VM state
  /vm pages           - See all pages with tier/type/tokens
  /vm page msg_abc123 - Inspect a specific page
  /vm stats           - Full diagnostic dump
"""

    @property
    def modes(self) -> CommandMode:
        return CommandMode.CHAT

    @property
    def parameters(self) -> list[CommandParameter]:
        return [
            CommandParameter(
                name="action",
                type=str,
                required=False,
                help="Subcommand: pages, page <id>, stats",
            ),
        ]

    async def execute(self, **kwargs: Any) -> CommandResult:
        """Execute the memory command."""
        chat_context = kwargs.get("chat_context")
        if not chat_context:
            return CommandResult(
                success=False,
                error="Memory command requires chat context.",
            )

        # Check VM is enabled
        session = getattr(chat_context, "session", None)
        vm = getattr(session, "vm", None) if session else None
        if not vm:
            return CommandResult(
                success=False,
                error="VM not enabled. Start with --vm flag.",
            )

        # Parse action from args
        action = self._parse_action(kwargs)

        if action == "pages":
            return self._show_pages(vm)
        elif action is not None and action.startswith("page "):
            page_id = action[5:].strip()
            return self._show_page_detail(vm, page_id)
        elif action == "stats":
            return self._show_full_stats(vm)
        else:
            return self._show_summary(vm, chat_context)

    # ------------------------------------------------------------------
    # Arg parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_action(kwargs: dict[str, Any]) -> str | None:
        """Extract the action string from kwargs/args."""
        action = kwargs.get("action")
        if action is not None:
            return action

        args_val = kwargs.get("args")
        if isinstance(args_val, list) and args_val:
            return " ".join(str(a) for a in args_val)
        if isinstance(args_val, str) and args_val:
            return args_val
        return None

    # ------------------------------------------------------------------
    # Subcommands
    # ------------------------------------------------------------------

    def _show_summary(self, vm: Any, chat_context: Any) -> CommandResult:
        """Show the summary dashboard."""
        metrics = vm.metrics
        ws_stats = vm.working_set.get_stats()
        pt_stats = vm.page_table.get_stats()

        # Budget from chat_context
        budget = getattr(chat_context, "_vm_budget", "?")

        # Build tier distribution string
        tier_parts = []
        for tier_name in ("L0", "L1", "L2", "L3", "L4"):
            from chuk_ai_session_manager.memory.models import StorageTier
            tier = StorageTier(tier_name)
            count = pt_stats.pages_by_tier.get(tier, 0)
            if count > 0:
                tier_parts.append(f"{tier_name}: {count}")
        tier_str = ", ".join(tier_parts) if tier_parts else "none"

        # TLB hit rate
        tlb_total = metrics.tlb_hits + metrics.tlb_misses
        tlb_rate = f"{metrics.tlb_hit_rate:.0%}" if tlb_total > 0 else "n/a"

        # Utilization bar
        util_pct = ws_stats.utilization
        bar_len = 20
        filled = int(util_pct * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)

        lines = [
            f"Mode: {vm.mode.value}   Turn: {vm.turn}   Budget: {budget:,} tokens",
            "",
            f"Working Set  [{bar}] {util_pct:.0%}",
            f"  L0 pages: {ws_stats.l0_pages}   L1 pages: {ws_stats.l1_pages}",
            f"  Tokens used: {ws_stats.tokens_used:,}   Available: {ws_stats.tokens_available:,}",
            "",
            f"Page Table   Total: {pt_stats.total_pages}   Dirty: {pt_stats.dirty_pages}",
            f"  By tier: {tier_str}",
            "",
            f"Metrics",
            f"  Faults: {metrics.faults_total} total, {metrics.faults_this_turn} this turn",
            f"  Evictions: {metrics.evictions_total} total, {metrics.evictions_this_turn} this turn",
            f"  TLB: {metrics.tlb_hits} hits, {metrics.tlb_misses} misses ({tlb_rate})",
        ]

        output.panel(
            "\n".join(lines),
            title="AI Virtual Memory",
            style="cyan",
        )
        return CommandResult(success=True)

    def _show_pages(self, vm: Any) -> CommandResult:
        """Show table of all pages."""
        entries = vm.page_table.entries
        if not entries:
            output.info("No pages in page table.")
            return CommandResult(success=True)

        # Build table rows sorted by tier then importance
        rows = []
        for page_id, entry in entries.items():
            rows.append({
                "Page ID": page_id,
                "Type": entry.page_type.value,
                "Tier": entry.tier.value,
                "Tokens": str(entry.size_tokens or "?"),
                "Importance": f"{entry.eviction_priority:.1f}",
                "Pinned": "Y" if entry.pinned else "",
                "Compression": entry.compression_level.name.lower(),
                "Accesses": str(entry.access_count),
            })

        # Sort: L0 first, then L1, etc., then by eviction_priority ascending
        tier_order = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4}
        rows.sort(key=lambda r: (tier_order.get(r["Tier"], 9), float(r["Importance"])))

        output.rule("[bold]Memory Pages[/bold]", style="primary")
        table = format_table(
            rows,
            title=None,
            columns=["Page ID", "Type", "Tier", "Tokens", "Importance", "Pinned", "Compression", "Accesses"],
        )
        output.print_table(table)
        output.print()
        output.tip("Use: /memory page <id> to see full page content")

        return CommandResult(success=True, data=rows)

    def _show_page_detail(self, vm: Any, page_id: str) -> CommandResult:
        """Show detailed view of a single page."""
        # Look up entry
        entry = vm.page_table.entries.get(page_id)
        if not entry:
            return CommandResult(
                success=False,
                error=f"Page not found: {page_id}",
            )

        # Look up content from page store
        page = vm._page_store.get(page_id)
        content = page.content if page else "[not in memory]"

        # Truncate very long content
        max_preview = 2000
        if isinstance(content, str) and len(content) > max_preview:
            content = content[:max_preview] + f"\n\n... ({len(content) - max_preview} more chars)"

        # Build detail view
        lines = [
            f"Page ID:     {entry.page_id}",
            f"Type:        {entry.page_type.value}",
            f"Tier:        {entry.tier.value}",
            f"Tokens:      {entry.size_tokens or '?'}",
            f"Compression: {entry.compression_level.name}",
            f"Pinned:      {entry.pinned}",
            f"Dirty:       {entry.dirty}",
            f"Accesses:    {entry.access_count}",
            f"Last access: {entry.last_accessed}",
            f"Modality:    {entry.modality.value}",
        ]
        if entry.provenance:
            lines.append(f"Provenance:  {', '.join(entry.provenance)}")

        lines.append("")
        lines.append("--- Content ---")
        lines.append(str(content))

        output.panel(
            "\n".join(lines),
            title=f"Page: {page_id}",
            style="cyan",
        )
        return CommandResult(success=True)

    def _show_full_stats(self, vm: Any) -> CommandResult:
        """Show full debug dump of all subsystem stats."""
        stats = vm.get_stats()

        # Format as indented JSON for readability
        formatted = json.dumps(stats, indent=2, default=str)

        output.panel(
            formatted,
            title="VM Full Stats Dump",
            style="cyan",
        )
        return CommandResult(success=True, data=stats)
