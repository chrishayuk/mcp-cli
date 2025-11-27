"""Playbook integration prompt template."""

from typing import Any
from mcp_cli.chat.prompt_templates.base import PromptTemplate
from mcp_cli.utils.preferences import get_preference_manager


class PlaybookPromptTemplate(PromptTemplate):
    """Template for playbook integration guidance."""

    @property
    def name(self) -> str:
        return "playbook"

    @property
    def enabled(self) -> bool:
        """Only enabled if playbook integration is enabled in preferences."""
        pref_manager = get_preference_manager()
        return pref_manager.is_playbook_enabled()

    def render(self, context: dict[str, Any] | None = None) -> str:
        """Render playbook prompt template."""
        return """
**PLAYBOOK INTEGRATION - CRITICAL:**

You have access to a playbook repository containing step-by-step procedures for tasks.

**MANDATORY FIRST STEP - Always query the playbook for:**
1. ANY question that might involve using tools or require a procedure
2. "How do I..." or "How to..." questions
3. Multi-step procedures requiring coordination of multiple tools
4. Safety assessments or condition checking
5. Information retrieval tasks (time, weather, data queries, etc.)
6. Complex workflows with established best practices

**When in doubt, query the playbook. It's fast and provides tested procedures.**

**Required workflow:**
1. FIRST: Call `query_playbook` with the user's question
2. Review the returned playbook for guidance
3. THEN execute the playbook's recommended steps using available tools
4. If no playbook exists, proceed with your best judgment

**Available playbook tools:**
- `query_playbook(question, top_k=3)` - Search for relevant playbooks
- `list_playbooks()` - Browse all available playbooks
- `get_playbook(title)` - Retrieve specific playbook by exact title

**Critical:** Playbooks contain tested, reliable procedures. Always check for relevant playbooks BEFORE attempting any task that might use tools.
"""
