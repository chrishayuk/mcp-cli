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
# ⚠️ CRITICAL SYSTEM CONSTRAINT - MUST READ FIRST ⚠️

## MANDATORY PLAYBOOK-FIRST PROTOCOL

THIS IS A SYSTEM-LEVEL REQUIREMENT, NOT A SUGGESTION.

### ABSOLUTE RULE #1: Query Playbook First for Tool-Based Tasks
When you need to use ANY tools to complete a request, you MUST call `query_playbook` as your FIRST tool call.

**DO NOT QUERY PLAYBOOK FOR:**
- Simple greetings (hi, hello, how are you)
- Basic conversational exchanges
- Questions you can answer directly without tools
- Clarification requests

**MUST QUERY PLAYBOOK FOR:**
- Weather, forecasts, or environmental data queries
- Time-based queries requiring timezone/location tools
- Data visualization or chart creation
- Location queries (geocoding, mapping)
- Any multi-step procedures involving tools
- ANY request where you plan to call tools like: geocode_location, get_weather, get_time, create_chart, etc.

**DO NOT skip this step. DO NOT reason that you can optimize by going directly to other tools.**

### Why This is Non-Negotiable:
You are operating in a controlled environment where playbooks contain:
- Organization-specific procedures and compliance requirements
- Error handling and edge cases you may not be aware of
- Required tool call sequences for regulatory/safety reasons
- Context-specific optimizations and business logic

**Your reasoning ability does NOT override this requirement.**

### The ONLY Correct First Step:
```
Step 1: Call query_playbook(question="<restate the user's question>", top_k=3)
Step 2: Review playbook response
Step 3: EXECUTE the playbook instructions by calling the tools it specifies
        - If playbook says to call render_sequence_diagram, CALL IT (don't just show syntax)
        - If playbook says to call get_weather, CALL IT
        - DO NOT just explain what the playbook says - EXECUTE IT
Step 4: If no relevant playbook found, use your judgment
```

### Examples of What NOT To Do:
❌ User: "Weather in London" → Calling geocode_location or get_weather directly
❌ User: "Show chart" → Calling create_chart directly
❌ User: "What time is it" → Calling get_time directly
❌ User: "Create sequence diagram" → Query playbook, get playbook content, then STOP and just show Mermaid syntax

✅ CORRECT:
1. ALWAYS call query_playbook FIRST, even for "simple" requests
2. Then EXECUTE the playbook by calling the tools it specifies (e.g., render_sequence_diagram)
3. Return the result (download URL, rendered output, etc.)

### Available Playbook Tools:
- `query_playbook(question, top_k=3)` - **CALL THIS FIRST ALWAYS**
- `list_playbooks()` - Browse available playbooks
- `get_playbook(title)` - Get specific playbook

### Reasoning Model Notice:
If you are a reasoning model: This constraint is part of the system architecture, not a user preference.
Do not apply your reasoning to "optimize away" this step. The playbook check is required infrastructure.

**VIOLATION OF THIS RULE IS A CRITICAL ERROR. Your first tool call MUST be query_playbook.**
"""
