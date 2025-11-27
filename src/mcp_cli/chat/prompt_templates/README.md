# Composable Prompt Templates

A modular system for building system prompts from reusable, conditional templates.

## Overview

Instead of hardcoding system prompts, we use a template-based approach where different sections can be:
- **Conditionally enabled** based on preferences/config
- **Reused** across different contexts
- **Easily modified** without touching core logic
- **Tested** independently

## Architecture

### Base Template (`base.py`)

```python
class PromptTemplate(ABC):
    @abstractmethod
    def render(self, context: dict | None = None) -> str:
        """Render the template with optional context."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Template name for identification."""
        pass

    @property
    def enabled(self) -> bool:
        """Whether this template should be included."""
        return True
```

### Composer (`composer.py`)

```python
composer = PromptComposer()
composer.add_template(PlaybookPromptTemplate())
composer.add_template(GeneralGuidelinesTemplate())

prompt = composer.compose()  # Only includes enabled templates
```

## Built-in Templates

### 1. PlaybookPromptTemplate

- **Location**: `playbook.py`
- **Purpose**: Guides LLM to use playbook repository
- **Conditional**: Only enabled when `playbook.enabled = True` in preferences
- **Content**: Instructions for querying and following playbooks

### 2. GeneralGuidelinesTemplate

- **Location**: `general.py`
- **Purpose**: General LLM best practices
- **Conditional**: Always enabled
- **Content**: Step-by-step reasoning, tool usage, communication guidelines

## Creating a New Template

```python
from mcp_cli.chat.prompt_templates.base import PromptTemplate

class MyCustomTemplate(PromptTemplate):
    @property
    def name(self) -> str:
        return "my_custom_template"

    @property
    def enabled(self) -> bool:
        # Add conditional logic
        return some_condition()

    def render(self, context: dict | None = None) -> str:
        return \"\"\"
        **MY CUSTOM SECTION:**

        Custom guidance here...
        \"\"\"
```

Then add to composer in `system_prompt.py`:

```python
composer.add_template(MyCustomTemplate())
```

## Usage in system_prompt.py

```python
def generate_system_prompt(tools):
    # Base tool prompt
    prompt_generator = SystemPromptGenerator()
    base_prompt = prompt_generator.generate_prompt({"tools": tools})

    # Compose additional sections
    composer = PromptComposer()
    composer.add_template(PlaybookPromptTemplate())
    composer.add_template(GeneralGuidelinesTemplate())
    # composer.add_template(YourCustomTemplate())

    additional_sections = composer.compose()

    return base_prompt + additional_sections
```

## Benefits

1. **Modularity**: Each template is self-contained
2. **Conditional Logic**: Templates can enable/disable themselves
3. **Reusability**: Templates can be reused across different contexts
4. **Testability**: Each template can be tested independently
5. **Maintainability**: Easy to add/remove/modify sections
6. **Context-Aware**: Templates can access context dictionary

## Example: Context-Aware Template

```python
class ServerSpecificTemplate(PromptTemplate):
    def render(self, context: dict | None = None) -> str:
        if not context or 'servers' not in context:
            return ""

        servers = context['servers']
        if 'database' in servers:
            return \"\"\"
            **DATABASE GUIDELINES:**
            You have access to database servers.
            Always check schema before queries.
            \"\"\"
        return ""
```

## Testing

```python
# Test individual template
template = PlaybookPromptTemplate()
assert template.enabled == True  # or False based on preferences
output = template.render()
assert "PLAYBOOK" in output

# Test composition
composer = PromptComposer()
composer.add_template(PlaybookPromptTemplate())
prompt = composer.compose()
assert len(prompt) > 0
```

## Future Templates

Ideas for additional templates:
- **CodebaseTemplate**: When working with code repositories
- **SecurityTemplate**: Security-focused guidelines
- **DataAnalysisTemplate**: Data science workflows
- **APITemplate**: API interaction best practices
- **DebuggingTemplate**: Debug-specific guidance
