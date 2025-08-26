# Unit Testing Guide for MCP CLI

## Overview

Unit testing focuses on testing individual functions and methods in isolation. This document covers principles and patterns for effective unit testing in the MCP CLI project.

## Core Principles

### Test Isolation
- Each unit test should be completely independent
- No shared state between tests
- Mock external dependencies
- Test one thing at a time

### Test Structure (AAA Pattern)
```python
@pytest.mark.asyncio
async def test_function_behavior():
    """Test specific behavior of function."""
    # Arrange - Set up test data and conditions
    input_data = prepare_test_data()
    expected_result = calculate_expected()
    
    # Act - Execute the function under test
    actual_result = await function_under_test(input_data)
    
    # Assert - Verify the result
    assert actual_result == expected_result
```

## Unit Test Organization

### File Structure
```
tests/
├── mcp_cli/
│   ├── chat/
│   │   ├── test_chat_handler.py
│   │   ├── test_chat_context.py
│   │   ├── test_tool_processor.py
│   │   └── test_ui_manager.py
│   ├── cli/
│   │   ├── test_cli_chat.py
│   │   ├── test_cli_interactive.py
│   │   └── test_cli_registry.py
│   ├── commands/
│   │   ├── test_provider.py
│   │   ├── test_tools.py
│   │   └── test_servers.py
│   ├── tools/
│   │   ├── test_tool_manager.py
│   │   └── test_adapter.py
│   ├── ui/
│   │   ├── test_theme.py      # Theme system tests (99% coverage)
│   │   ├── test_terminal.py   # Terminal management tests (99% coverage)
│   │   ├── test_output.py     # Output management tests (69% coverage)
│   │   └── test_prompts.py    # User prompts and interactions tests (82% coverage)
│   └── conftest.py           # Shared fixtures
```

### Test Class Organization
```python
class TestFunctionName:
    """Unit tests for function_name."""
    
    def test_normal_operation(self):
        """Test expected behavior with valid input."""
        pass
    
    def test_edge_cases(self):
        """Test boundary conditions."""
        pass
    
    def test_error_conditions(self):
        """Test error handling."""
        pass
    
    def test_type_validation(self):
        """Test input type handling."""
        pass
```

## Mocking Strategies

### Basic Mocking
```python
from unittest.mock import Mock, patch

def test_tool_manager_with_mock():
    """Test ToolManager with mocked MCP server."""
    # Create mock server
    mock_server = Mock()
    mock_server.list_tools.return_value = [
        {"name": "test_tool", "description": "Test tool"}
    ]
    
    # Inject mock
    tool_manager = ToolManager(server=mock_server)
    tools = tool_manager.get_available_tools()
    
    # Verify interaction
    mock_server.list_tools.assert_called_once()
    assert len(tools) == 1
    assert tools[0]["name"] == "test_tool"
```

### Patching Dependencies
```python
@patch('mcp_cli.llm.llm_client.ChukLLMClient')
def test_chat_with_patched_llm(mock_llm_class):
    """Test chat handler with patched LLM client."""
    mock_llm = Mock()
    mock_llm_class.return_value = mock_llm
    mock_llm.generate.return_value = "AI response"
    
    handler = ChatHandler()
    result = handler.process_message("Hello")
    
    assert result == "AI response"
    mock_llm.generate.assert_called_once()
```

### Async Mocking
```python
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_async_tool_execution():
    """Test async tool execution with mock."""
    mock_tool_processor = AsyncMock()
    mock_tool_processor.execute_tool.return_value = {
        "success": True,
        "result": "data"
    }
    
    chat_handler = ChatHandler(tool_processor=mock_tool_processor)
    result = await chat_handler.handle_tool_call("list_tables", {})
    
    mock_tool_processor.execute_tool.assert_awaited_once()
    assert result["success"] is True
```

## Testing Patterns

### Testing Pure Functions
```python
def test_tool_name_sanitization():
    """Test deterministic tool name sanitization."""
    from mcp_cli.tools.adapter import sanitize_tool_name
    
    # Pure function - always same output for same input
    assert sanitize_tool_name("list-tables") == "list_tables"
    assert sanitize_tool_name("list-tables") == "list_tables"  # Same result
    
    # Test properties
    assert sanitize_tool_name("") == ""  # Empty string
    assert sanitize_tool_name("valid_name") == "valid_name"  # No change needed
```

### Testing Stateful Functions
```python
class TestChatContext:
    """Test ChatContext with internal state."""
    
    def setup_method(self):
        """Reset state before each test."""
        self.context = ChatContext()
    
    def test_initial_state(self):
        """Test initial conversation state."""
        assert len(self.context.messages) == 0
        assert self.context.tool_history == []
    
    def test_message_addition(self):
        """Test adding messages to context."""
        self.context.add_message("user", "Hello")
        assert len(self.context.messages) == 1
        
        self.context.add_message("assistant", "Hi there!")
        assert len(self.context.messages) == 2
        assert self.context.messages[-1]["role"] == "assistant"
```

### Testing Error Conditions
```python
def test_command_validation():
    """Test that invalid commands raise appropriate errors."""
    from mcp_cli.commands import execute_command
    
    with pytest.raises(ValueError, match="Unknown command"):
        execute_command("invalid_command")
    
    with pytest.raises(TypeError, match="Arguments must be dict"):
        execute_command("tools", "not_a_dict")
```

### Testing Side Effects
```python
def test_conversation_save(tmp_path):
    """Test saving conversation to file."""
    from mcp_cli.chat import save_conversation
    
    output_file = tmp_path / "conversation.json"
    messages = [{"role": "user", "content": "Hello"}]
    
    save_conversation(messages, filepath=output_file)
    
    assert output_file.exists()
    import json
    saved_data = json.loads(output_file.read_text())
    assert saved_data[0]["content"] == "Hello"
```

## Parametrized Testing

### Basic Parametrization
```python
@pytest.mark.parametrize("provider,expected_client", [
    ("openai", "OpenAIClient"),
    ("anthropic", "AnthropicClient"),
    ("ollama", "OllamaClient"),
    ("groq", "GroqClient"),
])
def test_provider_selection(provider, expected_client):
    """Test LLM provider selection."""
    from mcp_cli.llm import get_llm_client
    
    client = get_llm_client(provider)
    assert client.__class__.__name__ == expected_client
```

### Complex Parametrization
```python
@pytest.mark.parametrize("command,args,expected_output", [
    ("tools", {}, "Available tools"),
    ("servers", {}, "Connected servers"),
    ("provider", {}, "Current provider"),
    ("model", {}, "Current model"),
])
def test_commands(command, args, expected_output):
    """Test various CLI commands."""
    from mcp_cli.commands import execute_command
    
    result = execute_command(command, args)
    assert expected_output in result
```

## Fixtures

### Basic Fixtures
```python
@pytest.fixture
def mock_tool_manager():
    """Provide mock ToolManager for tests."""
    manager = Mock()
    manager.list_tools.return_value = [
        {"name": "read_query", "description": "Execute SQL query"},
        {"name": "list_tables", "description": "List database tables"}
    ]
    return manager

def test_tool_listing(mock_tool_manager):
    """Test tool listing using fixture."""
    tools = mock_tool_manager.list_tools()
    assert len(tools) == 2
    assert tools[0]["name"] == "read_query"
```

### Fixture Scopes
```python
@pytest.fixture(scope="function")  # Default - per test
def chat_context():
    """Create fresh chat context for each test."""
    from mcp_cli.chat import ChatContext
    context = ChatContext()
    yield context
    context.clear()  # Cleanup

@pytest.fixture(scope="class")  # Per test class
def mock_llm_client():
    """Shared mock LLM client for test class."""
    client = Mock()
    client.generate.return_value = "Test response"
    return client

@pytest.fixture(scope="module")  # Per test module
def test_config():
    """Load test configuration once per module."""
    return {
        "provider": "ollama",
        "model": "gpt-oss",
        "test_mode": True
    }
```

## Coverage Guidelines

### What to Test
- All public functions/methods
- Complex private methods
- Error handling paths
- Edge cases and boundaries
- Different input types
- State transitions

### What Not to Test
- Simple getters/setters
- Framework code
- Third-party libraries
- Trivial functions (unless critical)
- Generated code

### Coverage Metrics

For comprehensive coverage guidance, see [Test Coverage Guide](./TEST_COVERAGE.md).

```bash
# Check coverage (using uv)
uv run pytest tests/ --cov=src/mcp_cli --cov-report=term-missing

# Enforce minimum coverage
uv run pytest tests/ --cov=src/mcp_cli --cov-fail-under=80

# Generate HTML report
uv run pytest tests/ --cov=src/mcp_cli --cov-report=html

# Using Makefile commands
make test          # Run all tests
make test-cov      # Run with coverage report
```

Target coverage levels:
- Overall: ≥ 80%
- Core modules: ≥ 90%
- New code: ≥ 95%

## Best Practices

### DO's
✅ Keep tests simple and focused  
✅ Use descriptive test names  
✅ Test behavior, not implementation  
✅ Use fixtures for common setup  
✅ Mock external dependencies  
✅ Test edge cases  
✅ Maintain test isolation  
✅ Write tests first (TDD)  

### DON'Ts
❌ Don't test multiple behaviors in one test  
❌ Don't use production data  
❌ Don't make tests dependent on order  
❌ Don't test private methods directly  
❌ Don't ignore test failures  
❌ Don't use hard-coded delays  
❌ Don't over-mock  
❌ Don't write brittle tests  

## Example: Complete Unit Test

```python
"""Unit tests for ToolManager.execute_tool function."""

import pytest
from unittest.mock import patch, AsyncMock
from mcp_cli.tools import ToolManager

class TestExecuteTool:
    """Test cases for tool execution."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.tool_manager = ToolManager()
    
    @pytest.mark.parametrize("tool_name,args,expected_key", [
        ("list_tables", {}, "tables"),
        ("read_query", {"query": "SELECT 1"}, "result"),
        ("describe_table", {"table": "users"}, "schema"),
    ])
    @pytest.mark.asyncio
    async def test_tool_execution(self, tool_name, args, expected_key):
        """Test various tool executions."""
        with patch.object(self.tool_manager, '_execute') as mock_execute:
            mock_execute.return_value = {expected_key: "data"}
            
            result = await self.tool_manager.execute_tool(tool_name, args)
            assert expected_key in result
    
    @pytest.mark.asyncio
    async def test_unknown_tool_error(self):
        """Test that unknown tool raises ValueError."""
        with pytest.raises(ValueError, match="Unknown tool"):
            await self.tool_manager.execute_tool("invalid_tool", {})
    
    @patch('mcp_cli.tools.logger')
    @pytest.mark.asyncio
    async def test_tool_logging(self, mock_logger):
        """Test that tool executions are logged."""
        with patch.object(self.tool_manager, '_execute') as mock_execute:
            mock_execute.return_value = {"success": True}
            
            await self.tool_manager.execute_tool("test_tool", {})
            mock_logger.info.assert_called()
```

## Related Documentation
- [Test Coverage Guide](./TEST_COVERAGE.md) - Coverage targets and best practices
- [Package Management](../PACKAGE_MANAGEMENT.md) - Using uv for test dependencies
- [UI Themes](../ui/themes.md) - Testing UI components across themes
- [Project README](../../README.md) - Project overview and setup