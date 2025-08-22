# Test Coverage Guide

## Overview

Test coverage measures how much of your code is executed during testing. This guide covers coverage measurement, targets, and best practices for the MCP CLI project.

## Coverage Tools

### Installation
```bash
# Install coverage tools using uv (preferred)
uv add --dev pytest-cov

# The tool is already included in pyproject.toml dev dependencies
```

### Running Coverage Reports

```bash
# Basic coverage report
uv run pytest --cov=src/mcp_cli

# Detailed terminal report with missing lines
uv run pytest --cov=src/mcp_cli --cov-report=term-missing

# Generate HTML coverage report
uv run pytest --cov=src/mcp_cli --cov-report=html

# Coverage for specific module
uv run pytest --cov=src/mcp_cli/chat tests/mcp_cli/chat/

# Fail tests if coverage drops below threshold
uv run pytest --cov=src/mcp_cli --cov-fail-under=80

# Using Makefile commands
make test          # Run tests
make test-cov      # Run tests with coverage report
```

## Coverage Targets

### Project Goals
- **Overall Coverage**: ≥ 80%
- **Core Modules**: ≥ 90%
- **New Code**: ≥ 95%
- **Critical Paths**: 100%

### Module-Specific Targets

| Module Category | Target Coverage | Priority |
|----------------|-----------------|----------|
| Core CLI Commands | 95% | Critical |
| Chat/Interactive Modes | 90% | Critical |
| Tool Management | 90% | High |
| LLM Integration | 85% | High |
| UI Components | 80% | Medium |
| Utility Functions | 75% | Medium |
| Example/Demo Code | 50% | Low |

## Understanding Coverage Reports

### Terminal Output
```
Name                                    Stmts   Miss  Cover   Missing
------------------------------------------------------------------------
src/mcp_cli/main.py                       145      8    94%   42-49
src/mcp_cli/chat/chat_handler.py          203     12    94%   89-100
src/mcp_cli/tools/tool_manager.py         178      5    97%   234-238
src/mcp_cli/ui/terminal.py                215      1    99%   487
src/mcp_cli/ui/output.py                  390    122    69%   Various
src/mcp_cli/ui/theme.py                   156      3    98%   234-236
src/mcp_cli/llm/llm_client.py            134     22    84%   67-88
------------------------------------------------------------------------
TOTAL                                     2456    412    83%
```

- **Stmts**: Total number of statements
- **Miss**: Number of statements not executed
- **Cover**: Percentage of statements covered
- **Missing**: Line numbers not covered

### HTML Reports
```bash
# Generate HTML report
uv run pytest --cov=src/mcp_cli --cov-report=html

# Using Makefile
make test-cov

# Open report (macOS)
open htmlcov/index.html

# Open report (Linux)
xdg-open htmlcov/index.html

# Report location: htmlcov/index.html
```

HTML reports provide:
- Interactive line-by-line coverage visualization
- Sortable module list
- Coverage trends over time
- Branch coverage details

## Coverage Types

### Line Coverage
Basic metric showing which lines were executed:
```python
def calculate(x, y):
    result = x + y  # ✓ Covered
    if result > 100:
        return 100  # ✗ Not covered if result ≤ 100
    return result   # ✓ Covered
```

### Branch Coverage
Ensures all code paths are tested:
```python
def process(value):
    if value > 0:      # Need tests for both True and False
        return "positive"
    elif value < 0:    # Need tests for both True and False
        return "negative"
    else:
        return "zero"
```

### Statement Coverage vs Functional Coverage
```python
# High statement coverage but poor functional coverage
async def divide(a, b):
    # Test might cover the line but miss edge cases
    return a / b  # ✓ Line covered, but did we test b=0?
```

## Best Practices

### 1. Focus on Meaningful Coverage
```python
# Good: Test actual functionality
@pytest.mark.asyncio
async def test_tool_execution():
    """Test tool execution flow, not just lines."""
    # Setup
    tool_manager = ToolManager()
    
    # Test actual tool call
    result = await tool_manager.execute_tool(
        "list_tables", 
        {"database": "test.db"}
    )
    
    # Verify behavior, not just execution
    assert result["success"] is True
    assert "tables" in result
    assert isinstance(result["tables"], list)
```

### 2. Don't Chase 100% Coverage Blindly
```python
# Not worth testing
if __name__ == "__main__":
    # Demo code - low priority for coverage
    demo()

# Platform-specific code
if sys.platform == "win32":
    # Only test on relevant platform
    windows_specific_function()
```

### 3. Prioritize Critical Paths
```python
# High priority - core CLI command handling
async def execute_command(command: str, args: dict):
    """Critical function - aim for 100% coverage."""
    # Every line and branch should be tested
    
# Lower priority - convenience wrapper
def format_output(text: str):
    """Simple formatting wrapper - basic test sufficient."""
    return f"[output] {text}"
```

### 4. Use Coverage to Find Gaps
```bash
# Identify untested modules
uv run pytest --cov=src/mcp_cli --cov-report=term-missing | grep "0%"

# Find partially tested modules
uv run pytest --cov=src/mcp_cli --cov-report=term-missing | grep -E "[0-9]{1,2}%"

# Using Makefile
make test-cov | grep "0%"  # Find untested modules
```

## Improving Coverage

### Step-by-Step Approach

1. **Measure Baseline**
   ```bash
   uv run pytest --cov=src/mcp_cli --cov-report=term > coverage_baseline.txt
   
   # Or using Makefile
   make test-cov > coverage_baseline.txt
   ```

2. **Identify Gaps**
   - Sort by coverage percentage
   - Focus on critical modules first
   - Look for easy wins (simple functions)

3. **Write Targeted Tests**
   ```python
   # Use coverage report to identify missing lines
   # Missing: lines 45-52 (error handling)
   @pytest.mark.asyncio
   async def test_error_conditions():
       """Target uncovered error paths."""
       with pytest.raises(ValueError):
           await function_that_needs_coverage(invalid_input)
   ```

4. **Verify Improvement**
   ```bash
   # Run coverage again and compare
   uv run pytest --cov=src/mcp_cli --cov-report=term
   
   # Or using Makefile
   make test-cov
   ```

## Coverage in CI/CD

### GitHub Actions
For GitHub Actions workflow configuration, see:
- **Template**: [github-actions-coverage.yaml](https://github.com/chrishayuk/vibe-coding-templates/blob/main/python/templates/cicd/workflows/github-actions-coverage.yaml)
- **Local Implementation**: [github-actions-coverage.yaml](../../templates/cicd/workflows/github-actions-coverage.yaml)

The workflow includes coverage reporting, Codecov integration, and artifact uploading.

### Pre-commit Hooks
For pre-commit hook configuration, see:
- **Template**: [pre-commit-coverage-hook.yaml](https://github.com/chrishayuk/vibe-coding-templates/blob/main/python/templates/cicd/hooks/pre-commit-coverage-hook.yaml)
- **Local Implementation**: [pre-commit-coverage-hook.yaml](../../templates/cicd/hooks/pre-commit-coverage-hook.yaml)

Quick setup:
```bash
# Install pre-commit
uv add --dev pre-commit

# Add hooks to .pre-commit-config.yaml from template

# Install hooks
pre-commit install

# Run coverage check
pre-commit run test-coverage --all-files
```

## Common Coverage Patterns

### Async Function Coverage
```python
@pytest.mark.asyncio
async def test_chat_handler():
    """Ensure async chat handling is properly covered."""
    handler = ChatHandler()
    
    # Test async message processing
    response = await handler.process_message("Hello")
    assert response is not None
    
    # Test streaming response
    async for chunk in handler.stream_response("Tell me a story"):
        assert chunk  # Verify chunks are yielded
```

### Error Path Coverage
```python
@pytest.mark.asyncio
async def test_error_paths():
    """Cover all error conditions."""
    tool_manager = ToolManager()
    
    # Invalid tool name
    with pytest.raises(ValueError, match="Unknown tool"):
        await tool_manager.execute_tool("invalid_tool", {})
    
    # Missing required arguments
    with pytest.raises(ValueError, match="Missing required"):
        await tool_manager.execute_tool("read_query", {})
    
    # Invalid server connection
    with pytest.raises(ConnectionError):
        await tool_manager.execute_tool("list_tables", {}, server="invalid")
```

### Branch Coverage
```python
@pytest.mark.parametrize("mode,expected_handler", [
    ("chat", "ChatHandler"),
    ("interactive", "InteractiveHandler"),
    ("cmd", "CommandHandler"),
    ("direct", "DirectHandler")
])
@pytest.mark.asyncio
async def test_all_modes(mode, expected_handler):
    """Ensure all CLI modes are covered."""
    from mcp_cli.main import get_handler
    
    handler = await get_handler(mode)
    assert handler.__class__.__name__ == expected_handler
```

## Troubleshooting

### Coverage Not Detected
```bash
# Ensure test discovery is working
uv run pytest --collect-only

# Check source path is correct
uv run pytest --cov=src/mcp_cli --cov-report=term

# Verify __init__.py files exist
find src -name "*.py" -type f | head
```

### Inconsistent Coverage
```bash
# Clear coverage cache
rm -rf .coverage .pytest_cache

# Run with fresh environment
uv run pytest --cov=src/mcp_cli --no-cov-on-fail
```

### Missing Async Coverage
```python
# Ensure pytest-asyncio is installed
uv add --dev pytest-asyncio

# Use proper async test marking
@pytest.mark.asyncio  # Required for async tests
async def test_async():
    result = await async_function()
```

## Coverage Badges

Add coverage badges to README:
```markdown
![Coverage](https://img.shields.io/badge/coverage-83%25-green)
![Tests](https://img.shields.io/badge/tests-156%20passed-green)
```

Or with dynamic coverage:
```markdown
[![codecov](https://codecov.io/gh/username/repo/branch/main/graph/badge.svg)](https://codecov.io/gh/username/repo)
```

## Related Documentation

- [Unit Testing](./UNIT_TESTING.md) - General unit testing practices
- [Package Management](../PACKAGE_MANAGEMENT.md) - Using uv for dependencies
- [UI Themes](../ui/themes.md) - Testing UI components across themes
- [Project README](../../README.md) - Project overview

## Template Information

- **Source**: [vibe-coding-templates](https://github.com/chrishayuk/vibe-coding-templates/blob/main/python/docs/testing/TEST_COVERAGE.md)
- **Version**: 1.0.0
- **Date**: 2025-01-19
- **Author**: chrishayuk
- **Last Synced**: 2025-01-19