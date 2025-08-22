# Package Management with uv

## Overview

This project uses `uv` as the primary package manager for Python dependencies. `uv` is a fast, reliable Python package and project manager written in Rust.

## Why uv?

- **Speed**: 10-100x faster than pip and pip-tools
- **Reliability**: Built-in resolver prevents dependency conflicts
- **Simplicity**: Single tool for package and project management
- **Compatibility**: Drop-in replacement for pip commands
- **Lock files**: Automatic dependency locking for reproducible builds

## Installation

### Install uv
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Using pip (fallback)
pip install uv

# Verify installation
uv --version
```

## Basic Usage

### Installing Dependencies

```bash
# Install project with all dependencies
uv sync

# Install including dev dependencies
uv sync --dev

# Install specific package
uv add package-name

# Install dev dependency
uv add --dev pytest-cov

# Install from requirements file
uv pip install -r requirements.txt
```

### Running Commands

```bash
# Run command in project environment
uv run python script.py

# Run pytest
uv run pytest

# Run with coverage
uv run pytest --cov=src/chuk_mcp_math

# Run any command
uv run make test
```

### Managing Dependencies

```bash
# Add a new dependency
uv add numpy

# Add dev dependency
uv add --dev pytest-benchmark

# Remove dependency
uv remove package-name

# Update dependencies
uv sync --upgrade

# Show installed packages
uv pip list
```

## Project Configuration

### pyproject.toml

The project uses `pyproject.toml` for dependency management:

```toml
[project]
dependencies = [
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",
]

[tool.uv]
dev-dependencies = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=6.0.0",
    "mypy>=1.0.0",
    "ruff>=0.8.0",
]
```

### Lock File

`uv.lock` ensures reproducible installations:
- Automatically generated and updated
- Should be committed to version control
- Ensures all developers use same dependency versions

## Common Commands Reference

### Development Workflow

```bash
# Initial setup
git clone <repo>
cd chuk-mcp-math
uv sync --dev

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/chuk_mcp_math

# Run linting
uv run ruff check .

# Run type checking
uv run mypy src/

# Format code
uv run ruff format .
```

### Package Management

```bash
# Add production dependency
uv add fastapi

# Add dev dependency
uv add --dev black

# Add with version constraint
uv add "pandas>=2.0.0"

# Remove package
uv remove pandas

# Update all dependencies
uv sync --upgrade

# Update specific package
uv add --upgrade numpy
```

### Environment Management

```bash
# Create virtual environment (automatic with uv sync)
uv venv

# Activate environment (usually not needed with uv run)
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Show environment info
uv pip list
uv pip show package-name
```

## Makefile Integration

The project's Makefile is configured to use uv:

```makefile
# Install dependencies
install:
	uv sync

# Install with dev dependencies
dev-install:
	uv sync --dev

# Run tests
test:
	uv run pytest

# Run with coverage
test-cov:
	uv run pytest --cov=src/chuk_mcp_math

# Linting
lint:
	uv run ruff check .

# Type checking
typecheck:
	uv run mypy src/
```

## CI/CD Integration

### GitHub Actions

```yaml
- name: Install uv
  uses: astral-sh/setup-uv@v3
  
- name: Install dependencies
  run: uv sync --dev

- name: Run tests
  run: uv run pytest --cov=src/chuk_mcp_math
```

## Migration from pip

### For Developers

Replace pip commands with uv equivalents:

| pip command | uv command |
|------------|------------|
| `pip install package` | `uv add package` |
| `pip install -e .` | `uv sync` |
| `pip install -r requirements.txt` | `uv pip install -r requirements.txt` |
| `pip uninstall package` | `uv remove package` |
| `pip list` | `uv pip list` |
| `pip freeze` | `uv pip freeze` |
| `python script.py` | `uv run python script.py` |
| `pytest` | `uv run pytest` |

### For Scripts

Update automation scripts:

```bash
# Old (pip)
pip install -e ".[dev]"
pytest

# New (uv)
uv sync --dev
uv run pytest
```

## Troubleshooting

### Common Issues

**uv command not found**
```bash
# Reinstall uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH
export PATH="$HOME/.cargo/bin:$PATH"
```

**Dependency conflicts**
```bash
# Clear cache and reinstall
uv cache clean
rm -rf .venv uv.lock
uv sync --dev
```

**Wrong Python version**
```bash
# Specify Python version
uv venv --python 3.11
uv sync
```

**Import errors after install**
```bash
# Ensure using uv run
uv run python  # Not just 'python'
uv run pytest  # Not just 'pytest'
```

## Best Practices

### DO's
✅ Always use `uv run` to execute commands  
✅ Commit `uv.lock` to version control  
✅ Use `uv add` instead of editing pyproject.toml manually  
✅ Keep dev dependencies separate with `--dev` flag  
✅ Use `uv sync` after pulling changes  

### DON'Ts
❌ Don't use pip alongside uv in the same project  
❌ Don't manually activate venv when using `uv run`  
❌ Don't ignore uv.lock in git  
❌ Don't edit uv.lock manually  
❌ Don't mix pip and uv commands  

## Advanced Usage

### Custom Package Index
```bash
# Add custom index
uv add --index-url https://custom.pypi.org/simple/ package

# Use extra index
uv add --extra-index-url https://extra.pypi.org/simple/ package
```

### Platform-specific Dependencies
```toml
[tool.uv]
dependencies = [
    "windows-only-package ; sys_platform == 'win32'",
    "unix-only-package ; sys_platform != 'win32'",
]
```

### Workspace Management
```bash
# For monorepo with multiple packages
uv sync --workspace
```

## Related Documentation

- [uv Documentation](https://docs.astral.sh/uv/)
- [pyproject.toml Specification](https://packaging.python.org/en/latest/specifications/pyproject-toml/)
- [Python Packaging Guide](https://packaging.python.org/)
- [Project README](../README.md)

## Template Information

- **Source**: [vibe-coding-templates](https://github.com/chrishayuk/vibe-coding-templates/blob/main/python/docs/PACKAGE_MANAGEMENT.md)
- **Version**: 1.0.0
- **Date**: 2025-01-19
- **Author**: chrishayuk
- **Last Synced**: 2025-01-19