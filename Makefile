# Makefile mixing uv for install/test and PYTHON3 for build/publish
PROJECT_NAME := mcp-cli
DEFAULT_GOAL := help

# Let users override the Python interpreter for build/publish (e.g. make build PYTHON3=python3.12)
PYTHON3 ?= python3

.PHONY: help
help:
	@echo "Makefile for $(PROJECT_NAME) - mixing uv for install/test, $(PYTHON3) for build/publish"
	@echo
	@echo "Targets:"
	@echo "  install       Install package in uv environment (editable mode)"
	@echo "  test          Run tests with uv"
	@echo "  clean         Remove build artifacts"
	@echo

# ------------------------------------------------------------------------
# 1) Install in the uv environment
# ------------------------------------------------------------------------
.PHONY: install
install:
	@echo "Installing package in uv environment (editable mode)..."
	uv run python -m pip install --no-cache-dir -e .
	@echo "Install complete."

# ------------------------------------------------------------------------
# 2) Build with system python3 (or PYTHON3 override)
# ------------------------------------------------------------------------
# .PHONY: build
# build:
# 	@echo "Building sdist and wheel with \`$(PYTHON3)\`..."
# 	$(PYTHON3) -m build
# 	@echo "Build complete. Artifacts in ./dist"

# ------------------------------------------------------------------------
# 3) Test with uv-run
# ------------------------------------------------------------------------
.PHONY: test
test:
	@echo "Running tests with uv run pytest..."
	uv run pytest

# ------------------------------------------------------------------------
# 4) Clean build artifacts
# ------------------------------------------------------------------------
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build dist *.egg-info .pytest_cache
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name '*.pyc' -exec rm -rf {} +
	@echo "Clean complete."

# ------------------------------------------------------------------------
# 5) Publish to PyPI using \`$(PYTHON3)\`
# ------------------------------------------------------------------------
# .PHONY: publish
# publish: clean build
# 	@echo "Publishing to PyPI with \`$(PYTHON3)\` and twine..."
# 	$(PYTHON3) -m twine check dist/*
# 	$(PYTHON3) -m twine upload dist/*
# 	@echo "Publish to PyPI complete."

# ------------------------------------------------------------------------
# 6) Publish to TestPyPI using \`$(PYTHON3)\`
# ------------------------------------------------------------------------
# .PHONY: publish-test
# publish-test: clean build
# 	@echo "Publishing to TestPyPI with \`$(PYTHON3)\` and twine..."
# 	$(PYTHON3) -m twine check dist/*
# 	$(PYTHON3) -m twine upload --repository testpypi dist/*
# 	@echo "Publish to TestPyPI complete."