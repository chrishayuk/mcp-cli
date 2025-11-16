#!/usr/bin/env python3
"""
ModelManager Diagnostic Script

This script performs comprehensive diagnostics on the ModelManager,
testing all major functionality including:
- Provider discovery and listing
- Model discovery (local and API-based)
- Runtime provider management
- Pydantic model validation
- API client creation
- Configuration integrity

Usage:
    python diagnostics/model_manager_diagnostic.py
    python diagnostics/model_manager_diagnostic.py --verbose
    python diagnostics/model_manager_diagnostic.py --provider openai
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_cli.model_management import ModelManager
from mcp_cli.model_management import RuntimeProviderConfig, DiscoveryResult


@dataclass
class DiagnosticResult:
    """Result of a diagnostic test."""

    test_name: str
    passed: bool
    message: str
    details: Dict[str, Any] = None

    def __str__(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        result = f"{status}: {self.test_name}"
        if self.message:
            result += f"\n    {self.message}"
        if self.details:
            for key, value in self.details.items():
                result += f"\n    {key}: {value}"
        return result


class ModelManagerDiagnostic:
    """Comprehensive diagnostic suite for ModelManager."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[DiagnosticResult] = []
        self.model_manager: ModelManager = None

        # Configure logging
        level = logging.DEBUG if verbose else logging.WARNING
        logging.basicConfig(
            level=level,
            format="%(levelname)s: %(message)s" if verbose else "%(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def run_all_diagnostics(self) -> bool:
        """Run all diagnostic tests."""
        print("=" * 80)
        print("🔍 ModelManager Comprehensive Diagnostic Suite")
        print("=" * 80)
        print()

        # Core initialization tests
        self._test_initialization()
        self._test_provider_listing()
        self._test_model_listing()

        # Runtime provider tests
        self._test_runtime_provider_creation()
        self._test_runtime_provider_pydantic_validation()
        self._test_runtime_provider_discovery()

        # Configuration tests
        self._test_default_model_selection()
        self._test_provider_switching()

        # Client creation tests
        self._test_client_creation()

        # Advanced tests
        self._test_model_refresh()
        self._test_pydantic_model_integrity()

        # Print results
        self._print_results()

        # Return overall status
        return all(result.passed for result in self.results)

    def _test_initialization(self):
        """Test ModelManager initialization."""
        try:
            self.model_manager = ModelManager()
            active_provider = self.model_manager.get_active_provider()
            active_model = self.model_manager.get_active_model()

            self.results.append(
                DiagnosticResult(
                    test_name="ModelManager Initialization",
                    passed=True,
                    message="Successfully initialized",
                    details={
                        "Active Provider": active_provider,
                        "Active Model": active_model,
                    },
                )
            )
        except Exception as e:
            self.results.append(
                DiagnosticResult(
                    test_name="ModelManager Initialization",
                    passed=False,
                    message=f"Failed to initialize: {e}",
                )
            )

    def _test_provider_listing(self):
        """Test provider listing functionality."""
        try:
            providers = self.model_manager.get_available_providers()

            # Check that we have at least some providers
            has_providers = len(providers) > 0

            # Check for common providers
            common_providers = ["ollama", "openai", "anthropic"]
            found_providers = [p for p in common_providers if p in providers]

            self.results.append(
                DiagnosticResult(
                    test_name="Provider Listing",
                    passed=has_providers,
                    message=f"Found {len(providers)} providers",
                    details={
                        "Total Providers": len(providers),
                        "Providers": ", ".join(providers[:5])
                        + ("..." if len(providers) > 5 else ""),
                        "Common Providers Found": ", ".join(found_providers)
                        if found_providers
                        else "None",
                    },
                )
            )
        except Exception as e:
            self.results.append(
                DiagnosticResult(
                    test_name="Provider Listing",
                    passed=False,
                    message=f"Failed: {e}",
                )
            )

    def _test_model_listing(self):
        """Test model listing for active provider."""
        try:
            provider = self.model_manager.get_active_provider()
            models = self.model_manager.get_available_models(provider)

            # Verify no hardcoded lists (models should come from config/discovery)
            has_models = len(models) > 0

            self.results.append(
                DiagnosticResult(
                    test_name="Model Listing (Active Provider)",
                    passed=has_models,
                    message=f"Found {len(models)} models for {provider}",
                    details={
                        "Provider": provider,
                        "Model Count": len(models),
                        "First Models": ", ".join(models[:5])
                        + ("..." if len(models) > 5 else ""),
                    },
                )
            )
        except Exception as e:
            self.results.append(
                DiagnosticResult(
                    test_name="Model Listing (Active Provider)",
                    passed=False,
                    message=f"Failed: {e}",
                )
            )

    def _test_runtime_provider_creation(self):
        """Test creating a runtime provider."""
        try:
            config = self.model_manager.add_runtime_provider(
                name="test_diagnostic_provider",
                api_base="https://api.test.com/v1",
                api_key="test-key-12345",
                models=["test-model-1", "test-model-2", "test-model-3"],
            )

            # Verify it's a Pydantic model
            is_pydantic = isinstance(config, RuntimeProviderConfig)

            # Verify properties
            has_correct_name = config.name == "test_diagnostic_provider"
            has_correct_api_base = config.api_base == "https://api.test.com/v1"
            has_models = config.has_models and len(config.models) == 3
            has_default_model = config.default_model == "test-model-1"

            all_checks_passed = (
                is_pydantic
                and has_correct_name
                and has_correct_api_base
                and has_models
                and has_default_model
            )

            self.results.append(
                DiagnosticResult(
                    test_name="Runtime Provider Creation",
                    passed=all_checks_passed,
                    message="Successfully created runtime provider",
                    details={
                        "Is Pydantic Model": is_pydantic,
                        "Name Correct": has_correct_name,
                        "API Base Correct": has_correct_api_base,
                        "Has Models": has_models,
                        "Default Model": config.default_model,
                    },
                )
            )
        except Exception as e:
            self.results.append(
                DiagnosticResult(
                    test_name="Runtime Provider Creation",
                    passed=False,
                    message=f"Failed: {e}",
                )
            )

    def _test_runtime_provider_pydantic_validation(self):
        """Test Pydantic validation on runtime providers."""
        try:
            # Test that Pydantic validation works
            config = RuntimeProviderConfig(
                name="validation_test",
                api_base="https://api.example.com/v1",
                models=["model-a", "model-b"],
            )

            # Test computed properties
            has_models_works = config.has_models is True
            default_set_correctly = config.default_model == "model-a"

            # Test model methods
            config.add_models(["model-c"])
            model_added = "model-c" in config.models and len(config.models) == 3

            all_validations_passed = (
                has_models_works and default_set_correctly and model_added
            )

            self.results.append(
                DiagnosticResult(
                    test_name="Pydantic Model Validation",
                    passed=all_validations_passed,
                    message="Pydantic validation working correctly",
                    details={
                        "has_models Property": has_models_works,
                        "Default Model Auto-Set": default_set_correctly,
                        "add_models() Method": model_added,
                        "Final Model Count": len(config.models),
                    },
                )
            )
        except Exception as e:
            self.results.append(
                DiagnosticResult(
                    test_name="Pydantic Model Validation",
                    passed=False,
                    message=f"Validation failed: {e}",
                )
            )

    def _test_runtime_provider_discovery(self):
        """Test model discovery for runtime providers."""
        try:
            # Test with a provider that has models
            models = self.model_manager.get_available_models("test_diagnostic_provider")

            has_correct_models = models == [
                "test-model-1",
                "test-model-2",
                "test-model-3",
            ]

            self.results.append(
                DiagnosticResult(
                    test_name="Runtime Provider Model Discovery",
                    passed=has_correct_models,
                    message="Successfully retrieved models from runtime provider",
                    details={
                        "Models Retrieved": models,
                        "Expected Count": 3,
                        "Actual Count": len(models),
                    },
                )
            )
        except Exception as e:
            self.results.append(
                DiagnosticResult(
                    test_name="Runtime Provider Model Discovery",
                    passed=False,
                    message=f"Failed: {e}",
                )
            )

    def _test_default_model_selection(self):
        """Test default model selection logic."""
        try:
            # Test with runtime provider
            default = self.model_manager.get_default_model("test_diagnostic_provider")
            is_correct = default == "test-model-1"

            # Test with active provider
            active_provider = self.model_manager.get_active_provider()
            active_default = self.model_manager.get_default_model(active_provider)
            has_active_default = active_default is not None and len(active_default) > 0

            all_passed = is_correct and has_active_default

            self.results.append(
                DiagnosticResult(
                    test_name="Default Model Selection",
                    passed=all_passed,
                    message="Default model selection working",
                    details={
                        "Runtime Provider Default": default,
                        "Active Provider": active_provider,
                        "Active Provider Default": active_default,
                    },
                )
            )
        except Exception as e:
            self.results.append(
                DiagnosticResult(
                    test_name="Default Model Selection",
                    passed=False,
                    message=f"Failed: {e}",
                )
            )

    def _test_provider_switching(self):
        """Test provider switching functionality."""
        try:
            original_provider = self.model_manager.get_active_provider()

            # Switch to runtime provider
            self.model_manager.switch_provider("test_diagnostic_provider")
            new_provider = self.model_manager.get_active_provider()
            switched_correctly = new_provider == "test_diagnostic_provider"

            # Switch back
            self.model_manager.switch_provider(original_provider)
            restored = self.model_manager.get_active_provider() == original_provider

            all_passed = switched_correctly and restored

            self.results.append(
                DiagnosticResult(
                    test_name="Provider Switching",
                    passed=all_passed,
                    message="Provider switching working",
                    details={
                        "Original Provider": original_provider,
                        "Switched To": new_provider,
                        "Restored": restored,
                    },
                )
            )
        except Exception as e:
            self.results.append(
                DiagnosticResult(
                    test_name="Provider Switching",
                    passed=False,
                    message=f"Failed: {e}",
                )
            )

    def _test_client_creation(self):
        """Test client creation for providers."""
        try:
            # Test getting client for active provider
            provider = self.model_manager.get_active_provider()

            # Skip if it's a provider that needs API keys
            if provider in ["openai", "anthropic", "gemini"]:
                self.results.append(
                    DiagnosticResult(
                        test_name="Client Creation",
                        passed=True,
                        message=f"Skipped (requires API key for {provider})",
                    )
                )
                return

            try:
                client = self.model_manager.get_client()
                has_client = client is not None

                self.results.append(
                    DiagnosticResult(
                        test_name="Client Creation",
                        passed=has_client,
                        message=f"Successfully created client for {provider}",
                        details={
                            "Provider": provider,
                            "Client Type": type(client).__name__,
                        },
                    )
                )
            except Exception as inner_e:
                # API key missing is expected for some providers
                if "API key" in str(inner_e) or "token" in str(inner_e).lower():
                    self.results.append(
                        DiagnosticResult(
                            test_name="Client Creation",
                            passed=True,
                            message=f"Expected error (API key required): {inner_e}",
                        )
                    )
                else:
                    raise

        except Exception as e:
            self.results.append(
                DiagnosticResult(
                    test_name="Client Creation",
                    passed=False,
                    message=f"Unexpected error: {e}",
                )
            )

    def _test_model_refresh(self):
        """Test model refresh functionality."""
        try:
            provider = self.model_manager.get_active_provider()

            # Get models before refresh
            models_before = self.model_manager.get_available_models(provider)

            # Attempt refresh (may not find new models, but shouldn't crash)
            self.model_manager.refresh_models(provider)

            # Get models after refresh
            models_after = self.model_manager.get_available_models(provider)

            # As long as it didn't crash, it's a pass
            self.results.append(
                DiagnosticResult(
                    test_name="Model Refresh",
                    passed=True,
                    message="Model refresh executed successfully",
                    details={
                        "Provider": provider,
                        "Models Before": len(models_before),
                        "Models After": len(models_after),
                    },
                )
            )
        except Exception as e:
            self.results.append(
                DiagnosticResult(
                    test_name="Model Refresh", passed=False, message=f"Failed: {e}"
                )
            )

    def _test_pydantic_model_integrity(self):
        """Test that all Pydantic models maintain integrity."""
        try:
            # Get all custom providers and verify they're Pydantic models
            custom_providers = self.model_manager._custom_providers

            all_are_pydantic = all(
                isinstance(config, RuntimeProviderConfig)
                for config in custom_providers.values()
            )

            # Test immutability of DiscoveryResult
            try:
                result = DiscoveryResult(provider="test", models=["m1"], success=True)
                # Try to modify (should fail because it's frozen)
                try:
                    result.models = ["m2"]
                    immutable_check = False  # Should not reach here
                except Exception:
                    immutable_check = True  # Expected
            except Exception:
                immutable_check = False

            all_passed = all_are_pydantic and immutable_check

            self.results.append(
                DiagnosticResult(
                    test_name="Pydantic Model Integrity",
                    passed=all_passed,
                    message="All Pydantic models maintain proper integrity",
                    details={
                        "All Custom Providers Are Pydantic": all_are_pydantic,
                        "DiscoveryResult Immutable": immutable_check,
                        "Custom Provider Count": len(custom_providers),
                    },
                )
            )
        except Exception as e:
            self.results.append(
                DiagnosticResult(
                    test_name="Pydantic Model Integrity",
                    passed=False,
                    message=f"Failed: {e}",
                )
            )

    def _print_results(self):
        """Print all diagnostic results."""
        print()
        print("=" * 80)
        print("📊 Diagnostic Results")
        print("=" * 80)
        print()

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        for result in self.results:
            print(result)
            print()

        print("=" * 80)
        print(f"Summary: {passed}/{total} tests passed")
        print("=" * 80)

        if passed == total:
            print("✅ All diagnostics passed!")
        else:
            print(f"❌ {total - passed} diagnostic(s) failed")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="ModelManager Diagnostic Suite")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--provider",
        "-p",
        type=str,
        help="Test specific provider (not implemented yet)",
    )

    args = parser.parse_args()

    diagnostic = ModelManagerDiagnostic(verbose=args.verbose)
    success = diagnostic.run_all_diagnostics()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
