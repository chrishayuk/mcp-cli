# tests/commands/test_enums.py
"""Tests for commands/enums.py - Enum definitions."""

from mcp_cli.commands.enums import (
    CommandAction,
    ErrorMessages,
    OutputFormat,
    ProviderCommand,
    ServerCommand,
    SpecialValues,
    SuccessMessages,
    TokenNamespace,
    TokenSource,
    ToolCommand,
    TransportType,
)


class TestCommandAction:
    """Test CommandAction enum."""

    def test_command_action_values(self):
        """Test all CommandAction values exist."""
        assert CommandAction.LIST.value == "list"
        assert CommandAction.ADD.value == "add"
        assert CommandAction.REMOVE.value == "remove"
        assert CommandAction.SET.value == "set"
        assert CommandAction.GET.value == "get"
        assert CommandAction.DELETE.value == "delete"
        assert CommandAction.CREATE.value == "create"
        assert CommandAction.UPDATE.value == "update"
        assert CommandAction.SHOW.value == "show"
        assert CommandAction.ENABLE.value == "enable"
        assert CommandAction.DISABLE.value == "disable"
        assert CommandAction.VALIDATE.value == "validate"
        assert CommandAction.STATUS.value == "status"
        assert CommandAction.DETAILS.value == "details"
        assert CommandAction.REFRESH.value == "refresh"
        assert CommandAction.CONFIG.value == "config"
        assert CommandAction.DIAGNOSTIC.value == "diagnostic"

    def test_command_action_is_str(self):
        """Test CommandAction is a string enum."""
        assert isinstance(CommandAction.LIST, str)
        assert CommandAction.LIST == "list"


class TestTokenNamespace:
    """Test TokenNamespace enum."""

    def test_token_namespace_values(self):
        """Test all TokenNamespace values exist."""
        assert TokenNamespace.GENERIC.value == "generic"
        assert TokenNamespace.PROVIDER.value == "provider"
        assert TokenNamespace.BEARER.value == "bearer"
        assert TokenNamespace.API_KEY.value == "api-key"
        assert TokenNamespace.OAUTH.value == "oauth"

    def test_token_namespace_is_str(self):
        """Test TokenNamespace is a string enum."""
        assert isinstance(TokenNamespace.GENERIC, str)


class TestTransportType:
    """Test TransportType enum."""

    def test_transport_type_values(self):
        """Test all TransportType values exist."""
        assert TransportType.STDIO.value == "stdio"
        assert TransportType.SSE.value == "sse"
        assert TransportType.HTTP.value == "http"


class TestOutputFormat:
    """Test OutputFormat enum."""

    def test_output_format_values(self):
        """Test all OutputFormat values exist."""
        assert OutputFormat.JSON.value == "json"
        assert OutputFormat.TABLE.value == "table"
        assert OutputFormat.TEXT.value == "text"


class TestProviderCommand:
    """Test ProviderCommand enum."""

    def test_provider_command_values(self):
        """Test all ProviderCommand values exist."""
        assert ProviderCommand.LIST.value == "list"
        assert ProviderCommand.ADD.value == "add"
        assert ProviderCommand.REMOVE.value == "remove"
        assert ProviderCommand.SET.value == "set"
        assert ProviderCommand.CONFIG.value == "config"
        assert ProviderCommand.DIAGNOSTIC.value == "diagnostic"
        assert ProviderCommand.CUSTOM.value == "custom"


class TestTokenSource:
    """Test TokenSource enum."""

    def test_token_source_values(self):
        """Test all TokenSource values exist."""
        assert TokenSource.ENV.value == "env"
        assert TokenSource.STORAGE.value == "storage"
        assert TokenSource.NONE.value == "none"


class TestServerCommand:
    """Test ServerCommand enum."""

    def test_server_command_values(self):
        """Test all ServerCommand values exist."""
        assert ServerCommand.LIST.value == "list"
        assert ServerCommand.ADD.value == "add"
        assert ServerCommand.REMOVE.value == "remove"
        assert ServerCommand.ENABLE.value == "enable"
        assert ServerCommand.DISABLE.value == "disable"
        assert ServerCommand.STATUS.value == "status"


class TestToolCommand:
    """Test ToolCommand enum."""

    def test_tool_command_values(self):
        """Test all ToolCommand values exist."""
        assert ToolCommand.LIST.value == "list"
        assert ToolCommand.DETAILS.value == "details"
        assert ToolCommand.VALIDATE.value == "validate"
        assert ToolCommand.STATUS.value == "status"
        assert ToolCommand.ENABLE.value == "enable"
        assert ToolCommand.DISABLE.value == "disable"
        assert ToolCommand.LIST_DISABLED.value == "list-disabled"
        assert ToolCommand.AUTO_FIX.value == "auto-fix"
        assert ToolCommand.CLEAR_VALIDATION.value == "clear-validation"
        assert ToolCommand.VALIDATION_ERRORS.value == "validation-errors"


class TestSpecialValues:
    """Test SpecialValues constants."""

    def test_special_values(self):
        """Test all SpecialValues constants exist."""
        assert SpecialValues.STDIN == "-"
        assert SpecialValues.STDOUT == "-"
        assert SpecialValues.OLLAMA == "ollama"
        assert SpecialValues.USER == "User"


class TestErrorMessages:
    """Test ErrorMessages constants."""

    def test_error_messages(self):
        """Test all ErrorMessages constants exist."""
        assert ErrorMessages.NO_CONTEXT == "Context not initialized"
        assert ErrorMessages.NO_TOOL_MANAGER == "Tool manager not available"
        assert ErrorMessages.NO_MODEL_MANAGER == "Model manager not available"
        assert ErrorMessages.NO_SERVER_MANAGER == "Server manager not available"
        assert ErrorMessages.INVALID_PROVIDER == "Unknown provider"
        assert ErrorMessages.INVALID_SERVER == "Unknown server"
        assert ErrorMessages.INVALID_TOOL == "Unknown tool"
        assert ErrorMessages.NO_OPERATION == "No operation specified"


class TestSuccessMessages:
    """Test SuccessMessages constants."""

    def test_success_messages(self):
        """Test all SuccessMessages constants exist."""
        assert SuccessMessages.TOKEN_STORED == "Token stored successfully"
        assert SuccessMessages.TOKEN_DELETED == "Token deleted successfully"
        assert SuccessMessages.SERVER_ADDED == "Server added successfully"
        assert SuccessMessages.SERVER_REMOVED == "Server removed successfully"
        assert SuccessMessages.PROVIDER_SWITCHED == "Provider switched successfully"
