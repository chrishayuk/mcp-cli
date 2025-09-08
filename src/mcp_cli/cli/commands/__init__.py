# mcp_cli/cli/commands/__init__.py
def register_all_commands() -> None:
    """
    Legacy function for backward compatibility.
    The actual commands are now registered via the unified system.
    """
    # This function is kept for backward compatibility
    # All actual command registration happens through the unified system
    # in mcp_cli.commands
    pass
