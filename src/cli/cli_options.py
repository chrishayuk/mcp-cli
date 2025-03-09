# src/cli/cli_options.py
import os
import json

def process_options(server, disable_filesystem, provider, model):
    """
    Process CLI options to produce a list of server names and set environment variables.
    
    Returns a tuple of:
      - servers_list: The final list of server names.
      - user_specified: The list of servers specified by the user.
    """
    servers_list = []
    user_specified = []
    
    if server:
        # Allow comma-separated servers.
        user_specified = [s.strip() for s in server.split(",")]
        servers_list.extend(user_specified)
    
    # Always add 'filesystem' unless explicitly disabled.
    #if not disable_filesystem and "filesystem" not in servers_list:
    #    servers_list.insert(0, "filesystem")
        
    # Use a default model if none is provided.
    if not model:
        model = "gpt-4o-mini" if provider.lower() == "openai" else "qwen2.5-coder"
    
    # Set environment variables used by the MCP code.
    os.environ["LLM_PROVIDER"] = provider
    os.environ["LLM_MODEL"] = model
    if not disable_filesystem:
        os.environ["SOURCE_FILESYSTEMS"] = json.dumps([os.getcwd()])
    
    return servers_list, user_specified
