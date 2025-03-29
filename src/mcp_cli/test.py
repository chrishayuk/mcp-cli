# src/cli/test.py
import anyio
import logging
import sys

# mcp imports
from chuk_mcp.mcp_client.messages.initialize.send_messages import send_initialize
from chuk_mcp.mcp_client.messages.ping.send_messages import send_ping
from chuk_mcp.mcp_client.messages.tools.send_messages import send_tools_list
from chuk_mcp.mcp_client.transport.stdio.stdio_client import stdio_client

# cli imports
from mcp_cli.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)

async def main():
    """Stripped-down script to initialize the server and send a ping."""
    # Configuration values
    config_path = "server_config.json"
    server_name = "sqlite"

    # Load server configuration
    server_params = await load_config(config_path, server_name)

    # Establish stdio communication
    async with stdio_client(server_params) as (read_stream, write_stream):
        # Initialize the server
        init_result = await send_initialize(read_stream, write_stream)

        # check we got a result
        if not init_result:
            print("Server initialization failed")
            return
        
        # connected
        print(f"We're connected!!!")

        # Send a ping
        result = await send_ping(read_stream, write_stream)
        print("Ping successful" if result else "Ping failed")

        # get tools
        result = await send_tools_list(read_stream, write_stream)
        print(result)

# Run the script
if __name__ == "__main__":
    anyio.run(main)
