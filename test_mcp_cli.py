#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, 'src')

os.environ['MCP_TOOL_TIMEOUT'] = '300'

import subprocess
result = subprocess.run([
    '/Users/chrishay/chris-source/mcp-cli/.venv/bin/python3', '-m', 'mcp_cli', 'chat', 
    '--config', 'test_config.json'
], env=os.environ)
