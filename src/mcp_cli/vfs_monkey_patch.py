# vfs_monkeypatch.py
import builtins
import os
import sys
import io
import logging

# Example import: your existing Python VFS
from chuk_virtual_fs import VirtualFileSystem

# Create or load your VFS instance
vfs = VirtualFileSystem(security_profile="default")

# --- 1) Patch Python builtins.open, os.remove, etc. ---

_real_open = builtins.open

def vfs_open(file, mode='r', buffering=-1, encoding=None,
             errors=None, newline=None, closefd=True, opener=None):
    logging.debug(f"[VFS Intercept] open => {file} (mode={mode})")
    
    # Example security check: if VFS denies, raise
    allowed = vfs.touch(file) if 'w' in mode else vfs.exists(file)
    if not allowed:
        raise PermissionError(f"Access denied by VFS for path: {file}")
    
    # If you want actual disk I/O after checks:
    return _real_open(file, mode, buffering, encoding, errors, newline, closefd, opener)

builtins.open = vfs_open


# --- 2) Patch sys.stdin / sys.stdout ---

# We'll create custom file-like classes that can talk to VFS or just be placeholders

class VFSStdin(io.StringIO):
    """
    A simple example that might store initial 'input' or intercept reads.
    Real usage might talk to vfs or block certain reads, etc.
    """
    def __init__(self, initial_value='', newline='\n'):
        super().__init__(initial_value, newline=newline)

    def read(self, size=-1):
        data = super().read(size)
        logging.debug(f"[VFS Intercept] stdin read => {data!r}")
        # You could consult vfs for data or do something else
        return data


class VFSStdout(io.StringIO):
    """
    A simple example capturing writes.
    Real usage might pass them to vfs or store them in a buffer, etc.
    """
    def write(self, s):
        logging.debug(f"[VFS Intercept] stdout write => {s!r}")
        # You could route this to vfs if you like:
        # vfs.write_file('/virtual/stdout.txt', s, append=True)
        return super().write(s)

# Replace sys.stdin / sys.stdout with our custom streams
sys.stdin = VFSStdin("Pretend user input here\n")
sys.stdout = VFSStdout()
