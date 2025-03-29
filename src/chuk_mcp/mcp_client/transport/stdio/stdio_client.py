# chuk_mcp/mcp_client/transport/stdio/stdio_client.py
import json
import logging
import sys
import traceback
import anyio
from anyio.streams.text import TextReceiveStream
from contextlib import asynccontextmanager

# host imports
from chuk_mcp.mcp_client.host.environment import get_default_environment

# mcp imports
from chuk_mcp.mcp_client.messages.json_rpc_message import JSONRPCMessage
from chuk_mcp.mcp_client.transport.stdio.stdio_server_parameters import StdioServerParameters

class StdioClient:
    def __init__(self, server: StdioServerParameters):
        if not server.command:
            raise ValueError("Server command must not be empty.")
        if not isinstance(server.args, (list, tuple)):
            raise ValueError("Server arguments must be a list or tuple.")
        self.server = server
        self.read_stream_writer, self.read_stream = anyio.create_memory_object_stream(0)
        self.write_stream, self.write_stream_reader = anyio.create_memory_object_stream(0)
        self.process = None
        self.tg = None

    async def _process_json_line(self, line: str):
        try:
            logging.debug(f"Processing line: {line.strip()}")
            data = json.loads(line)
            logging.debug(f"Parsed JSON data: {data}")
            message = JSONRPCMessage.model_validate(data)
            logging.debug(f"Validated JSONRPCMessage: {message}")
            await self.read_stream_writer.send(message)
        except json.JSONDecodeError as exc:
            logging.error(f"JSON decode error: {exc}. Line: {line.strip()}")
        except Exception as exc:
            logging.error(f"Error processing message: {exc}. Line: {line.strip()}")
            logging.debug(f"Traceback:\n{traceback.format_exc()}")

    async def _stdout_reader(self):
        """Read JSON-RPC messages from the server's stdout."""
        assert self.process.stdout, "Opened process is missing stdout"
        buffer = ""
        logging.debug("Starting stdout_reader")
        try:
            async with self.read_stream_writer:
                async for chunk in TextReceiveStream(self.process.stdout):
                    lines = (buffer + chunk).split("\n")
                    buffer = lines.pop()
                    for line in lines:
                        if line.strip():
                            await self._process_json_line(line)
                if buffer.strip():
                    await self._process_json_line(buffer)
        except anyio.ClosedResourceError:
            logging.debug("Read stream closed.")
        except Exception as exc:
            logging.error(f"Unexpected error in stdout_reader: {exc}")
            logging.debug(f"Traceback:\n{traceback.format_exc()}")
            raise
        finally:
            logging.debug("Exiting stdout_reader")

    async def _stdin_writer(self):
        """Send JSON-RPC messages from the write stream to the server's stdin."""
        assert self.process.stdin, "Opened process is missing stdin"
        logging.debug("Starting stdin_writer")
        try:
            async with self.write_stream_reader:
                async for message in self.write_stream_reader:
                    json_str = message.model_dump_json(exclude_none=True)
                    logging.debug(f"Sending: {json_str}")
                    await self.process.stdin.send((json_str + "\n").encode())
        except anyio.ClosedResourceError:
            logging.debug("Write stream closed.")
        except Exception as exc:
            logging.error(f"Unexpected error in stdin_writer: {exc}")
            logging.debug(f"Traceback:\n{traceback.format_exc()}")
            raise
        finally:
            logging.debug("Exiting stdin_writer")

    async def _terminate_process(self):
        """Terminate the subprocess, first gracefully then forcefully if needed."""
        try:
            if self.process.returncode is None:
                logging.debug("Terminating subprocess gracefully...")
                self.process.terminate()
                try:
                    with anyio.fail_after(5):
                        await self.process.wait()
                    logging.info("Process terminated gracefully.")
                except TimeoutError:
                    logging.warning("Graceful termination timed out. Forcefully killing process...")
                    try:
                        self.process.kill()
                        with anyio.fail_after(5):
                            await self.process.wait()
                        logging.info("Process killed successfully.")
                    except Exception as kill_exc:
                        logging.error(f"Error killing process: {kill_exc}")
            else:
                logging.info("Process already terminated.")
        except Exception as exc:
            logging.error(f"Error during process termination: {exc}")

    async def __aenter__(self):
        # Start the subprocess.
        self.process = await anyio.open_process(
            [self.server.command, *self.server.args],
            env=self.server.env or get_default_environment(),
            stderr=sys.stderr,
        )
        logging.debug(
            f"Subprocess started with PID {self.process.pid}, command: {self.server.command}"
        )
        # Create a task group for background tasks.
        self.tg = anyio.create_task_group()
        await self.tg.__aenter__()
        self.tg.start_soon(self._stdout_reader)
        self.tg.start_soon(self._stdin_writer)
        # Return the streams to the caller.
        return self.read_stream, self.write_stream

    async def __aexit__(self, exc_type, exc, tb):
        # Cancel the task group if it exists.
        if self.tg is not None:
            self.tg.cancel_scope.cancel()  # Synchronous cancellation.
            try:
                await self.tg.__aexit__(None, None, None)
            except RuntimeError as re:
                # Catch the cancel scope error if it occurs during shutdown.
                if "Attempted to exit cancel scope" in str(re):
                    logging.debug("Caught cancel scope error in __aexit__: %s", re)
                else:
                    raise
        # Terminate the subprocess.
        await self._terminate_process()
        return False


@asynccontextmanager
async def stdio_client(server: StdioServerParameters):
    """
    Async context manager for the stdio client.
    Usage:
        async with stdio_client(server_params) as (read_stream, write_stream):
            ...
    """
    client = StdioClient(server)
    try:
        streams = await client.__aenter__()
        yield streams
    except Exception:
        raise
    finally:
        await client.__aexit__(None, None, None)
