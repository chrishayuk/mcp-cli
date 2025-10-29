#!/usr/bin/env python3
"""
MCP CLI Command Mode - Python Integration Demo

Demonstrates how to use mcp-cli cmd mode from Python scripts
for automation and data processing pipelines.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional


def run_mcp_cmd(
    tool: Optional[str] = None,
    tool_args: Optional[dict] = None,
    input_text: Optional[str] = None,
    input_file: Optional[str] = None,
    output_file: Optional[str] = None,
    prompt: Optional[str] = None,
    server: str = "sqlite",
    raw: bool = True,
) -> dict:
    """
    Execute mcp-cli cmd mode from Python.

    Args:
        tool: Tool name to execute
        tool_args: Tool arguments as dict
        input_text: Input text to process
        input_file: Input file path
        output_file: Output file path
        prompt: Prompt text
        server: MCP server to use
        raw: Use raw output mode

    Returns:
        Dict with result and metadata
    """
    cmd = ["uv", "run", "mcp-cli", "cmd", "--server", server]

    if tool:
        cmd.extend(["--tool", tool])
        if tool_args:
            cmd.extend(["--tool-args", json.dumps(tool_args)])

    if prompt:
        cmd.extend(["--prompt", prompt])

    if input_file:
        cmd.extend(["--input", input_file])
    elif input_text:
        cmd.extend(["--input", "-"])

    if output_file:
        cmd.extend(["--output", output_file])

    if raw:
        cmd.append("--raw")

    # Execute command
    try:
        if input_text:
            result = subprocess.run(
                cmd,
                input=input_text.encode(),
                capture_output=True,
                timeout=30,
            )
        else:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=30,
            )

        stdout = result.stdout.decode()
        stderr = result.stderr.decode()

        # Try to parse as JSON if raw mode
        if raw and stdout:
            try:
                data = json.loads(stdout)
            except json.JSONDecodeError:
                data = stdout
        else:
            data = stdout

        return {
            "success": result.returncode == 0,
            "data": data,
            "stdout": stdout,
            "stderr": stderr,
            "returncode": result.returncode,
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "data": None,
            "error": "Command timed out",
            "returncode": -1,
        }
    except Exception as e:
        return {
            "success": False,
            "data": None,
            "error": str(e),
            "returncode": -1,
        }


def demo_direct_tool_execution():
    """Demo 1: Execute tools directly from Python."""
    print("\n" + "=" * 70)
    print("Demo 1: Direct Tool Execution")
    print("=" * 70)

    # Execute list_tables tool
    print("\n📋 Listing tables...")
    result = run_mcp_cmd(tool="list_tables")

    if result["success"]:
        print(f"✓ Success: {result['data']}")
    else:
        print(f"✗ Error: {result.get('error', 'Tool execution failed')}")


def demo_file_processing():
    """Demo 2: File-based processing."""
    print("\n" + "=" * 70)
    print("Demo 2: File Processing")
    print("=" * 70)

    # Create input file
    input_file = Path("/tmp/mcp_python_demo_input.txt")
    output_file = Path("/tmp/mcp_python_demo_output.json")

    input_file.write_text("Sample data for processing\nLine 2\nLine 3")
    print(f"\n📄 Created input file: {input_file}")

    # Process with cmd mode
    print(f"🔄 Processing file with cmd mode...")
    result = run_mcp_cmd(
        tool="list_tables",
        output_file=str(output_file),
    )

    if result["success"] and output_file.exists():
        print(f"✓ Output saved to: {output_file}")
        print(f"   Content: {output_file.read_text()}")
    else:
        print(f"✗ Processing failed")

    # Cleanup
    input_file.unlink(missing_ok=True)
    output_file.unlink(missing_ok=True)


def demo_pipeline_processing():
    """Demo 3: Pipeline-style processing."""
    print("\n" + "=" * 70)
    print("Demo 3: Pipeline Processing")
    print("=" * 70)

    # Stage 1: Get data
    print("\n🔄 Stage 1: Getting data...")
    stage1_result = run_mcp_cmd(tool="list_tables")

    if not stage1_result["success"]:
        print("✗ Stage 1 failed")
        return

    print(f"✓ Stage 1 complete")

    # Stage 2: Process data (example - would use the output)
    print("🔄 Stage 2: Processing data...")
    print("✓ Stage 2 complete")

    print("\n✓ Pipeline complete!")


def demo_batch_processing():
    """Demo 4: Batch processing multiple items."""
    print("\n" + "=" * 70)
    print("Demo 4: Batch Processing")
    print("=" * 70)

    items = ["item1", "item2", "item3"]
    results = []

    print(f"\n🔄 Processing {len(items)} items...")

    for i, item in enumerate(items, 1):
        print(f"   [{i}/{len(items)}] Processing {item}...")
        result = run_mcp_cmd(tool="list_tables")
        results.append({
            "item": item,
            "success": result["success"],
            "data": result.get("data"),
        })

    # Summary
    successful = sum(1 for r in results if r["success"])
    print(f"\n✓ Batch complete: {successful}/{len(items)} successful")


def demo_error_handling():
    """Demo 5: Error handling in automation."""
    print("\n" + "=" * 70)
    print("Demo 5: Error Handling")
    print("=" * 70)

    print("\n🧪 Testing error handling with invalid tool...")
    result = run_mcp_cmd(tool="nonexistent_tool")

    if not result["success"]:
        print(f"✓ Error properly caught: {result.get('error', 'Unknown error')}")
        print(f"   Return code: {result['returncode']}")
    else:
        print("✗ Should have failed but didn't")


def demo_data_transformation():
    """Demo 6: Data transformation pipeline."""
    print("\n" + "=" * 70)
    print("Demo 6: Data Transformation Pipeline")
    print("=" * 70)

    print("\n🔄 Extract -> Transform -> Load pattern...")

    # Extract
    print("   1. Extract: Getting data from MCP server...")
    extract_result = run_mcp_cmd(tool="list_tables")

    if not extract_result["success"]:
        print("   ✗ Extract failed")
        return

    # Transform
    print("   2. Transform: Processing data...")
    data = extract_result["data"]
    # In a real scenario, you'd transform the data here
    transformed_data = {"processed": True, "original": data}

    # Load
    print("   3. Load: Saving results...")
    output_file = Path("/tmp/mcp_etl_result.json")
    output_file.write_text(json.dumps(transformed_data, indent=2))

    print(f"   ✓ ETL pipeline complete: {output_file}")

    # Cleanup
    output_file.unlink(missing_ok=True)


def main():
    """Run all demos."""
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "MCP CLI Command Mode - Python Integration" + " " * 12 + "║")
    print("╚" + "=" * 68 + "╝")

    demos = [
        ("Direct Tool Execution", demo_direct_tool_execution),
        ("File Processing", demo_file_processing),
        ("Pipeline Processing", demo_pipeline_processing),
        ("Batch Processing", demo_batch_processing),
        ("Error Handling", demo_error_handling),
        ("Data Transformation", demo_data_transformation),
    ]

    print("\nRunning 6 demonstrations...")
    print("Note: Some demos may show errors if MCP server is not properly configured.")

    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\n✗ Demo '{name}' failed: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary: Python Integration Patterns")
    print("=" * 70)
    print("""
Key Patterns Demonstrated:

1. Direct Tool Execution
   - Call MCP tools from Python
   - Get structured results back

2. File-Based Processing
   - Process files with cmd mode
   - Save results to disk

3. Pipeline Processing
   - Chain multiple operations
   - Pass data between stages

4. Batch Processing
   - Process multiple items
   - Collect results

5. Error Handling
   - Catch and handle errors
   - Proper return codes

6. ETL Workflows
   - Extract, Transform, Load pattern
   - Data transformation pipelines

Python Integration Benefits:
  ✓ Type safety with Python
  ✓ Easy error handling
  ✓ Integration with other libraries
  ✓ Complex control flow
  ✓ Data transformation with pandas, etc.
  ✓ Async processing with asyncio
  ✓ Parallel processing with multiprocessing

Example Usage:
  # In your Python script
  from cmd_mode_python_demo import run_mcp_cmd

  result = run_mcp_cmd(
      tool="list_tables",
      server="sqlite",
      raw=True
  )

  if result["success"]:
      print(f"Tables: {result['data']}")
""")

    print("\n✓ Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
