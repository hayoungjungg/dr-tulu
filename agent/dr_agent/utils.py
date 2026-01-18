"""
Utility functions for checking and launching services required by workflows.

This module provides functions to:
- Check if services are running on specific ports
- Launch MCP servers and vLLM servers in the background
- Extract port numbers from URLs
"""

import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from rich.console import Console


def check_port(port: int, timeout: float = 1.0) -> bool:
    """Check if a port is listening."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    result = sock.connect_ex(("localhost", port))
    sock.close()
    return result == 0


def extract_port_from_url(url_str: str) -> Optional[int]:
    """Extract port number from URL string."""
    # Remove the scheme (http:// or https://) to avoid false positives
    if "://" in url_str:
        url_str = url_str.split("://", 1)[1]

    url = url_str.rstrip("/")

    if ":" in url:
        port_str = url.split(":")[-1].split("/")[0]
        return int(port_str)
    return None


def launch_mcp_server(
    port: int = 8000, logger: Optional[logging.Logger] = None, log_file: Optional[Path] = None
) -> Optional[subprocess.Popen]:
    """Launch MCP server in background.
    
    Args:
        port: Port number for MCP server
        logger: Optional logger instance
        log_file: Optional log file path. If None, uses centralized mcp_actions log or /tmp fallback.
    """
    console = Console()

    if logger:
        logger.info(f"Launching MCP server on port {port}...")
    else:
        console.print(
            f"[cyan]🚀[/cyan] Launching MCP server on port [bold]{port}[/bold]..."
        )

    env = os.environ.copy()
    env["MCP_CACHE_DIR"] = (
        f".cache-{os.uname().nodename if hasattr(os, 'uname') else 'localhost'}"
    )

    # Use centralized log file if provided, otherwise try to get from root logger
    if log_file is None:
        # Try to get the centralized log file from root logger
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_file = Path(handler.baseFilename)
                break
        
        # Fallback to /tmp if no file handler found
        if log_file is None:
            log_file = Path(f"/tmp/mcp_server_{port}.log")
    
    if logger:
        logger.info(f"MCP server output will be logged to {log_file}")
    else:
        console.print(f"[dim]📋 MCP server output will be logged to {log_file}[/dim]")

    # Use append mode to add to existing log file
    with open(log_file, "a", encoding='utf-8') as f:
        # Write a separator to distinguish MCP server output
        f.write(f"\n{'='*80}\n")
        f.write(f"MCP Server Started (PID: will be set after launch) - Port: {port}\n")
        f.write(f"{'='*80}\n")
        f.flush()
        
        process = subprocess.Popen(
            [sys.executable, "-m", "dr_agent.mcp_backend.main", "--port", str(port)],
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
        
        # Write PID after process is created
        f.write(f"MCP Server PID: {process.pid}\n")
        f.flush()

    # Wait for server to start
    if logger:
        logger.info("Waiting for MCP server to start...")
    else:
        console.print("[yellow]⏳[/yellow] Waiting for MCP server to start...")

    for _ in range(20):
        time.sleep(0.5)
        if check_port(port):
            if logger:
                logger.info(f"MCP server started (PID: {process.pid})")
            else:
                console.print(
                    f"[green]✓[/green] MCP server started [dim](PID: {process.pid})[/dim]"
                )
            return process

    if process.poll() is None:
        if logger:
            logger.warning(
                "MCP server process started but port check failed. Continuing anyway..."
            )
        else:
            console.print(
                "[yellow]⚠[/yellow] MCP server process started but port check failed. Continuing anyway..."
            )
        return process
    else:
        if logger:
            logger.error(
                f"MCP server failed to start (exit code: {process.returncode}). Check logs: {log_file}"
            )
        else:
            console.print(
                f"[red]❌[/red] MCP server failed to start [dim](exit code: {process.returncode})[/dim]"
            )
            console.print(f"[dim]Check logs: {log_file}[/dim]")
        return None


def launch_vllm_server(
    model_name: str, port: int, gpu_id: int = 0, logger: Optional[logging.Logger] = None
) -> Optional[subprocess.Popen]:
    """Launch vLLM server in background."""
    console = Console()

    if logger:
        logger.info(f"Launching vLLM server for model {model_name} on port {port}...")
    else:
        console.print(
            f"[cyan]🚀[/cyan] Launching vLLM server for model [bold cyan]{model_name}[/bold cyan] on port [bold]{port}[/bold]..."
        )

    # Try to find vllm command
    import shutil

    vllm_base_cmd = None

    is_uv = (
        "uv" in sys.executable.lower()
        or os.environ.get("UV_PROJECT_ENVIRONMENT")
        or os.environ.get("VIRTUAL_ENV", "").endswith(".venv")
    )

    if shutil.which("vllm"):
        vllm_base_cmd = ["vllm", "serve"]
    elif is_uv and shutil.which("uv"):
        vllm_base_cmd = ["uv", "run", "vllm", "serve"]
    elif sys.executable:
        vllm_base_cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]

    if not vllm_base_cmd:
        if logger:
            logger.error(
                "vllm command not found. Install vllm with: uv pip install -e '.[vllm]' or uv pip install 'dr_agent[vllm]'"
            )
        else:
            console.print(
                "[red]❌[/red] Error: vllm command not found. Tried: vllm, uv run vllm, python -m vllm.entrypoints.openai.api_server"
            )
            console.print(
                "[blue]💡[/blue] Install vllm with: [dim]uv pip install -e '.[vllm]'[/dim] or [dim]uv pip install 'dr_agent[vllm]'[/dim]"
            )
        return None

    cmd = vllm_base_cmd + [
        model_name,
        "--port",
        str(port),
        "--dtype",
        "auto",
        "--max-model-len",
        "40960",
        "--gpu-memory-utilization",
        "0.80",  # Use 80% instead of default 90% to avoid OOM errors (reduced from 0.85)
    ]

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Use centralized log file if available, otherwise fallback to /tmp
    log_file = None
    if log_file is None:
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_file = Path(handler.baseFilename)
                break
    
    # Fallback to /tmp if no centralized log found
    if log_file is None:
        log_file = Path(f"/tmp/vllm_server_{port}.log")
    
    if logger:
        logger.info(f"vLLM output for {model_name} will be logged to {log_file}")
        logger.info(
            "Waiting for vLLM server to become ready (this may take a few minutes for large models)..."
        )
    else:
        console.print(
            f"[dim]📋 vLLM output for {model_name} will be logged to {log_file}[/dim]"
        )
        console.print(
            "[yellow]⏳[/yellow] Waiting for vLLM server to become ready [dim](this may take a few minutes for large models)...[/dim]"
        )

    # Use append mode to add to centralized log file
    with open(log_file, "a", encoding='utf-8') as f:
        # Write a separator to distinguish vLLM server output
        f.write(f"\n{'='*80}\n")
        f.write(f"vLLM Server Started (PID: will be set after launch) - Model: {model_name}, Port: {port}, GPU: {gpu_id}\n")
        f.write(f"{'='*80}\n")
        f.flush()
        
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
        
        # Write PID after process is created
        f.write(f"vLLM Server PID: {process.pid}\n")
        f.flush()

    start_time = time.time()
    while time.time() - start_time < 300:
        # Check for errors in log file before checking port
        has_error = False
        error_lines = []
        try:
            with open(log_file, "r", encoding='utf-8') as f:
                lines = f.readlines()
                # Check for common error patterns
                for line in lines[-50:]:  # Check last 50 lines
                    line_lower = line.lower()
                    if any(keyword in line_lower for keyword in [
                        "error", "failed", "exception", "traceback", 
                        "valueerror", "runtimeerror", "enginecore failed",
                        "free memory on device", "less than desired"
                    ]):
                        has_error = True
                        if "error" in line_lower or "failed" in line_lower:
                            error_lines.append(line.strip())
        except:
            pass
        
        if has_error and error_lines:
            # Server has errors - don't wait for port, fail immediately
            if logger:
                logger.error(
                    f"vLLM server failed to start - errors detected in logs. Check logs: {log_file}"
                )
                logger.error("Detected errors:")
                for err_line in error_lines[-5:]:  # Show last 5 error lines
                    logger.error(f"  {err_line}")
            else:
                console.print(
                    f"[red]❌[/red] vLLM server failed to start - errors detected in logs [dim](PID: {process.pid})[/dim]"
                )
                console.print(f"[dim]Check logs: {log_file}[/dim]")
                console.print("[dim]Detected errors:[/dim]")
                for err_line in error_lines[-5:]:
                    console.print(f"[dim]  {err_line[:100]}...[/dim]")
            return None
        
        if check_port(port):
            # Port is open, but verify server is actually ready by checking for successful initialization
            try:
                with open(log_file, "r", encoding='utf-8') as f:
                    lines = f.readlines()
                    # Check if server actually initialized successfully
                    has_success = any("application startup complete" in line.lower() or 
                                    "uvicorn running" in line.lower() or
                                    "api server version" in line.lower() 
                                    for line in lines[-20:])
                    if has_success and not has_error:
                        if logger:
                            logger.info(f"vLLM server started successfully (PID: {process.pid})")
                        else:
                            console.print(
                                f"[green]✓[/green] vLLM server started successfully [dim](PID: {process.pid})[/dim]"
                            )
                        return process
            except:
                pass
            # If we can't verify, still return but log warning
            if logger:
                logger.warning(f"vLLM server port is open but initialization status unclear (PID: {process.pid})")
            else:
                console.print(
                    f"[yellow]⚠[/yellow] vLLM server port is open but initialization status unclear [dim](PID: {process.pid})[/dim]"
                )
            return process

        if process.poll() is not None:
            if logger:
                logger.error(
                    f"vLLM server failed to start (exit code: {process.returncode}). Check logs: {log_file}"
                )
            else:
                console.print(
                    f"[red]❌[/red] vLLM server failed to start [dim](exit code: {process.returncode})[/dim]"
                )
                console.print(f"[dim]Check logs: {log_file}[/dim]")
            # Print last few lines of log
            try:
                with open(log_file, "r", encoding='utf-8') as f:
                    lines = f.readlines()
                    if logger:
                        logger.error("Last 15 lines of log:")
                        for line in lines[-15:]:
                            logger.error(line.rstrip())
                    else:
                        console.print("[dim]Last 15 lines of log:[/dim]")
                        console.print("".join(lines[-15:]))
            except:
                pass
            return None

        time.sleep(2)

        elapsed = int(time.time() - start_time)
        if elapsed > 0 and elapsed % 30 == 0:
            if logger:
                logger.info(f"Still waiting for vLLM server ({elapsed}s)...")
                # Show last meaningful log line
                try:
                    with open(log_file, "r", encoding='utf-8') as f:
                        lines = f.readlines()
                        last_lines = [l for l in lines[-5:] if l.strip()]
                        if last_lines:
                            logger.info(f"Latest log: {last_lines[-1].strip()[:100]}")
                except:
                    pass
            else:
                console.print(
                    f"[yellow]⏳[/yellow] Still waiting for vLLM server [dim]({elapsed}s)...[/dim]"
                )
                # Show last meaningful log line
                try:
                    with open(log_file, "r", encoding='utf-8') as f:
                        lines = f.readlines()
                        last_lines = [l for l in lines[-5:] if l.strip()]
                        if last_lines:
                            console.print(f"[dim]   Latest: {last_lines[-1].strip()[:80]}...[/dim]")
                except:
                    pass

    if process.poll() is None:
        if logger:
            logger.warning(
                "vLLM server process started but port check timed out after 10 minutes. It may still be initializing..."
            )
        else:
            console.print(
                "[yellow]⚠[/yellow] vLLM server process started but port check timed out after 10 minutes. It may still be initializing..."
            )
            console.print(f"[dim]Check logs: {log_file}[/dim]")
        return process
    else:
        if logger:
            logger.error(
                f"vLLM server failed to start (exit code: {process.returncode}). Check logs: {log_file}"
            )
        else:
            console.print(
                f"[red]❌[/red] vLLM server failed to start [dim](exit code: {process.returncode})[/dim]"
            )
            console.print(f"[dim]Check logs: {log_file}[/dim]")
        return None
