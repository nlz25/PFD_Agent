#!/usr/bin/env python3
"""
MCP Server Startup Script (Python version)
Automatically starts all MCP servers for the PFD-Agent system
"""

import os
import sys
import time
import signal
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# ANSI color codes
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    NC = '\033[0m'  # No Color

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
LOG_DIR = PROJECT_ROOT / "logs" / "mcp_servers"
PID_FILE = LOG_DIR / "mcp_servers.json"

# Server configurations: name -> (port, script_path)
SERVERS = {
    "database": (50001, PROJECT_ROOT / "tools/database/server.py"),
    "dpa": (50002, PROJECT_ROOT / "tools/dpa/server.py"),
    "abacus": (50003, PROJECT_ROOT / "tools/abacus/server.py"),
    "quest": (50004, PROJECT_ROOT / "tools/quest/server.py"),
    "vasp": (50005, PROJECT_ROOT / "tools/vasp/server.py"),
}

# Default configuration
DEFAULT_HOST = os.environ.get("MCP_HOST", "localhost")
DEFAULT_TRANSPORT = os.environ.get("MCP_TRANSPORT", "sse")


def print_message(color: str, message: str) -> None:
    """Print colored message to console."""
    print(f"{color}{message}{Colors.NC}")


def check_port(port: int) -> bool:
    """Check if a port is in use."""
    try:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            return result == 0
    except Exception:
        return False


def start_server(
    name: str,
    port: int,
    script: Path,
    host: str = DEFAULT_HOST,
    transport: str = DEFAULT_TRANSPORT
) -> Optional[Dict]:
    """Start a single MCP server."""
    log_file = LOG_DIR / f"{name}-server.log"

    # Check if server script exists
    if not script.exists():
        print_message(Colors.RED, f"✗ Server script not found: {script}")
        return None

    # Check if port is already in use
    if check_port(port):
        print_message(Colors.YELLOW, f"⚠ Port {port} is already in use ({name} may already be running)")
        return None

    # Start the server
    print_message(Colors.GREEN, f"Starting {name} server on port {port}...")
    
    try:
        with open(log_file, 'w') as log:
            process = subprocess.Popen(
                [
                    sys.executable,  # Use the current Python interpreter
                    str(script),
                    "--port", str(port),
                    "--host", host,
                    "--transport", transport
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True  # Detach from parent
            )
        
        # Wait a moment and check if process is still running
        time.sleep(1)
        if process.poll() is None:  # Process is still running
            server_info = {
                "pid": process.pid,
                "port": port,
                "script": str(script),
                "log_file": str(log_file),
                "started_at": time.time()
            }
            print_message(Colors.GREEN, f"✓ {name} server started (PID: {process.pid}, Port: {port})")
            return server_info
        else:
            print_message(Colors.RED, f"✗ Failed to start {name} server. Check log: {log_file}")
            return None
    except Exception as e:
        print_message(Colors.RED, f"✗ Error starting {name} server: {e}")
        return None


def stop_servers(server_names: Optional[List[str]] = None) -> None:
    """Stop all or selected running MCP servers."""
    if not PID_FILE.exists():
        print_message(Colors.YELLOW, "No PID file found. Servers may not be running.")
        return

    try:
        with open(PID_FILE, 'r') as f:
            servers = json.load(f)
    except Exception as e:
        print_message(Colors.RED, f"Error reading PID file: {e}")
        return

    # Determine which servers to stop
    if server_names:
        # Validate server names
        invalid_names = [name for name in server_names if name not in servers]
        if invalid_names:
            print_message(Colors.RED, f"✗ Invalid server name(s): {', '.join(invalid_names)}")
            print_message(Colors.YELLOW, f"Running servers: {', '.join(servers.keys())}")
            return
        
        servers_to_stop = {name: servers[name] for name in server_names}
        print_message(Colors.YELLOW, f"Stopping selected MCP servers: {', '.join(server_names)}...")
    else:
        servers_to_stop = servers
        print_message(Colors.YELLOW, "Stopping all MCP servers...")

    # Stop the selected servers
    for name, info in servers_to_stop.items():
        pid = info.get("pid")
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
                print_message(Colors.GREEN, f"✓ Stopped {name} server (PID: {pid})")
            except ProcessLookupError:
                print_message(Colors.YELLOW, f"⚠ {name} server (PID: {pid}) not running")
            except Exception as e:
                print_message(Colors.RED, f"✗ Error stopping {name}: {e}")

    # Update PID file
    if server_names:
        # Remove only the stopped servers from PID file
        remaining_servers = {name: info for name, info in servers.items() if name not in servers_to_stop}
        if remaining_servers:
            with open(PID_FILE, 'w') as f:
                json.dump(remaining_servers, f, indent=2)
            print_message(Colors.GREEN, f"Selected servers stopped. {len(remaining_servers)} server(s) still running.")
        else:
            PID_FILE.unlink()
            print_message(Colors.GREEN, "All servers stopped.")
    else:
        # Remove PID file when stopping all servers
        PID_FILE.unlink()
        print_message(Colors.GREEN, "All servers stopped.")


def check_status() -> None:
    """Check the status of all MCP servers."""
    print_message(Colors.GREEN, "Checking MCP server status...")
    print()

    if not PID_FILE.exists():
        print_message(Colors.YELLOW, "No servers are currently registered (PID file not found)")
        return

    try:
        with open(PID_FILE, 'r') as f:
            servers = json.load(f)
    except Exception as e:
        print_message(Colors.RED, f"Error reading PID file: {e}")
        return

    all_running = True
    for name, info in servers.items():
        pid = info.get("pid")
        port = info.get("port")
        
        try:
            os.kill(pid, 0)  # Check if process exists
            if check_port(port):
                print_message(Colors.GREEN, f"✓ {name}: running (PID: {pid}, Port: {port})")
            else:
                print_message(Colors.YELLOW, f"⚠ {name}: process running but port {port} not listening (PID: {pid})")
                all_running = False
        except ProcessLookupError:
            print_message(Colors.RED, f"✗ {name}: not running (stale PID: {pid})")
            all_running = False

    if all_running:
        print()
        print_message(Colors.GREEN, "All servers are running properly.")


def start_all_servers(
    host: str = DEFAULT_HOST, 
    transport: str = DEFAULT_TRANSPORT,
    server_names: Optional[List[str]] = None
) -> None:
    """Start all or selected MCP servers."""
    # Determine which servers to start
    if server_names:
        # Validate server names
        invalid_names = [name for name in server_names if name not in SERVERS]
        if invalid_names:
            print_message(Colors.RED, f"✗ Invalid server name(s): {', '.join(invalid_names)}")
            print_message(Colors.YELLOW, f"Available servers: {', '.join(SERVERS.keys())}")
            return
        
        servers_to_start = {name: SERVERS[name] for name in server_names}
        print_message(Colors.GREEN, f"Starting selected MCP servers: {', '.join(server_names)}...")
    else:
        servers_to_start = SERVERS
        print_message(Colors.GREEN, "Starting all MCP servers...")
    
    print()

    # Create log directory
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing server info if available
    existing_servers = {}
    if PID_FILE.exists():
        try:
            with open(PID_FILE, 'r') as f:
                existing_servers = json.load(f)
        except Exception:
            pass

    servers_info = existing_servers.copy()
    success_count = 0
    total_count = len(servers_to_start)

    for name, (port, script) in servers_to_start.items():
        info = start_server(name, port, script, host, transport)
        if info:
            servers_info[name] = info
            success_count += 1
        print()

    # Save PID file
    with open(PID_FILE, 'w') as f:
        json.dump(servers_info, f, indent=2)

    print("=" * 50)
    print_message(Colors.GREEN, f"Startup complete: {success_count}/{total_count} servers running")
    print("=" * 50)
    print()
    print(f"Log directory: {LOG_DIR}")
    print(f"PID file: {PID_FILE}")
    print()
    print_message(Colors.YELLOW, "Use './script/start_mcp_servers.py status' to check server status")
    print_message(Colors.YELLOW, "Use './script/start_mcp_servers.py stop' to stop all servers")


def restart_servers() -> None:
    """Restart all MCP servers."""
    stop_servers()
    time.sleep(2)
    start_all_servers()


def main():
    """Main entry point."""
    command = sys.argv[1] if len(sys.argv) > 1 else "start"
    
    # Get optional server names (for start/stop commands)
    server_names = sys.argv[2:] if len(sys.argv) > 2 else None

    if command == "start":
        start_all_servers(server_names=server_names)
    elif command == "stop":
        stop_servers(server_names=server_names)
    elif command == "restart":
        restart_servers()
    elif command == "status":
        check_status()
    else:
        print(f"Usage: {sys.argv[0]} {{start|stop|restart|status}} [server_names...]")
        print()
        print("Commands:")
        print("  start [server_names...]  - Start all or selected MCP servers (default)")
        print("  stop [server_names...]   - Stop all or selected running MCP servers")
        print("  restart                  - Restart all MCP servers")
        print("  status                   - Check the status of all MCP servers")
        print()
        print("Available servers:")
        for name, (port, _) in SERVERS.items():
            print(f"  {name:<12} - Port {port}")
        print()
        print("Examples:")
        print(f"  {sys.argv[0]} start                    # Start all servers")
        print(f"  {sys.argv[0]} start database dpa       # Start only database and dpa servers")
        print(f"  {sys.argv[0]} stop                     # Stop all servers")
        print(f"  {sys.argv[0]} stop quest vasp          # Stop only quest and vasp servers")
        print(f"  {sys.argv[0]} status                   # Check status of all servers")
        print()
        print("Environment variables:")
        print("  MCP_HOST      - Server host (default: localhost)")
        print("  MCP_TRANSPORT - Transport protocol (default: sse)")
        sys.exit(1)


if __name__ == "__main__":
    main()
