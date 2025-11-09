#!/usr/bin/env python3
"""
Unified startup script - runs both frontend and backend together
"""
import subprocess
import sys
import os
from pathlib import Path
import time
import signal
import atexit

# Get project root
project_root = Path(__file__).parent.absolute()

# Store process references for cleanup
processes = []

def cleanup():
    """Kill all child processes on exit"""
    print("\nShutting down servers...")
    for p in processes:
        try:
            p.terminate()
            p.wait(timeout=5)
        except:
            try:
                p.kill()
            except:
                pass

atexit.register(cleanup)
signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))

def main():
    print("=" * 80)
    print("ALPHAEARTH INSURANCE AI - UNIFIED STARTUP")
    print("=" * 80)
    
    # Check if npm is installed - use shell=True on Windows
    try:
        subprocess.run(["npm", "--version"], capture_output=True, check=True, shell=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: npm not found. Please install Node.js first.")
        sys.exit(1)
    
    # Check frontend dependencies
    frontend_dir = project_root / 'frontend'
    node_modules = frontend_dir / 'node_modules'
    
    if not node_modules.exists():
        print("\nInstalling frontend dependencies...")
        subprocess.run(["npm", "install"], cwd=frontend_dir, check=True, shell=True)
    
    # Start backend server
    print("\n[1/2] Starting Backend API Server...")
    backend_dir = project_root / 'backend'
    backend_cmd = [sys.executable, "server.py"]
    backend_process = subprocess.Popen(
        backend_cmd,
        cwd=backend_dir,
        shell=True
    )
    processes.append(backend_process)
    
    # Wait for backend to start
    time.sleep(3)
    
    # Start frontend dev server
    print("\n[2/2] Starting Frontend Dev Server...")
    frontend_cmd = ["npm", "run", "dev"]
    frontend_process = subprocess.Popen(
        frontend_cmd,
        cwd=frontend_dir,
        shell=True
    )
    processes.append(frontend_process)
    
    print("\n" + "=" * 80)
    print("SERVERS RUNNING")
    print("=" * 80)
    print("Backend API:  http://localhost:5000")
    print("Frontend UI:  http://localhost:3000")
    print("=" * 80)
    print("\nPress Ctrl+C to stop all servers\n")
    
    # Stream output from both processes
    try:
        while True:
            # Check if processes are still running
            if backend_process.poll() is not None:
                print("Backend server stopped unexpectedly")
                break
            if frontend_process.poll() is not None:
                print("Frontend server stopped unexpectedly")
                break
            
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    
    cleanup()
    print("Shutdown complete")

if __name__ == "__main__":
    main()
