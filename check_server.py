#!/usr/bin/env python3
"""
Quick script to check if the ScoutSnout backend server is running
"""

import socket
import sys
from urllib.parse import urlparse

def check_port(host, port, timeout=2):
    """Check if a port is open on the given host"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        print(f"❌ Error checking port: {e}")
        return False

def main():
    # Default backend URL from Flutter app
    backend_url = "http://192.168.8.169:5001"
    
    # Allow custom URL as argument
    if len(sys.argv) > 1:
        backend_url = sys.argv[1]
    
    print("🔍 Checking ScoutSnout Backend Server Status")
    print("=" * 50)
    print(f"📍 Checking: {backend_url}")
    print()
    
    parsed = urlparse(backend_url)
    host = parsed.hostname
    port = parsed.port or 5001
    
    # Check if port is open
    print(f"🔌 Checking port {port} on {host}...")
    port_open = check_port(host, port)
    
    if port_open:
        print(f"✅ Port {port} is open")
        print()
        print("🎉 Backend server appears to be running!")
        print(f"   The server is listening on {host}:{port}")
        print()
        print("💡 To verify it's working, you can test with:")
        print(f"   curl {backend_url}/")
        print()
        return 0
    else:
        print(f"❌ Port {port} is not open on {host}")
        print()
        print("💡 The backend server is not running. To start it:")
        print("   1. cd backend")
        print("   2. python3 app.py")
        print("   OR")
        print("   2. ./start_server.sh")
        print()
        print("📝 Make sure:")
        print("   - You're in the backend directory")
        print("   - Python dependencies are installed (pip install -r requirements.txt)")
        print("   - The server IP matches your Flutter app configuration")
        return 1

if __name__ == "__main__":
    sys.exit(main())

