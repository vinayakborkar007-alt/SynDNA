"""
Standalone static file server for the SynDNA frontend.

Runs the frontend completely independently from the backend, on its own
port (default 5500). The frontend talks to the FastAPI backend at
http://localhost:8000 purely over HTTP (CORS is already enabled on the
backend), so the two processes never need to know about each other beyond
that URL.

Usage:
    python server.py            # serves on http://localhost:5500
    python server.py 3000       # serves on a custom port
"""
import http.server
import socketserver
import sys
import os

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 5500

os.chdir(os.path.dirname(os.path.abspath(__file__)))

Handler = http.server.SimpleHTTPRequestHandler

class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

if __name__ == "__main__":
    with ReusableTCPServer(("", PORT), Handler) as httpd:
        print(f"SynDNA frontend running at http://localhost:{PORT}")
        print("Make sure the backend is running separately at http://localhost:8000")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down frontend server.")
            httpd.shutdown()
