"""
Simple UI to visuaslize the detection results
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import os
import sys

class DashboardHandler(SimpleHTTPRequestHandler):
    """Custom handler to serve the dashboard"""
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == "/" or self.path == "":
            self.path = "/dashboard.html"
        return super().do_GET()
    
    def log_message(self, format, *args):
        """Custom logging"""
        print(f"[{self.log_date_time_string()}] {format % args}")

def run_server(port=8000):
    """Run the dashboard server"""
    # Change to static directory
    static_dir = Path(__file__).parent / "app" / "static"
    
    if not static_dir.exists():
        print(f"Error: Static directory not found at {static_dir}")
        sys.exit(1)
    
    os.chdir(static_dir)
    
    server_address = ("", port)
    httpd = HTTPServer(server_address, DashboardHandler)
    
    print(f"✓ Dashboard server running at http://localhost:{port}")
    print(f"✓ Serving files from: {static_dir}")
    print(f"✓ Press Ctrl+C to stop")
    print("\n" + "="*60)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n✓ Server stopped")
        sys.exit(0)

if __name__ == "__main__":
    run_server()
