"""
Vercel entry point for AIGenius Ticketing API
"""
import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Set environment variables for serverless
os.environ.setdefault("ENVIRONMENT", "production")
os.environ.setdefault("SLA_CONFIG_PATH", "/tmp/sla_config.yaml")
os.environ.setdefault("SLA_EVALUATION_INTERVAL", "0")  # Disable scheduler in serverless

from mangum import Adapter
from src.main import app

# Lambda handler for ASGI app (disable lifespan for serverless)
handler = Adapter(app, lifespan="off")
