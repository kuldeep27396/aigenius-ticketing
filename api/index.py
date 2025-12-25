"""
Vercel serverless function entry point
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import app

# Vercel handler - export the ASGI app directly
# Vercel's Python runtime will handle the ASGI app
app_handler = app

# Lambda-style handler for Vercel
def handler(event, context):
    """AWS Lambda-style handler for Vercel"""
    from mangum import Adapter
    adapter = Adapter(app, lifespan="off")
    return adapter(event, context)
