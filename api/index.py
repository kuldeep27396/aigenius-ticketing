"""
Vercel entry point for AIGenius Ticketing API
"""
import sys
import os

# Add src to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mangum import Adapter
from src.main import app

# Vercel/Lambda handler for ASGI app
handler = Adapter(app, lifespan="off")
