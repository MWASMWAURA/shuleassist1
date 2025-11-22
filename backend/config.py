"""
Configuration file for the backend
"""
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')

# Allow overriding the Gemini model via env; default to gemini-2.5-flash
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")