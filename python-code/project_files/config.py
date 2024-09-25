# config.py

import os
import logging
import asyncio
from dotenv import load_dotenv
from aiolimiter import AsyncLimiter

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")

HF_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CHUNK_SIZE = 16000 * 5 * 2  # 5 seconds of audio at 16kHz, 16-bit
TARGET_LANGUAGES = ['ger']  # Only German for testing
OUTPUT_DIR = "output"
MAX_CHUNK_SIZE = 25 * 1024 * 1024  # 25 MB
CURRENT_ENVIRONMENT = "M1 Max"
EXPERIMENT_COMPLETED = False

# Model configurations
CURRENT_GPT_MODEL = "gpt-4"  # Set a default GPT model
CURRENT_WHISPER_MODEL = "base"  # Set a default Whisper model

# Environments
ENVIRONMENTS = ["M1 Max", "NVIDIA 4080", "Hetzner Cloud", "Vultr Cloud"]

# Performance logging
PERFORMANCE_LOGS = {
    "transcription": {},
    "translation": {},
    "analysis": {},
    "total_processing": {}
}

# Rate limiting
RATE_LIMIT = AsyncLimiter(10, 60)  # 10 requests per minute
API_SEMAPHORE = asyncio.Semaphore(5)  # Adjust based on API limits

# AsyncOpenAI Client (using openai-python package)
import openai

