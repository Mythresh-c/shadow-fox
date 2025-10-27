import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Gemini API Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Model Configuration
MODEL_NAME = "gemini-pro"
MAX_OUTPUT_TOKENS = 2048
TEMPERATURE = 0.7

# Analysis Parameters
BATCH_SIZE = 5
MAX_RETRIES = 3
TIMEOUT_SECONDS = 30

# Output Directories
RESULTS_DIR = "results"
VISUALIZATIONS_DIR = "results/visualizations"
DATA_DIR = "results/data"

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

print("Configuration loaded successfully!")
print(f"Model: {MODEL_NAME}")
print(f"Results directory: {RESULTS_DIR}")
