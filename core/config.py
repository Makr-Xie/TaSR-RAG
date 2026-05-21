# API Configuration
CHAT_API_BASE = "http://localhost:1225/v1"
EMBED_API_BASE = "http://localhost:12261/v1"
# Default to EMPTY for local vLLM, can be overridden by env vars if needed in future
API_KEY = "EMPTY" 

# Model Configuration
# Single model for all chat tasks
CHAT_MODEL = "Qwen2.5-72B-Instruct"    # Used for decomposition, query typing, and content generation
EMBED_MODEL = "Qwen3-Embedding-8B"

# Timeout settings
DEFAULT_TIMEOUT = 120.0
