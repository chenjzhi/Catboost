# config.py
API_KEY = "sk-a43cbdbf5d25fdcb57dcbe42a7d76ec21629a9ab83c0231f069d4f3ebdf05f06"
API_BASE_URL = "https://chatnio.cdreader.vip/v1"
MODEL_NAME = "qwen-plus"
CSV_FILE_PATH = "test/tests.csv"
TIMEOUT = 30.0  # API request timeout in seconds
MAX_RETRIES = 3  # Max retries for API calls
RETRY_DELAY = 2  # Seconds to wait between retries
MAX_WORKERS = 4  # Max threads for concurrent processing
RESULTS_FILE_PATH = "data/analysis_results.json"
CSV_OUTPUT_PATH = "data/analysis_results.csv"