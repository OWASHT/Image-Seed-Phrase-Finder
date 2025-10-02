# Google Vision API ID file path (JSON)
GOOGLE_CREDENTIALS_PATH = "google-credentials.json"

# Main folder to be scanned
SCAN_DIRECTORY = "images"
OUTPUT_DIRECTORY = None

MAX_WORKERS = 4                  # Number of parallel processes (adjust according to the number of CPU cores)
MIN_WORDS_FOR_COPY = 5           # Minimum number of words to be copied to the ImageWords folder
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB - larger files are skipped

# Prohibited first words
FORBIDDEN_FIRST_WORDS = {
    "the", "and", "you", "that", "was", "for", "are", "with", "his", "they",
    "this", "have", "from", "one", "had", "word", "but", "not", "what", "all",
    "were", "when", "your", "said", "there", "use", "each", "which", "she", "how"
}

# OCR engine: Google" or "Tesseract"
OCR_ENGINE = "tesseract" # Default: tesseract. Backup: "Google Vision (paid)"

TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
