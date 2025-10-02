# Seed Phrase Image Finder

A powerful Python tool that automatically detects BIP39 seed phrases in images using advanced OCR (Optical Character Recognition) technology. This tool can process large collections of images and identify valid cryptocurrency seed phrases with high accuracy.

## ⚠️ Security Warning

**This tool is designed for educational purposes and legitimate recovery scenarios only. Always handle seed phrases securely and never share them with others. The authors are not responsible for any misuse of this software.**

## Features

- **Multi-OCR Engine Support**: Choose between Google Vision API (high accuracy) and Tesseract OCR (free, local processing)
- **Batch Processing**: Process thousands of images automatically with parallel processing
- **Smart Image Enhancement**: Applies various preprocessing techniques to improve OCR accuracy
- **BIP39 Validation**: Validates detected phrases using proper BIP39 checksum verification
- **Multiple Layout Detection**: Handles both horizontal and vertical seed phrase arrangements
- **Format Support**: Works with PNG, JPEG, GIF, BMP, TIFF, and WebP image formats
- **Intelligent Filtering**: Automatically categorizes images based on detected content
- **Detailed Logging**: Comprehensive logging of all detected phrases and processing results

## Prerequisites

### System Requirements

- Python 3.7 or higher
- Windows operating system

### Required Software

**For Tesseract OCR (Recommended for beginners):**
- Download and install [Tesseract OCR](https://github.com/tesseract-ocr/tesseract/releases/tag/5.5.0)
  - Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)

**For Google Vision API (Optional, higher accuracy):**
- Google Cloud Platform account
- Vision API enabled and credentials JSON file

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/seed-phrase-image-finder.git
cd bip39-seed-detector
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Basic installation (Tesseract OCR)
pip install -r requirements.txt

# For Google Vision API support (optional)
pip install google-cloud-vision
```

### 4. Download BIP39 Wordlist

Download the official BIP39 English wordlist and place it in the project directory:

```bash
# Using wget
wget https://raw.githubusercontent.com/bitcoin/bips/master/bip-0039/english.txt

# Or using curl
curl -O https://raw.githubusercontent.com/bitcoin/bips/master/bip-0039/english.txt
```

### 5. Configure Settings

Edit `config.py` to match your system setup:

```python
# For Tesseract OCR
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# OCR Engine Selection
OCR_ENGINE = "tesseract"  # or "google" for Google Vision API

# Processing Settings
SCAN_DIRECTORY = "images"  # Directory containing images to scan
MAX_WORKERS = 4           # Adjust based on your CPU cores
```

## Usage

### Basic Usage

1. **Place your images** in the `images` folder (or specify a different directory in `config.py`)

2. **Run the scanner**:
```bash
python main.py
```

3. **Check the results**:
   - `detected_seeds.txt` - List of found seed phrases
   - `detected_seeds_detailed.txt` - Detailed log with source files
   - `images_with_seeds/` - Copies of images containing valid seed phrases
   - `images_with_words/` - Images with BIP39 words but no complete phrases

### Advanced Configuration

#### Using Google Vision API

1. **Set up Google Cloud credentials**:
   - Create a Google Cloud project
   - Enable the Vision API
   - Download service account JSON credentials
   - Place the JSON file in the project directory

2. **Update config.py**:
```python
GOOGLE_CREDENTIALS_PATH = "google-credentials.json"
OCR_ENGINE = "google"
```

#### Custom Processing Parameters

```python
# Adjust these in config.py based on your needs
MAX_WORKERS = 8                    # More workers for faster processing
MIN_WORDS_FOR_COPY = 3            # Lower threshold for interesting images
MAX_IMAGE_SIZE = 50 * 1024 * 1024 # Increase for larger images
```

### Example Output

```
🚀 Seed Phrase Detector
==================================================
📚 Loaded 2048 BIP39 words
🖥️  Using Tesseract OCR
📁 Found 1,247 images. Starting processing...
==================================================
🔍 Processing: images/wallet_backup_1.png
✅ Seed found: images/wallet_backup_1.png
💾 Copied to: images_with_seeds/wallet_backup_1.png
🔍 Processing: images/note_photo.jpg
🔤 Interesting (found 7 BIP39 words)

==================================================
✅ SCAN COMPLETE
==================================================
Total files:        1,247
Processed files:    1,247
Seeds found:        3
Time elapsed:       342.18 seconds
```

## Project Structure

```
bip39-seed-detector/
├── main.py                    # Main application code
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies
├── english.txt               # BIP39 wordlist
├── images/                   # Input images directory
├── images_with_seeds/        # Output: Images with valid seeds
├── images_with_words/        # Output: Images with BIP39 words
├── detected_seeds.txt        # Output: List of found phrases
└── detected_seeds_detailed.txt # Output: Detailed log
```

## Configuration Options

### Core Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `OCR_ENGINE` | OCR engine to use (`"tesseract"` or `"google"`) | `"tesseract"` |
| `SCAN_DIRECTORY` | Directory to scan for images | `"images"` |
| `MAX_WORKERS` | Number of parallel processing threads | `4` |
| `MIN_WORDS_FOR_COPY` | Minimum BIP39 words to flag as interesting | `5` |
| `MAX_IMAGE_SIZE` | Maximum image file size to process (bytes) | `10MB` |

### OCR Engine Paths

| Setting | Description | Example |
|---------|-------------|---------|
| `TESSERACT_CMD` | Path to Tesseract executable | `C:\Program Files\Tesseract-OCR\tesseract.exe` |
| `GOOGLE_CREDENTIALS_PATH` | Path to Google Cloud credentials JSON | `"google-credentials.json"` |

## Contributing

We welcome contributions! Please follow these guidelines:

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes with proper documentation
4. Add tests if applicable
5. Submit a pull request

### Code Style

- Follow PEP 8 Python style guidelines
- Use meaningful variable names and add comments
- Include docstrings for all functions and classes
- Maintain backward compatibility when possible

### Bug Reports

Please include:
- Python version and operating system
- Complete error messages
- Steps to reproduce the issue
- Sample images (if safe to share)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This software is provided for educational and legitimate recovery purposes only. Users are responsible for:

- Ensuring they have proper authorization to process any images
- Handling detected seed phrases securely
- Complying with applicable laws and regulations
- Not using this tool for unauthorized access to cryptocurrency wallets

The authors assume no responsibility for misuse of this software.

## Support

### Troubleshooting

**Common Issues:**

1. **Tesseract not found**: Ensure Tesseract is installed and `TESSERACT_CMD` path is correct
2. **Google Vision API errors**: Check credentials file path and API billing status
3. **Memory issues**: Reduce `MAX_WORKERS` or `MAX_IMAGE_SIZE` for systems with limited RAM
4. **No images found**: Verify image file extensions and directory path

## Acknowledgments

- [Bitcoin Improvement Proposal 39 (BIP39)](https://github.com/bitcoin/bips/blob/master/bip-0039.mediawiki)
- [Tesseract OCR Project](https://github.com/tesseract-ocr/tesseract)
- [Google Cloud Vision API](https://cloud.google.com/vision)
- All contributors who help improve this project

---

**⭐ Star this repository if you find it useful!**