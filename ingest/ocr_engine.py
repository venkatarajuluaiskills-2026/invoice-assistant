"""
OCR engine with image preprocessing for invoice text extraction.
Tesseract OCR is preinstalled on TCS lab machines — system PATH.
Handles deskewing, contrast enhancement, and noise reduction.
"""
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance
from pathlib import Path
import re
import logging

logger = logging.getLogger(__name__)

# On TCS lab machines Tesseract is in system PATH — no custom path needed
# Uncomment below ONLY if pytesseract cannot find Tesseract automatically:
import os
pytesseract.pytesseract.tesseract_cmd = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tesseract.exe")


def preprocess_image(img: Image.Image) -> Image.Image:
    """
    Preprocess invoice image for optimal OCR accuracy.
    Applies grayscale conversion, contrast/sharpness enhancement,
    and deskewing using OpenCV moment-based angle detection.

    Args:
        img: PIL Image to preprocess

    Returns:
        Preprocessed PIL Image ready for OCR
    """
    # Convert to grayscale
    img_gray = img.convert("L")

    # Enhance contrast and sharpness
    img_gray = ImageEnhance.Contrast(img_gray).enhance(2.0)
    img_gray = ImageEnhance.Sharpness(img_gray).enhance(2.0)

    # Deskewing using OpenCV
    img_np = np.array(img_gray)
    _, thresh = cv2.threshold(
        img_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) > 10:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) > 0.5:
            img_gray = img_gray.rotate(angle, expand=True, fillcolor=255)

    return img_gray


def extract_text_from_image(img: Image.Image) -> str:
    """
    Extract text from preprocessed image using Tesseract OCR.
    Post-processes output to fix common OCR artifacts in numeric contexts.

    Args:
        img: PIL Image (should be preprocessed first)

    Returns:
        Cleaned extracted text string
    """
    text = pytesseract.image_to_string(img, config="--psm 6 --oem 3")

    # Fix common OCR substitutions in numeric contexts
    text = re.sub(r'(?<=\d)[lI](?=\d)', '1', text)
    text = re.sub(r'(?<=\d)O(?=\d)', '0', text)

    # Normalize whitespace and remove empty lines
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(line for line in lines if line)

    return text
