"""Document OCR service using Tesseract for identity document text extraction.

Extracts text from identity document images and parses structured fields
using document-type-specific regex patterns. Primary use case: Turkish TC Kimlik
cards that lack a machine-readable zone (MRZ).
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import pytesseract
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Result of OCR text extraction and field parsing.

    Attributes:
        raw_text: Full OCR output text.
        fields: Parsed key-value fields extracted from the text.
        confidence: Estimated confidence (0.0-1.0) based on how many
            expected fields were successfully extracted.
        method: Extraction method identifier.
    """

    raw_text: str = ""
    fields: dict = field(default_factory=dict)
    confidence: float = 0.0
    method: str = "tesseract_ocr"


# ---------------------------------------------------------------------------
# Turkish TC Kimlik regex patterns
# ---------------------------------------------------------------------------

# 11-digit TC Kimlik number (must start with non-zero digit)
_TC_NUMBER_PATTERN = re.compile(
    r"(?:T\.?C\.?\s*K[İI]ML[İI]K\s*(?:NO|NUMARASI)?\s*[:\-]?\s*)?(\d{11})"
)

# Surname field: SOYADI / SOYAD
_SURNAME_PATTERN = re.compile(
    r"SOYADI?\s*[:\-/]?\s*([A-ZÇĞİÖŞÜa-zçğıöşü\s]+?)(?:\n|$)",
    re.IGNORECASE,
)

# Given name field: ADI / AD
_GIVEN_NAME_PATTERN = re.compile(
    r"(?<![SOY])ADI\s*[:\-/]?\s*([A-ZÇĞİÖŞÜa-zçğıöşü\s]+?)(?:\n|$)",
    re.IGNORECASE,
)

# Date of birth: DOGUM TARIHI
_DOB_PATTERN = re.compile(
    r"DO[ĞG]UM\s*TAR[İI]H[İI]\s*[:\-/]?\s*(\d{1,2}[./]\d{1,2}[./]\d{4})",
    re.IGNORECASE,
)

# Expiry date: GECERLILIK TARIHI / SON GECERLILIK TARIHI
_EXPIRY_PATTERN = re.compile(
    r"(?:SON\s*)?GE[ÇC]ERL[İI]L[İI]K\s*TAR[İI]H[İI]\s*[:\-/]?\s*(\d{1,2}[./]\d{1,2}[./]\d{4})",
    re.IGNORECASE,
)

# Gender: CINSIYET / CINSIYETI
_GENDER_PATTERN = re.compile(
    r"C[İI]NS[İI]YET[İI]?\s*[:\-/]?\s*([EKek])",
    re.IGNORECASE,
)

# Nationality: UYRUGU / UYRUKLUGU
_NATIONALITY_PATTERN = re.compile(
    r"UYRU[ĞG]U?\s*[:\-/]?\s*([A-ZÇĞİÖŞÜa-zçğıöşü\s]+?)(?:\n|$)",
    re.IGNORECASE,
)


class DocumentOCR:
    """Extract text from identity document images using Tesseract OCR."""

    def extract_text(self, image: Image.Image, lang: str = "tur+eng") -> str:
        """Run full-page OCR on an image.

        Args:
            image: PIL Image of the document.
            lang: Tesseract language code(s). Defaults to Turkish + English.

        Returns:
            Extracted text string.
        """
        # Light preprocessing: sharpen to improve OCR accuracy on card images
        processed = image.convert("L").filter(ImageFilter.SHARPEN)

        try:
            text = pytesseract.image_to_string(processed, lang=lang)
        except pytesseract.TesseractNotFoundError:
            logger.error("Tesseract binary not found. Is tesseract-ocr installed?")
            return ""
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return ""

        logger.debug(f"OCR extracted {len(text)} chars")
        return text

    def extract_tc_kimlik(self, image: Image.Image) -> OCRResult:
        """Extract and parse Turkish TC Kimlik card fields from an image.

        Runs OCR then applies TC-Kimlik-specific regex patterns to pull out
        structured identity fields.

        Args:
            image: PIL Image of the TC Kimlik card.

        Returns:
            OCRResult with parsed fields and confidence score.
        """
        raw_text = self.extract_text(image, lang="tur+eng")
        if not raw_text.strip():
            return OCRResult(raw_text="", confidence=0.0)

        fields = self._parse_tc_kimlik_fields(raw_text)

        # Confidence based on how many of the 7 expected fields were found
        expected_keys = [
            "tc_number", "surname", "name",
            "date_of_birth", "expiry_date", "gender", "nationality",
        ]
        found = sum(1 for k in expected_keys if fields.get(k))
        confidence = round(found / len(expected_keys), 2)

        logger.info(
            f"TC Kimlik OCR: extracted {found}/{len(expected_keys)} fields, "
            f"confidence={confidence}"
        )

        return OCRResult(
            raw_text=raw_text,
            fields=fields,
            confidence=confidence,
            method="tesseract_ocr_tc_kimlik",
        )

    def extract_fields_from_text(
        self, text: str, document_type: str
    ) -> OCRResult:
        """Parse known document formats from raw OCR text.

        Currently supports:
        - ``turkish_id`` / ``tc_kimlik``: Turkish national ID card.

        Args:
            text: Raw OCR text.
            document_type: Document type key (e.g. ``turkish_id``).

        Returns:
            OCRResult with parsed fields.
        """
        if document_type in ("turkish_id", "tc_kimlik"):
            fields = self._parse_tc_kimlik_fields(text)
            expected = 7
            found = sum(1 for v in fields.values() if v)
            return OCRResult(
                raw_text=text,
                fields=fields,
                confidence=round(found / expected, 2),
                method="text_parse_tc_kimlik",
            )

        # Unknown document type -- return raw text only
        return OCRResult(raw_text=text, confidence=0.0, method="text_parse_unknown")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_tc_kimlik_fields(text: str) -> dict:
        """Apply TC Kimlik regex patterns to OCR text.

        Returns a dict with keys: tc_number, surname, name, date_of_birth,
        expiry_date, gender, nationality. Missing fields are ``None``.
        """
        fields: dict[str, Optional[str]] = {}

        m = _TC_NUMBER_PATTERN.search(text)
        fields["tc_number"] = m.group(1) if m else None

        m = _SURNAME_PATTERN.search(text)
        fields["surname"] = m.group(1).strip() if m else None

        m = _GIVEN_NAME_PATTERN.search(text)
        fields["name"] = m.group(1).strip() if m else None

        m = _DOB_PATTERN.search(text)
        fields["date_of_birth"] = m.group(1) if m else None

        m = _EXPIRY_PATTERN.search(text)
        fields["expiry_date"] = m.group(1) if m else None

        m = _GENDER_PATTERN.search(text)
        fields["gender"] = m.group(1).upper() if m else None

        m = _NATIONALITY_PATTERN.search(text)
        fields["nationality"] = m.group(1).strip() if m else None

        return fields
