"""Machine Readable Zone (MRZ) parser for identity documents.

Supports:
- TD1 format: ID cards (3 lines x 30 characters)
- TD3 format: Passports (2 lines x 44 characters)

Pure string parsing — no ML or OCR dependencies.
References: ICAO Doc 9303 (Machine Readable Travel Documents).
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MRZData:
    """Parsed MRZ data from an identity document.

    Attributes:
        format: MRZ format type (TD1 or TD3)
        document_type: Document type code (P=passport, I=ID card, etc.)
        country_code: Issuing country (3-letter ISO code)
        surname: Holder's surname
        given_names: Holder's given name(s)
        document_number: Document number
        nationality: Holder's nationality (3-letter ISO code)
        date_of_birth: Date of birth (YYMMDD)
        sex: Sex (M/F/<)
        expiry_date: Expiry date (YYMMDD)
        optional_data_1: Optional data field 1
        optional_data_2: Optional data field 2 (TD1 only)
        check_digits_valid: Whether all check digits passed validation
        raw_mrz: Raw MRZ lines
        errors: List of validation errors
    """

    format: str  # "TD1" or "TD3"
    document_type: str = ""
    country_code: str = ""
    surname: str = ""
    given_names: str = ""
    document_number: str = ""
    nationality: str = ""
    date_of_birth: str = ""
    sex: str = ""
    expiry_date: str = ""
    optional_data_1: str = ""
    optional_data_2: str = ""
    check_digits_valid: bool = False
    raw_mrz: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _clean_filler(text: str) -> str:
    """Replace MRZ filler characters '<' with spaces and strip."""
    return text.replace("<", " ").strip()


def _compute_check_digit(data: str) -> int:
    """Compute MRZ check digit using ICAO 9303 algorithm.

    Weights cycle: 7, 3, 1
    Characters: 0-9 = 0-9, A-Z = 10-35, < = 0

    Args:
        data: MRZ field string

    Returns:
        Check digit (0-9)
    """
    weights = [7, 3, 1]
    total = 0

    for i, char in enumerate(data):
        if char.isdigit():
            value = int(char)
        elif char.isalpha():
            value = ord(char.upper()) - ord("A") + 10
        elif char == "<":
            value = 0
        else:
            value = 0

        total += value * weights[i % 3]

    return total % 10


def _validate_check_digit(data: str, expected: str) -> bool:
    """Validate a check digit against computed value.

    Args:
        data: MRZ field string
        expected: Expected check digit character

    Returns:
        True if check digit is valid
    """
    if not expected.isdigit():
        return False
    return _compute_check_digit(data) == int(expected)


def parse_td3(lines: list[str]) -> MRZData:
    """Parse TD3 (passport) MRZ format.

    Format: 2 lines x 44 characters each.

    Line 1: P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<
    Line 2: L898902C36UTO7408122F1204159ZE184226B<<<<<10

    Args:
        lines: List of 2 MRZ lines, each 44 characters

    Returns:
        MRZData with parsed fields
    """
    result = MRZData(format="TD3", raw_mrz=lines)
    errors = []

    line1 = lines[0].ljust(44, "<")
    line2 = lines[1].ljust(44, "<")

    # Line 1: Document type (2) + Country (3) + Names (39)
    result.document_type = _clean_filler(line1[0:2])
    result.country_code = _clean_filler(line1[2:5])

    # Names: surname<<given_names
    names_field = line1[5:44]
    name_parts = names_field.split("<<", 1)
    result.surname = _clean_filler(name_parts[0])
    if len(name_parts) > 1:
        result.given_names = _clean_filler(name_parts[1])

    # Line 2: Doc number (9) + check (1) + nationality (3) + DOB (6) + check (1)
    #          + sex (1) + expiry (6) + check (1) + optional (14) + check (1) + overall check (1)
    doc_number = line2[0:9]
    doc_check = line2[9]
    result.nationality = _clean_filler(line2[10:13])
    dob = line2[13:19]
    dob_check = line2[19]
    result.sex = _clean_filler(line2[20:21])
    expiry = line2[21:27]
    expiry_check = line2[27]
    optional = line2[28:42]
    optional_check = line2[42]
    overall_check = line2[43]

    result.document_number = _clean_filler(doc_number)
    result.date_of_birth = dob
    result.expiry_date = expiry
    result.optional_data_1 = _clean_filler(optional)

    # Validate check digits
    all_valid = True

    if not _validate_check_digit(doc_number, doc_check):
        errors.append("Document number check digit invalid")
        all_valid = False

    if not _validate_check_digit(dob, dob_check):
        errors.append("Date of birth check digit invalid")
        all_valid = False

    if not _validate_check_digit(expiry, expiry_check):
        errors.append("Expiry date check digit invalid")
        all_valid = False

    if not _validate_check_digit(optional, optional_check):
        errors.append("Optional data check digit invalid")
        all_valid = False

    # Overall check digit: doc_number + check + dob + check + expiry + check + optional + check
    composite = doc_number + doc_check + dob + dob_check + expiry + expiry_check + optional + optional_check
    if not _validate_check_digit(composite, overall_check):
        errors.append("Overall composite check digit invalid")
        all_valid = False

    result.check_digits_valid = all_valid
    result.errors = errors

    logger.info(
        f"TD3 MRZ parsed: doc_type={result.document_type}, "
        f"country={result.country_code}, name={result.surname}/{result.given_names}, "
        f"valid={all_valid}"
    )

    return result


def parse_td1(lines: list[str]) -> MRZData:
    """Parse TD1 (ID card) MRZ format.

    Format: 3 lines x 30 characters each.

    Line 1: I<UTOD231458907<<<<<<<<<<<<<<<
    Line 2: 7408122F1204159UTO<<<<<<<<<<<6
    Line 3: ERIKSSON<<ANNA<MARIA<<<<<<<<<<

    Args:
        lines: List of 3 MRZ lines, each 30 characters

    Returns:
        MRZData with parsed fields
    """
    result = MRZData(format="TD1", raw_mrz=lines)
    errors = []

    line1 = lines[0].ljust(30, "<")
    line2 = lines[1].ljust(30, "<")
    line3 = lines[2].ljust(30, "<")

    # Line 1: Document type (2) + Country (3) + Doc number (9) + check (1) + Optional (15)
    result.document_type = _clean_filler(line1[0:2])
    result.country_code = _clean_filler(line1[2:5])
    doc_number = line1[5:14]
    doc_check = line1[14]
    result.optional_data_1 = _clean_filler(line1[15:30])

    result.document_number = _clean_filler(doc_number)

    # Line 2: DOB (6) + check (1) + sex (1) + expiry (6) + check (1) + nationality (3) + optional (11) + overall check (1)
    dob = line2[0:6]
    dob_check = line2[6]
    result.sex = _clean_filler(line2[7:8])
    expiry = line2[8:14]
    expiry_check = line2[14]
    result.nationality = _clean_filler(line2[15:18])
    result.optional_data_2 = _clean_filler(line2[18:29])
    overall_check = line2[29]

    result.date_of_birth = dob
    result.expiry_date = expiry

    # Line 3: Names (30) - surname<<given_names
    names_field = line3[0:30]
    name_parts = names_field.split("<<", 1)
    result.surname = _clean_filler(name_parts[0])
    if len(name_parts) > 1:
        result.given_names = _clean_filler(name_parts[1])

    # Validate check digits
    all_valid = True

    if not _validate_check_digit(doc_number, doc_check):
        errors.append("Document number check digit invalid")
        all_valid = False

    if not _validate_check_digit(dob, dob_check):
        errors.append("Date of birth check digit invalid")
        all_valid = False

    if not _validate_check_digit(expiry, expiry_check):
        errors.append("Expiry date check digit invalid")
        all_valid = False

    # Overall: line1[5:30] + line2[0:7] + line2[8:15] + line2[18:29]
    composite = line1[5:30] + line2[0:7] + line2[8:15] + line2[18:29]
    if not _validate_check_digit(composite, overall_check):
        errors.append("Overall composite check digit invalid")
        all_valid = False

    result.check_digits_valid = all_valid
    result.errors = errors

    logger.info(
        f"TD1 MRZ parsed: doc_type={result.document_type}, "
        f"country={result.country_code}, name={result.surname}/{result.given_names}, "
        f"valid={all_valid}"
    )

    return result


def detect_and_parse_mrz(text: str) -> Optional[MRZData]:
    """Auto-detect MRZ format from text and parse it.

    Scans text for MRZ patterns (lines of uppercase + digits + < characters)
    and attempts to parse as TD1 or TD3 format.

    Args:
        text: Text that may contain MRZ lines

    Returns:
        MRZData if MRZ found and parsed, None otherwise
    """
    # MRZ lines contain only uppercase letters, digits, and '<'
    mrz_line_pattern = re.compile(r"^[A-Z0-9<]{20,44}$")

    lines = text.strip().split("\n")
    mrz_lines = []

    for line in lines:
        cleaned = line.strip()
        if mrz_line_pattern.match(cleaned):
            mrz_lines.append(cleaned)

    if not mrz_lines:
        logger.debug("No MRZ lines detected in text")
        return None

    # Try TD3 (2 lines x 44 chars)
    if len(mrz_lines) >= 2:
        td3_candidates = [line for line in mrz_lines if len(line) == 44]
        if len(td3_candidates) >= 2:
            logger.info("Detected TD3 (passport) MRZ format")
            return parse_td3(td3_candidates[:2])

    # Try TD1 (3 lines x 30 chars)
    if len(mrz_lines) >= 3:
        td1_candidates = [line for line in mrz_lines if len(line) == 30]
        if len(td1_candidates) >= 3:
            logger.info("Detected TD1 (ID card) MRZ format")
            return parse_td1(td1_candidates[:3])

    # Fallback: try with whatever lines we have, padding if needed
    if len(mrz_lines) >= 2 and all(len(line) >= 40 for line in mrz_lines[:2]):
        logger.info("Attempting TD3 parse with approximate-length lines")
        return parse_td3(mrz_lines[:2])

    if len(mrz_lines) >= 3 and all(len(line) >= 28 for line in mrz_lines[:3]):
        logger.info("Attempting TD1 parse with approximate-length lines")
        return parse_td1(mrz_lines[:3])

    logger.warning(f"Found {len(mrz_lines)} MRZ-like lines but could not determine format")
    return None


def format_date(yymmdd: str) -> str:
    """Convert YYMMDD to human-readable date string.

    Uses pivot year 2000: YY < 30 -> 20YY, YY >= 30 -> 19YY.

    Args:
        yymmdd: Date in YYMMDD format

    Returns:
        Date in YYYY-MM-DD format, or original string if invalid
    """
    if len(yymmdd) != 6 or not yymmdd.isdigit():
        return yymmdd

    yy = int(yymmdd[0:2])
    mm = yymmdd[2:4]
    dd = yymmdd[4:6]

    year = 2000 + yy if yy < 30 else 1900 + yy

    return f"{year}-{mm}-{dd}"
