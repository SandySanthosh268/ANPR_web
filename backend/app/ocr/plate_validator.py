import re

# Indian plate format: 2-letter state code, 1-2 digit RTO code, 0-3 letter
# series, 3-4 digit number (e.g. TN01BA7320, TN01AB1234, KA05MH1234). PaddleOCR
# on a small, often blurry crop regularly returns near-miss garbage (missing
# leading letters, digit/letter confusion, truncated reads) — those are
# rejected here rather than shown as if they were a confident, correct read.
_PLATE_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{3,4}$")
_MIN_LENGTH = 8
_MAX_LENGTH = 11


def is_valid_plate(text: str) -> bool:
    if not (_MIN_LENGTH <= len(text) <= _MAX_LENGTH):
        return False
    return bool(_PLATE_PATTERN.fullmatch(text))
