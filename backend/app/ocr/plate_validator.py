import re

# Indian plate format: 2-letter state code, 1-2 digit RTO code, 0-3 letter
# series, 3-4 digit number (e.g. TN01BA7320, TN01AB1234, KA05MH1234). PaddleOCR
# on a small, often blurry crop regularly returns near-miss garbage (missing
# leading letters, digit/letter confusion, truncated reads) — those are
# rejected here rather than shown as if they were a confident, correct read.
_PLATE_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{3,4}$")
_MIN_LENGTH = 8
_MAX_LENGTH = 11

# All current Indian state/UT RTO codes, plus BH (the unified "Bharat" series).
# Used to correct a single misread letter in the state-code prefix (e.g. OCR
# confusing visually similar letters like N/H) — a real plate's first two
# letters must be one of these, so a near-miss with exactly one of these as
# its unique closest match is overwhelmingly likely the intended code rather
# than a coincidence.
_VALID_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "DN", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK",
    "UP", "UA", "WB", "BH",
}

# Letter pairs PaddleOCR commonly confuses at small plate-crop resolution,
# because they're visually similar in the font Indian plates use (e.g. N and
# H both being tall verticals joined by a stroke). Used to break ties when
# more than one real state code is a single-letter edit away from a misread
# prefix — plain edit distance alone can't tell 'TH' -> 'TN' (a real, common
# confusion) apart from 'TH' -> 'TR' or 'TH' -> 'TS' (not realistic misreads),
# so without this every one of those looks equally "close" and the correction
# would have to give up as ambiguous.
_COMMON_CONFUSIONS = {
    frozenset({"N", "H"}), frozenset({"O", "Q"}), frozenset({"I", "L"}),
    frozenset({"B", "R"}), frozenset({"D", "O"}), frozenset({"E", "F"}),
    frozenset({"G", "C"}), frozenset({"M", "N"}), frozenset({"V", "Y"}),
}


def is_valid_plate(text: str) -> bool:
    if not (_MIN_LENGTH <= len(text) <= _MAX_LENGTH):
        return False
    if not _PLATE_PATTERN.fullmatch(text):
        return False
    return text[:2] in _VALID_STATE_CODES


def _differing_letter_pair(a: str, b: str) -> frozenset | None:
    diffs = [frozenset({x, y}) for x, y in zip(a, b) if x != y]
    return diffs[0] if len(diffs) == 1 else None


def _correct_state_code(text: str) -> str:
    """If text's first two letters aren't a real state code, but exactly one
    known code is both a single letter away *and* that letter swap is a
    known OCR confusion, use it. Falls back to plain single-letter distance
    only when that alone is unique; otherwise leaves text unchanged (rather
    than guess between multiple equally-plausible corrections).
    """
    prefix, rest = text[:2], text[2:]
    if prefix in _VALID_STATE_CODES:
        return text

    close_matches = [
        code for code in _VALID_STATE_CODES if _differing_letter_pair(code, prefix) is not None
    ]
    plausible_matches = [
        code for code in close_matches if _differing_letter_pair(code, prefix) in _COMMON_CONFUSIONS
    ]
    if len(plausible_matches) == 1:
        return plausible_matches[0] + rest
    if len(close_matches) == 1:
        return close_matches[0] + rest
    return text


def normalize_plate(text: str) -> str | None:
    """Returns a valid plate reading for `text`, correcting common OCR
    artifacts, or None if it can't be made valid.

    Two corrections are tried, independently and combined:
    - A plate crop's top edge sometimes catches a sliver of a sticker/screw/
      state-emblem text above the actual plate, which PaddleOCR concatenates
      onto the real reading as one spurious leading character (observed e.g.
      'STN09CS1812' for an actual 'TN09CS1812').
    - The state-code prefix is corrected against the real list of Indian
      state/UT codes when a single-letter OCR confusion (e.g. 'TH' misread
      for 'TN') is the only plausible explanation (see _correct_state_code).

    Each candidate still has to pass the strict format check, so a
    correction is only applied when it actually produces something
    plate-shaped — this doesn't loosen what counts as a valid plate.
    """
    candidates = [text]
    if len(text) > _MIN_LENGTH:
        candidates.append(text[1:])

    for candidate in candidates:
        corrected = _correct_state_code(candidate)
        if is_valid_plate(corrected):
            return corrected
        if is_valid_plate(candidate):
            return candidate
    return None
