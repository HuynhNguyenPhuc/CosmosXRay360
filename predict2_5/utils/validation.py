"""Validation and format checking utilities."""

import re

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
)


def is_uuid_format(text: str) -> bool:
    """
    Check whether text matches canonical UUID format.

    Args:
        text: String to validate.

    Returns:
        True if text is a valid UUID, False otherwise.
    """
    return bool(_UUID_RE.fullmatch(text.strip()))
