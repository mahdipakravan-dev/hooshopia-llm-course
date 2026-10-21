"""Persian text normalization.

The same word can be written with different Unicode characters (Arabic yeh vs Persian yeh,
Arabic digits vs Persian digits, with or without diacritics or tatweel). For the model these
are different tokens, so we map every variant to one canonical form before anything else.
"""
import re
import unicodedata

# Arabic letters that look like Persian ones but have different code points.
_CHAR_MAP = str.maketrans({
    "ي": "ی",  # ي Arabic yeh        -> ی
    "ى": "ی",  # ى alef maksura      -> ی
    "ك": "ک",  # ك Arabic kaf        -> ک
    "ة": "ه",  # ة teh marbuta       -> ه
    "٠": "۰", "١": "۱", "٢": "۲", "٣": "۳", "٤": "۴",
    "٥": "۵", "٦": "۶", "٧": "۷", "٨": "۸", "٩": "۹",
    " ": " ", "\t": " ",  # no-break space and tab -> space
})

ZWNJ = "‌"  # نیم‌فاصله

_DIACRITICS = re.compile("[ً-ْٰ]")  # harakat, tanwin, shadda, sukun, superscript alef
_TATWEEL = re.compile("ـ+")                  # ـــ
_INVISIBLE = re.compile("[​‎‏‪-‮⁦-⁩﻿]")  # bidi marks, BOM, ZWSP
_ZWNJ_RUN = re.compile(ZWNJ + "{2,}")
_ZWNJ_EDGE = re.compile(rf"\s*{ZWNJ}+\s+|\s+{ZWNJ}+\s*|^{ZWNJ}+|{ZWNJ}+$", re.M)
_SPACES = re.compile(r"[ ]{2,}")
_NEWLINES = re.compile(r"\n{3,}")


def normalize(text: str) -> str:
    """Return the canonical form of a Persian text."""
    text = unicodedata.normalize("NFKC", text)  # Arabic presentation forms (ﻻ ﮐ ...) -> base letters
    text = _INVISIBLE.sub("", text)
    text = text.translate(_CHAR_MAP)
    text = _DIACRITICS.sub("", text)
    text = _TATWEEL.sub("", text)
    text = _ZWNJ_RUN.sub(ZWNJ, text)
    text = _ZWNJ_EDGE.sub(" ", text)  # a ZWNJ next to a space is always a typo
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _NEWLINES.sub("\n\n", text)
    return text.strip()


def _is_persian_letter(ch: str) -> bool:
    cp = ord(ch)
    return 0x0600 <= cp <= 0x06FF or 0xFB50 <= cp <= 0xFDFF or 0xFE70 <= cp <= 0xFEFF


def persian_ratio(text: str) -> float:
    """Share of letters that are Persian/Arabic script (0.0 to 1.0). Digits and symbols are ignored."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(_is_persian_letter(c) for c in letters) / len(letters)
