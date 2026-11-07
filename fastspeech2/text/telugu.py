"""Rule-based Telugu front end.

Telugu orthography is close to phonemic, so instead of a learned G2P we map the
script directly: every consonant carries an inherent /a/ unless it is followed by
a vowel sign (matra) or a virama. Anusvara becomes the nasal that matches the
place of articulation of the next consonant.
"""

import re
import unicodedata

from .base import Frontend

VOWELS = {
    "అ": ["a"], "ఆ": ["aa"], "ఇ": ["i"], "ఈ": ["ii"], "ఉ": ["u"], "ఊ": ["uu"],
    "ఋ": ["r", "u"], "ౠ": ["r", "uu"], "ఎ": ["e"], "ఏ": ["ee"], "ఐ": ["ai"],
    "ఒ": ["o"], "ఓ": ["oo"], "ఔ": ["au"],
}

MATRAS = {
    "ా": ["aa"], "ి": ["i"], "ీ": ["ii"], "ు": ["u"], "ూ": ["uu"], "ృ": ["r", "u"],
    "ౄ": ["r", "uu"], "ె": ["e"], "ే": ["ee"], "ై": ["ai"], "ొ": ["o"], "ో": ["oo"], "ౌ": ["au"],
}

CONSONANTS = {
    "క": "k", "ఖ": "kh", "గ": "g", "ఘ": "gh", "ఙ": "ng",
    "చ": "ch", "ఛ": "chh", "జ": "j", "ఝ": "jh", "ఞ": "nj",
    "ట": "T", "ఠ": "Th", "డ": "D", "ఢ": "Dh", "ణ": "N",
    "త": "t", "థ": "th", "ద": "d", "ధ": "dh", "న": "n",
    "ప": "p", "ఫ": "ph", "బ": "b", "భ": "bh", "మ": "m",
    "య": "y", "ర": "r", "ఱ": "r", "ల": "l", "ళ": "L", "వ": "v",
    "శ": "sh", "ష": "S", "స": "s", "హ": "h", "ౘ": "ch", "ౙ": "j",
}

VIRAMA, ANUSVARA, VISARGA, CANDRABINDU = "్", "ం", "ః", "ఁ"

# nasal used for anusvara, chosen by the following consonant
_NASAL = {}
for chars, nasal in [("కఖగఘ", "ng"), ("చఛజఝ", "nj"), ("టఠడఢ", "N"), ("తథదధ", "n"), ("పఫబభ", "m")]:
    for c in chars:
        _NASAL[c] = nasal

TELUGU_PHONES = sorted({p for v in VOWELS.values() for p in v}
                       | {p for v in MATRAS.values() for p in v}
                       | set(CONSONANTS.values()) | set(_NASAL.values()))

# ---------------- numbers ----------------

ONES = ["సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది",
        "పది", "పదకొండు", "పన్నెండు", "పదమూడు", "పద్నాలుగు", "పదిహేను", "పదహారు", "పదిహేడు",
        "పద్దెనిమిది", "పందొమ్మిది"]
TENS = {2: "ఇరవై", 3: "ముప్పై", 4: "నలభై", 5: "యాభై", 6: "అరవై", 7: "డెబ్బై", 8: "ఎనభై", 9: "తొంభై"}


def number_to_words(n):
    if n < 20:
        return ONES[n]
    if n < 100:
        t, r = divmod(n, 10)
        return TENS[t] + (" " + ONES[r] if r else "")
    if n < 1000:
        h, r = divmod(n, 100)
        if r == 0:
            return "వంద" if h == 1 else ONES[h] + " వందలు"
        return ("నూట" if h == 1 else ONES[h] + " వందల") + " " + number_to_words(r)
    if n < 100000:
        th, r = divmod(n, 1000)
        if r == 0:
            return "వెయ్యి" if th == 1 else number_to_words(th) + " వేలు"
        return ("వెయ్యి" if th == 1 else number_to_words(th) + " వేల") + " " + number_to_words(r)
    if n < 10 ** 7:
        la, r = divmod(n, 100000)
        if r == 0:
            return "లక్ష" if la == 1 else number_to_words(la) + " లక్షలు"
        return ("లక్ష" if la == 1 else number_to_words(la) + " లక్షల") + " " + number_to_words(r)
    return " ".join(ONES[int(d)] for d in str(n))


def ordinal_to_words(n):
    # 12వ -> పన్నెండవ : drop the final vowel sign and add వ
    words = number_to_words(n)
    if words[-1] in MATRAS:
        words = words[:-1]
    return words + "వ"


_TELUGU_DIGITS = str.maketrans("౦౧౨౩౪౫౬౭౮౯", "0123456789")
_time_re = re.compile(r"(\d{1,2}):(\d{2})")
_ordinal_re = re.compile(r"(\d+)వ(?![ఀ-౿])")
_number_re = re.compile(r"\d+")
_token_re = re.compile(r"[ఀ-౿‌‍]+|[,.;:!?।॥]")


def _time(m):
    h, mins = int(m.group(1)), int(m.group(2))
    return number_to_words(h) + ("" if mins == 0 else " " + number_to_words(mins))


class TeluguFrontend(Frontend):
    phones = TELUGU_PHONES

    def normalize(self, text):
        text = unicodedata.normalize("NFC", text).translate(_TELUGU_DIGITS)
        text = _time_re.sub(lambda m: " " + _time(m) + " ", text)
        text = _ordinal_re.sub(lambda m: " " + ordinal_to_words(int(m.group(1))) + " ", text)
        text = _number_re.sub(lambda m: " " + number_to_words(int(m.group(0))) + " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def g2p_words(self, text):
        out = []
        for token in _token_re.findall(text):
            if token[0] in ",.;:!?।॥":
                out.append((token, None))
                continue
            phones = word_to_phones(token)
            if phones:
                out.append((token.replace("‌", "").replace("‍", ""), phones))
        return out


def word_to_phones(word):
    phones = []
    pending = False  # last consonant still waiting for its vowel
    chars = list(word)
    for i, ch in enumerate(chars):
        if ch in CONSONANTS:
            if pending:
                phones.append("a")
            phones.append(CONSONANTS[ch])
            pending = True
        elif ch in MATRAS:
            # a matra with no consonant before it (e.g. after a stray space) is just read as the vowel
            phones.extend(MATRAS[ch])
            pending = False
        elif ch == VIRAMA:
            pending = False
        elif ch in VOWELS:
            if pending:
                phones.append("a")
            phones.extend(VOWELS[ch])
            pending = False
        elif ch == ANUSVARA:
            if pending:
                phones.append("a")
                pending = False
            nxt = chars[i + 1] if i + 1 < len(chars) else ""
            phones.append(_NASAL.get(nxt, "m"))
        elif ch == VISARGA:
            if pending:
                phones.append("a")
                pending = False
            phones.append("h")
        elif ch == CANDRABINDU:
            if pending:
                phones.append("a")
                pending = False
        # anything else (ZWJ/ZWNJ, nukta, stray marks) is ignored
    if pending:
        phones.append("a")
    return phones
