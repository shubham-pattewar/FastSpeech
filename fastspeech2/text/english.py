import re
import unicodedata

from .base import Frontend

_CONSONANTS = "B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH".split()
_VOWELS = "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split()
ARPABET = _CONSONANTS + [v + s for v in _VOWELS for s in ("0", "1", "2")]

MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]

ABBREVIATIONS = {
    "mr": "mister", "mrs": "missus", "dr": "doctor", "prof": "professor",
    "vs": "versus", "etc": "et cetera",
}

# letter names, for spelled-out tokens such as "A M" or "U S"
LETTERS = {
    "A": "EY1", "B": "B IY1", "C": "S IY1", "D": "D IY1", "E": "IY1", "F": "EH1 F",
    "G": "JH IY1", "H": "EY1 CH", "I": "AY1", "J": "JH EY1", "K": "K EY1", "L": "EH1 L",
    "M": "EH1 M", "N": "EH1 N", "O": "OW1", "P": "P IY1", "Q": "K Y UW1", "R": "AA1 R",
    "S": "EH1 S", "T": "T IY1", "U": "Y UW1", "V": "V IY1", "W": "D AH1 B AH0 L Y UW0",
    "X": "EH1 K S", "Y": "W AY1", "Z": "Z IY1",
}

_month_re = "|".join(m.capitalize() for m in MONTHS)
_time_re = re.compile(r"\b(\d{1,2}):(\d{2})\s*([AaPp])\.?\s*[Mm]\b\.?|\b(\d{1,2}):(\d{2})\b")
_day_month_re = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_month_re})\b", re.IGNORECASE)
_month_day_re = re.compile(rf"\b({_month_re})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", re.IGNORECASE)
_year_re = re.compile(r"\b(1[1-9]\d\d|20\d\d)\b")
_ordinal_re = re.compile(r"\b(\d+)(st|nd|rd|th)\b")
_number_re = re.compile(r"\d+(?:\.\d+)?")
_abbrev_re = re.compile(r"\b(Mr|Mrs|Dr|Prof|vs|etc)\.", re.IGNORECASE)
_token_re = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*|[,.;:!?]")


class EnglishFrontend(Frontend):
    phones = ARPABET

    def __init__(self):
        super().__init__()
        _ensure_nltk()  # before importing g2p_en, which tries to download on import
        import inflect
        from g2p_en import G2p

        self.inflect = inflect.engine()
        self.g2p = G2p()

    # ---------------- normalisation ----------------

    def _num(self, n):
        words = self.inflect.number_to_words(n, andword="")
        return words.replace("-", " ").replace(",", "")

    def _ordinal(self, n):
        return self._num(self.inflect.ordinal(int(n)))

    def _year(self, y):
        if 2000 <= y < 2010 or y % 1000 == 0:
            return self._num(y)
        hi, lo = divmod(y, 100)
        if lo == 0:
            return self._num(hi) + " hundred"
        return self._num(hi) + (" oh " if lo < 10 else " ") + self._num(lo)

    def _time(self, m):
        if m.group(1):
            h, mins, ampm = int(m.group(1)), int(m.group(2)), m.group(3).upper()
        else:
            h, mins, ampm = int(m.group(4)), int(m.group(5)), None
        words = self._num(h)
        if mins == 0:
            words += " o'clock" if not ampm else ""
        else:
            words += (" oh " if mins < 10 else " ") + self._num(mins)
        if ampm:
            words += f" {ampm}M"  # kept upper case so it gets spelled out as letters
        return " " + words + " "

    def normalize(self, text):
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
        text = text.replace("&", " and ").replace("%", " percent ")
        text = _time_re.sub(self._time, text)
        text = _day_month_re.sub(lambda m: f"{self._ordinal(m.group(1))} {m.group(2)}", text)
        text = _month_day_re.sub(lambda m: f"{m.group(1)} {self._ordinal(m.group(2))}", text)
        text = _year_re.sub(lambda m: self._year(int(m.group(1))), text)
        text = _ordinal_re.sub(lambda m: self._ordinal(m.group(1)), text)
        text = _number_re.sub(lambda m: " " + self._num(m.group(0)) + " ", text)
        text = _abbrev_re.sub(lambda m: ABBREVIATIONS[m.group(1).lower()], text)
        return re.sub(r"\s+", " ", text).strip()

    # ---------------- g2p ----------------

    def _pron(self, word, pos):
        # same lookup order as g2p_en: homographs (by POS) -> CMUdict -> neural model
        g = self.g2p
        if word in g.homograph2features:
            pron1, pron2, pos1 = g.homograph2features[word]
            return pron1 if pos.startswith(pos1) else pron2
        if word in g.cmu:
            return g.cmu[word][0]
        return g.predict(word)

    def g2p_words(self, text):
        import nltk

        tokens = _token_re.findall(text)
        if not tokens:
            return []
        out = []
        for token, pos in nltk.pos_tag(tokens):
            if not token[0].isalpha():
                out.append((token, None))
            elif token.isupper() and len(token) <= 3 and token not in ("I", "A"):
                # short all-caps tokens (AM, PM, US ...) are read letter by letter
                for ch in token:
                    out.append((ch.lower(), LETTERS[ch].split()))
            else:
                word = token.lower()
                out.append((word, list(self._pron(word, pos))))
        return out


def _ensure_nltk():
    import nltk

    # nltk >= 3.9 renamed the tagger; g2p_en still asks for the old name
    for res, path in [("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
                      ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
                      ("cmudict", "corpora/cmudict")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(res, quiet=True)
