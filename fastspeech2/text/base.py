PAD = "_"
SILENCES = ["sp", "spn"]

# punctuation that becomes a short pause ("sp") at inference time
PAUSE_PUNCT = set(",.;:!?।॥")


def make_symbols(phones):
    return [PAD] + SILENCES + sorted(phones)


class Frontend:
    """Text -> words -> phonemes. Subclasses implement `normalize` and `g2p_words`.

    `g2p_words` returns a list of (token, phones) pairs where phones is None for
    punctuation. The same pronunciations are written into the MFA lexicon, so
    whatever phonemes the aligner sees during training are the ones we feed the
    model at inference.
    """

    phones = []

    def __init__(self):
        self.symbols = make_symbols(self.phones)
        self.symbol_to_id = {s: i for i, s in enumerate(self.symbols)}

    def normalize(self, text):
        raise NotImplementedError

    def g2p_words(self, text):
        raise NotImplementedError

    def words(self, text):
        """Normalised words with their pronunciations (punctuation dropped)."""
        return [(w, p) for w, p in self.g2p_words(self.normalize(text)) if p]

    def text_to_phones(self, text):
        out = []
        for token, phones in self.g2p_words(self.normalize(text)):
            if phones:
                out.extend(phones)
            elif token in PAUSE_PUNCT and out and out[-1] != "sp":
                out.append("sp")
        while out and out[-1] == "sp":
            out.pop()
        return out

    def phones_to_ids(self, phones):
        unknown = sorted({p for p in phones if p not in self.symbol_to_id})
        if unknown:
            raise ValueError(f"phonemes not in the symbol set: {unknown}")
        return [self.symbol_to_id[p] for p in phones]
