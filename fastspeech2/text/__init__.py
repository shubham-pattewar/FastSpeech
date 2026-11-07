from .base import PAUSE_PUNCT, SILENCES, Frontend, make_symbols


def get_frontend(language):
    language = language.lower()
    if language == "english":
        from .english import EnglishFrontend
        return EnglishFrontend()
    if language == "telugu":
        from .telugu import TeluguFrontend
        return TeluguFrontend()
    raise ValueError(f"unsupported language: {language}")


def get_symbols(language):
    """Symbol table without building the full front end (no g2p model load)."""
    language = language.lower()
    if language == "english":
        from .english import ARPABET as phones
    elif language == "telugu":
        from .telugu import TELUGU_PHONES as phones
    else:
        raise ValueError(f"unsupported language: {language}")
    return make_symbols(phones)
