import os
import json
from collections import Counter
from typing import List, Tuple, TypedDict, cast
import spacy
from spacy.language import Language


class SpecialTokens(TypedDict):
    unk: str
    pad: str
    sos: str
    eos: str


class Tokenizer:
    """
    Combines:
      • lazy-loaded spaCy tokenizer   (shared across Tokenizer instances)
      • tokenization / vocabulary build / encode / decode / save-load logic
    """

    _spacy_cache: dict[str, Language] = {}  # lang -> nlp object

    _MODEL_NAME = {
        "en": "en_core_web_sm",
        "de": "de_core_news_sm",
        "fr": "fr_core_news_sm",
        "es": "es_core_news_sm",
        # add more languages as needed
    }

    @classmethod
    def _nlp(cls, lang: str) -> Language:
        """Load spaCy pipeline once per language (lazy, thread-safe)."""
        if lang not in cls._spacy_cache:
            model = cls._MODEL_NAME.get(lang)
            if model is None:
                raise ValueError(f"Unsupported language: {lang}")
            try:
                cls._spacy_cache[lang] = spacy.load(model)
            except OSError:
                raise RuntimeError(
                    f"spaCy model '{model}' not found - run:\n"
                    f"    python -m spacy download {model}"
                )
        return cls._spacy_cache[lang]

    @staticmethod
    def _default_specials() -> SpecialTokens:
        return {"unk": "<unk>", "pad": "<pad>", "sos": "<sos>", "eos": "<eos>"}

    @classmethod
    def build(
        cls,
        sentences: list[str],
        language: str,
        specials: SpecialTokens | None = None,
        min_freq: int = 1,
    ) -> "Tokenizer":
        """Create a Tokenizer from raw sentences (training split)."""
        specials = specials or cls._default_specials()
        counter = Counter()

        nlp = cls._nlp(language)
        for s in sentences:
            counter.update(tok.text.lower() for tok in nlp(s))

        vocab = cast(List[str], list(specials.values()))
        vocab += [t for t, f in counter.items() if f >= min_freq and t not in vocab]

        stoi = {tok: i for i, tok in enumerate(vocab)}
        return cls(language, stoi, specials)

    @classmethod
    def load_from_model(cls, path: str) -> Tuple["Tokenizer", "Tokenizer"]:
        """Load a Tokenizer from a model checkpoint."""
        # find any file starting with src in the path
        src_tokenizer_path = None
        trg_tokenizer_path = None
        for file in os.listdir(path):
            if file.startswith("src") and file.endswith(".json"):
                src_tokenizer_path = os.path.join(path, file)
            elif file.startswith("trg") and file.endswith(".json"):
                trg_tokenizer_path = os.path.join(path, file)
        if src_tokenizer_path is None:
            raise ValueError("No source tokenizer found in the model checkpoint.")

        if trg_tokenizer_path is None:
            raise ValueError("No target tokenizer found in the model checkpoint.")

        src_tokenizer = cls.load(src_tokenizer_path)
        trg_tokenizer = cls.load(trg_tokenizer_path)
        return src_tokenizer, trg_tokenizer

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        """Deserialize from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return cls(obj["language"], obj["stoi"], obj["specials"])

    def __init__(self, language: str, stoi: dict[str, int], specials: SpecialTokens):
        self.language = language
        self.stoi = stoi
        self.itos = {i: t for t, i in stoi.items()}

        self.specials = specials
        self.unk_index = stoi[specials["unk"]]
        self.pad_index = stoi[specials["pad"]]
        self.sos_index = stoi[specials["sos"]]
        self.eos_index = stoi[specials["eos"]]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as tokenizer_file:
            json.dump(
                {
                    "language": self.language,
                    "stoi": self.stoi,
                    "specials": self.specials,
                },
                tokenizer_file,
            )

    def encode(self, sentence: str | list[str]) -> list[int]:
        """
        • Accepts raw string **or** pre-tokenised list.
        • Always wraps with <sos> / <eos>.
        """
        if isinstance(sentence, str):
            nlp = self._nlp(self.language)
            tokens = (
                [self.specials["sos"]]
                + [tok.text.lower() for tok in nlp(sentence)]
                + [self.specials["eos"]]
            )
        else:  # list[str]
            tokens = sentence
        return [self.stoi.get(t, self.unk_index) for t in tokens]

    def decode(self, indices: list[int], skip_special: bool = True) -> list[str]:
        return [
            self.itos[idx]
            for idx in indices
            if not skip_special or self.itos[idx] not in self.specials.values()
        ]

    def __len__(self) -> int:
        return len(self.itos)
