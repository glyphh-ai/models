"""Build data/words.jsonl from the Moby Part-of-Speech list.

    python build.py <mobypos.txt>      # Grady Ward's Moby Part-of-Speech list, Mac Roman encoded

Single words only, since an Ada text role scores words. Per word: `pos` is the
set of its parts of speech (weight 0.9 in spec.json), `class` its place as a
function word or a content word (weight 0.1), and the record's outcome is its
primary part of speech: its function-word code when it has one, else the
first code the list gives it. Forms are
lower-cased; entries that fall together keep the union of their codes.
"""

import json
import sys
from pathlib import Path

POS = {"N": "noun", "p": "plural", "h": "noun_phrase", "V": "verb", "t": "transitive_verb",
       "i": "intransitive_verb", "A": "adjective", "v": "adverb", "C": "conjunction",
       "P": "preposition", "!": "interjection", "r": "pronoun", "D": "definite_article",
       "I": "indefinite_article", "o": "nominative"}
CLASSES = (("determiner", "DI"), ("pronoun", "ro"), ("preposition", "P"), ("conjunction", "C"), ("interjection", "!"))


def entries(moby: Path):
    for line in moby.read_text(encoding="mac_roman").splitlines():
        form, _, codes = line.rpartition("\\")
        codes = [c for c in codes if c in POS]
        if form and " " not in form and codes:
            yield form.lower(), codes


def word_class(codes: set[str]) -> str:
    return next((name for name, marks in CLASSES if codes & set(marks)), "content")


def primary(codes: list[str]) -> str:
    """A function word's outcome is its function code; a content word's is the first code the list gives it."""
    return next((c for c in codes if any(c in marks for _, marks in CLASSES)), codes[0])


def records(moby: Path):
    codes_of = {}
    for form, codes in entries(moby):
        held = codes_of.setdefault(form, [])
        held.extend(c for c in codes if c not in held)
    for form in sorted(codes_of):
        codes = codes_of[form]
        grammar = {"pos": sorted({POS[c] for c in codes}), "class": word_class(set(codes))}
        yield {"situation": {"word": {"identity": {"form": form}, "grammar": grammar}}, "outcome": POS[primary(codes)]}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write(__doc__)
        sys.exit(2)
    out = Path(__file__).resolve().parent / "data" / "words.jsonl"
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as f:
        n = sum(1 for r in records(Path(sys.argv[1])) if f.write(json.dumps(r) + "\n"))
    sys.stdout.write(f"{n} words -> {out}\n")
