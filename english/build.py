"""Build data/words.jsonl from WordNet 3.0's data files.

    python build.py <wordnet dir>      # the nltk corpus layout: data.noun, data.verb, data.adj, data.adv

Single words only, since an Ada text role scores words. Per word: `synonyms`
is every other word in its synsets (weight 0.9 in spec.json), `near` is the
words of the synsets it points at by hypernym, hyponym, similar-to, also-see,
attribute, pertainym and derivational form (weight 0.6), `pos` its part of
speech, which is also the record's outcome. Empty sets are left out: a set
role with nothing in it carries no voice.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

POS = {"n": "noun", "v": "verb", "a": "adjective", "s": "adjective", "r": "adverb"}
NEAR_POINTERS = frozenset({"@", "@i", "~", "~i", "&", "^", "=", "\\", "+"})
FILES = ("data.noun", "data.verb", "data.adj", "data.adv")
MAX_SET = 40


def _word(raw: str) -> str | None:
    w = raw.split("(", 1)[0].lower()
    return w if w and "_" not in w and w.replace("'", "").isalnum() else None


def synsets(wordnet: Path) -> dict:
    out = {}
    for name in FILES:
        for line in (wordnet / name).read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(" ") or not line.strip():
                continue
            fields = line.split("|", 1)[0].split()
            offset, pos, count = fields[0], fields[2], int(fields[3], 16)
            words = [w for w in (_word(fields[4 + 2 * i]) for i in range(count)) if w]
            at = 4 + 2 * count
            pointers = [(fields[at + 1 + 4 * i], fields[at + 2 + 4 * i], fields[at + 3 + 4 * i]) for i in range(int(fields[at]))]
            out[(pos, offset)] = (words, pointers)
    return out


def records(wordnet: Path):
    held = synsets(wordnet)
    syn, near, pos_of = defaultdict(set), defaultdict(set), {}
    for (pos, _), (words, pointers) in held.items():
        linked = {w for symbol, offset, ppos in pointers if symbol in NEAR_POINTERS
                  for w in held.get((ppos if ppos != "s" else "a", offset), ((), ()))[0]}
        for w in words:
            pos_of.setdefault(w, POS[pos])
            syn[w].update(x for x in words if x != w)
            near[w].update(x for x in linked if x != w)
    for w in sorted(pos_of):
        meaning = {"pos": pos_of[w]}
        if syn[w]:
            meaning["synonyms"] = sorted(syn[w])[:MAX_SET]
        if near[w] - syn[w]:
            meaning["near"] = sorted(near[w] - syn[w])[:MAX_SET]
        yield {"situation": {"word": {"identity": {"lemma": w}, "meaning": meaning}}, "outcome": pos_of[w]}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write(__doc__)
        sys.exit(2)
    out = Path(__file__).resolve().parent / "data" / "words.jsonl"
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as f:
        n = sum(1 for r in records(Path(sys.argv[1])) if f.write(json.dumps(r) + "\n"))
    sys.stdout.write(f"{n} words -> {out}\n")
