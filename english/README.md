# English

The English base for [Ada](https://docs.glyphh.ai/ada): one record per word
from WordNet 3.0, its synonyms and its neighbours as sets. Any Ada model whose
text roles name this base takes a word's neighbours from here, so "receipt"
scores close to "invoice" with no language model in the path.

This is a hosted Ada model, not a runtime build. It is created from
`spec.json` and loaded from `data/words.jsonl` through `POST /ada/load`.

## Shape

| role | type | weight | holds |
|---|---|---|---|
| `word.identity.lemma` | category, key part | 0 | the word |
| `word.meaning.synonyms` | set | 0.9 | the other words of its synsets |
| `word.meaning.near` | set | 0.6 | words of the synsets it points at: hypernyms, hyponyms, similar, also-see, attributes, pertainyms, derivations |
| `word.meaning.pos` | category | 0.1 | noun, verb, adjective, adverb; also the record's outcome |

When a model's text role names this base, each set role's weight is the
closeness its items get as the word's neighbours, capped at Ada's lexical
ceiling (0.95), so a word is never exactly another word.

77,818 single words. Phrases are left out: a text role scores words.

## Published as `glyphh-ada-eng-1.0`

Glyphh hosts this model for every organization under the catalog row
`glyphh-ada-eng-1.0`, priced at almost nothing. Name it as a base by that row:
`"base": "glyphh-ada-eng-1.0"`. The server's deploy loads a new version from
this folder whenever `version` here changes.

## Load it yourself

```bash
# 1. the model, with an Ada API key from the Glyphh app
curl https://api.glyphh.ai/ada -H "x-glyphh-api-key: Bearer $GLYPHH_API_KEY" -H "content-type: application/json" \
  -d "$(jq -n --slurpfile spec spec.json '{op: "create_model", name: "english", storage: "cloud", spec: $spec[0]}')"
# -> {"data": {"model_id": "am_…", ...}}

# 2. the words, streamed; the first line names the model
{ echo '{"model_id": "am_…"}'; cat data/words.jsonl; } | \
  curl https://api.glyphh.ai/ada/load -H "x-glyphh-api-key: Bearer $GLYPHH_API_KEY" -H "content-type: application/x-ndjson" --data-binary @-
```

Or with the SDK: `ada.createModel({ name: "english", storage: "cloud", spec })`,
then `model.stream(records)` (TypeScript) / `model.stream(records)` (Python).

## Use it

In your own model's spec, a text role of words or an open category names it:

```json
{"name": "subject", "type": "text", "base": "am_…"}
```

A closed category never takes a base: "allow" must never be a little like "deny".

## Rebuild

```bash
python build.py /path/to/wordnet   # the nltk corpus layout (data.noun, data.verb, data.adj, data.adv)
```

## Licence

The model files are MIT. The words, synonyms and neighbours derive from
WordNet 3.0, © Princeton University, under the WordNet licence in
`WORDNET_LICENSE`.
