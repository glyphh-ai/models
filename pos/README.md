# Parts of speech

The parts-of-speech base for [Ada](https://docs.glyphh.ai/ada): one record per
English word form from the Moby Part-of-Speech list, its parts of speech as a
set and its place as a function word or a content word. Any Ada model whose
text roles name this base reads a word's grammatical place from here, so
"the" sits with "a" and "running" with "refused", with no language model in
the path.

This is a hosted Ada model, not a runtime build. It is created from
`spec.json` and loaded from `data/words.jsonl` through `POST /ada/load`.

## Shape

| role | type | weight | holds |
|---|---|---|---|
| `word.identity.form` | category, key part | 0 | the word form, lower-cased |
| `word.grammar.pos` | set | 0.9 | its parts of speech: noun, plural, noun_phrase, verb, transitive_verb, intransitive_verb, adjective, adverb, conjunction, preposition, interjection, pronoun, definite_article, indefinite_article, nominative |
| `word.grammar.class` | category | 0.1 | determiner, pronoun, preposition, conjunction, interjection, or content; a function-word code wins over a content code |

The record's outcome is the word's primary part of speech: its function-word
code when it has one, else the first one the list gives it. When a model's text role names this base, each set role's
weight is the closeness its items get as the word's neighbours, capped at
Ada's lexical ceiling (0.95), so a word is never exactly another word.

192,960 single word forms. Phrases are left out: a text role scores words.

## Published as `glyphh-ada-pos-1.0`

Glyphh hosts this model for every organization under the catalog row
`glyphh-ada-pos-1.0`, priced at almost nothing. Name it as a base by that row:
`"base": "glyphh-ada-pos-1.0"`. The server's deploy loads a new version from
this folder whenever `version` here changes.

## Load it yourself

```bash
# 1. the model, with an Ada API key from the Glyphh app
curl https://api.glyphh.ai/ada -H "x-glyphh-api-key: Bearer $GLYPHH_API_KEY" -H "content-type: application/json" \
  -d "$(jq -n --slurpfile spec spec.json '{op: "create_model", name: "pos", spec: $spec[0]}')"
# -> {"data": {"model_id": "am_…", ...}}

# 2. the words, streamed; the first line names the model
{ echo '{"model_id": "am_…"}'; cat data/words.jsonl; } | \
  curl https://api.glyphh.ai/ada/load -H "x-glyphh-api-key: Bearer $GLYPHH_API_KEY" -H "content-type: application/x-ndjson" --data-binary @-
```

Or with the SDK: `ada.createModel({ name: "pos", spec })`, then
`model.stream(records)` (TypeScript) / `model.stream(records)` (Python).

## Use it

In your own model's spec, a text role of words or an open category names it:

```json
{"name": "subject", "type": "text", "base": "glyphh-ada-pos-1.0"}
```

A closed category never takes a base: "allow" must never be a little like "deny".

## Rebuild

```bash
python build.py /path/to/mobypos.txt   # the Moby Part-of-Speech list, Mac Roman encoded
```

## Licence

The model files are MIT. The word list is the Moby Part-of-Speech list by
Grady Ward, released into the public domain; see `MOBY_LICENSE`.
