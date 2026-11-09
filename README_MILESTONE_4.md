# PromoScout — Milestone 4: Single-episode pilot

This milestone adds explicit episode selection and a read-only dry-run to the
semantic analysis CLI. A selected episode is chosen before any Gemini client
is created. Selected results merge into the existing episode-number keyed
analysis and discovery files, preserving results for other episodes. Cache
keys and chunk-level resume behavior are unchanged.

## Prepare the environment

Install project dependencies from the repository's dependency file:

```bash
python -m pip install -r req.txt
```

Verify the Gemini SDK import:

```bash
python -c "from google import genai; print('google-genai import OK')"
```

Set `GEMINI_API_KEY` and `GEMINI_MODEL` in `.env` or the environment. The
project does not choose a model identifier. Before sending screenplay content,
confirm that the selected Gemini account and model are appropriate for the
material and its confidentiality requirements. No API requests run during
installation checks, dry-runs, or tests.

## Pilot episode 1

First inspect the selection, configuration presence, cache, and estimated
request count. This command makes no Gemini requests and does not write analysis
or discovery outputs:

```bash
python -m promoscout.analysis.episode_analyzer \
  data/processed/parsed_all_episodes.json \
  --episode-number 1 \
  --dry-run
```

If the dry-run looks correct and the account is approved for this material,
process only episode 1:

```bash
python -m promoscout.analysis.episode_analyzer \
  data/processed/parsed_all_episodes.json \
  --episode-number 1
```

To analyze the first input episode instead, use `--limit 1`. `--limit` and
`--episode-number` cannot be combined. With neither option, the analyzer keeps
its existing behavior and processes every episode in the input.

Generate only episode 1's candidate windows from its discoveries:

```bash
python -m promoscout.generation.semantic \
  data/processed/parsed_all_episodes.json \
  data/processed/event_discoveries.json \
  --episode-number 1 \
  --output data/outputs/semantic_candidates.json
```

Candidate generation is local and deterministic; it makes no LLM requests.
Inspect the outputs with:

```bash
python -m json.tool data/processed/episode_analysis.json
python -m json.tool data/processed/event_discoveries.json
python -m json.tool data/outputs/semantic_candidates.json
```

Analysis and discovery files are keyed by episode number. Running one episode
updates that episode's entry and preserves other entries. API responses are
cached separately under `data/cache/analysis`; rerunning the same unchanged
episode with the same model and prompt/schema versions can reuse its completed
cache entry.

## Local verification

```bash
python -m pytest -q
python -m compileall -q promoscout tests
python -m promoscout.analysis.episode_analyzer --help
python -m promoscout.generation.semantic --help
```

