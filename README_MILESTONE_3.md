# PromoScout — Semantic episode discovery

The analyzer sends each episode's logline, continuity notes, character names,
scene headings, and complete timestamped source events to Gemini. It asks for
narrative analysis and potential hook/ending event references together. Python
validates the structured response and uses the original event records for all
candidate timing. Semantic event IDs are the primary discovery path; the
existing regex generator remains available as a fallback.

## Configuration

Copy `.env.example` to `.env`, set `GEMINI_API_KEY`, and set `GEMINI_MODEL` to
a model identifier available to your account and supporting structured output.
The project intentionally does not select a model identifier for you. The
Google GenAI SDK structured output configuration is based on the official
[Gemini structured output documentation](https://ai.google.dev/gemini-api/docs/structured-output).

## Analyze all parsed episodes

```bash
python -m promoscout.analysis.episode_analyzer \
  data/processed/parsed_all_episodes.json
```

This writes `data/processed/episode_analysis.json` and
`data/processed/event_discoveries.json`. Completed responses are cached under
`data/cache/analysis`; set `PROMOSCOUT_CACHE_DIR` to change the directory.
Episode results are saved after each successful episode, so reruns reuse cache
entries and continue after an interrupted run. The model and prompt/schema
versions are part of cache identity. Oversized episodes are grouped by scene;
oversized individual scenes are split only between complete events.

## Generate windows from semantic discoveries

```bash
python -m promoscout.generation.semantic \
  data/processed/parsed_all_episodes.json \
  data/processed/event_discoveries.json \
  --output data/outputs/semantic_candidates.json
```

For the prior heuristic fallback, use `python -m promoscout.generation.continuous`
with the parsed episodes path. It remains an optional discovery route.

## Tests

```bash
python -m pytest -q
```

Tests use mocked LLM responses and do not call Gemini.
