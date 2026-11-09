# PromoScout — Milestone 2: Continuous candidate generation

This milestone creates provisional promo windows with deterministic Python
heuristics. It does not use Gemini and does not claim that candidates are good
promos. The next evaluation stage will judge their narrative quality.

## Run

From the outer project directory (the folder named `Kuku` in this setup), with
`.venv` activated:

```bash
python -m promoscout.generation.continuous \
  data/processed/parsed_all_episodes.json \
  --output data/outputs/continuous_candidates.json
```

Optional parameters:

```bash
python -m promoscout.generation.continuous \
  data/processed/parsed_all_episodes.json \
  --output data/outputs/continuous_candidates.json \
  --min-duration 20 \
  --max-duration 60 \
  --max-per-episode 25
```

## Tests

```bash
python -m pytest tests/test_candidates.py -v
```

Run the parser and candidate tests together:

```bash
python -m pytest tests/ -v
```

## Candidate data

Each candidate records:

- Start and end source-event IDs and exact original timestamps.
- Hook and ending event IDs and their original text.
- Deterministic heuristic scores and matched signals.
- IDs for every screenplay event overlapping the candidate window.

The algorithm may use events from multiple scenes in the same episode. It does
not combine different episodes; cross-episode composites are a later stage.

## Important limitations

The hook and ending rules are intentionally simple and can produce false
positives or miss subtle moments. A high heuristic score is not a final promo
score. Do not present this ranking as the final ranking for Kuku. The LLM
promotional evaluator, blind-viewer QA, pairwise ranking, and human review are
still required.
