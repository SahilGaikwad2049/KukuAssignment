# PromoScout Milestone 8 — Episode 1 Semantic Discovery Pilot

## Purpose

This controlled experiment compares independently discovered semantic candidates with the existing heuristic candidates for Episode 1. Semantic discovery receives the parsed screenplay events directly. It does not receive heuristic candidate windows or use them to choose what Gemini inspects. The model returns original event IDs; Python retrieves the source events and constructs deterministic continuous promo windows with their original timestamps.

The blind-viewer evaluator remains a separate step. It receives only chronological screenplay events inside a candidate's clip window, with relative clip times. It does not receive episode summaries, relationship descriptions, semantic discovery rationales, or other series context. A saved AI verdict is not a human approval.

## Current local readiness

- Parsed source: `data/processed/parsed_all_episodes.json`
- Episode 1: 95 original screenplay events, including dialogue and speakers, actions, camera directions, sound events where present, scene heading, timestamps, and original IDs.
- Existing heuristic candidates: `data/outputs/continuous_candidates.json` (Episode 1 has 25 candidates).
- Saved heuristic blind-viewer evaluations include `EP001_PROMO_001` and `EP001_PROMO_011`; valid saved results are reused in the comparison.
- Semantic analysis, discovery, and candidate files are currently absent. The dry-run estimated four requests (three event chunks and one aggregation request). No live request has been made by this implementation.

## Semantic analysis dry-run

The Episode 1 dry-run was run as:

```bash
python -m promoscout.analysis.episode_analyzer data/processed/parsed_all_episodes.json --episode-number 1 --dry-run
```

It reported model `gemini-3.8-flash`, 95 events, no completed episode cache, three chunks, four estimated requests, and Episode 1 as the only selected episode. It did not initialize Gemini or write completed analysis outputs. Review this output and confirm that your configured provider is appropriate for the confidential screenplay material before running the live command.

## Manual pilot sequence

Run one step at a time and inspect the resulting local files.

1. Analyze and discover Episode 1 from its original parsed screenplay events:

```bash
python -m promoscout.analysis.episode_analyzer data/processed/parsed_all_episodes.json --episode-number 1
```

This produces/merges `data/processed/episode_analysis.json` and `data/processed/event_discoveries.json`, preserving other episode entries. Chunks and completed episodes use the existing content/model/prompt/schema-aware cache.

2. Generate continuous semantic candidates from those original event references:

```bash
python -m promoscout.generation.semantic data/processed/parsed_all_episodes.json data/processed/event_discoveries.json --episode-number 1 --output data/outputs/semantic_candidates.json
```

This writes a separate semantic candidate artifact. It does not overwrite `continuous_candidates.json` or make an LLM call. Candidate windows are rebuilt in Python; invalid source IDs are rejected, source timestamps retained, overlapping events included, duration constrained to 20–60 seconds, hook lead-in limited to three seconds, duplicate windows removed, and candidate families deterministic.

3. Run structural validation for Episode 1:

```bash
python -m promoscout.evaluation.pilot_validator --episode-number 1
```

Review `data/outputs/pilot_validation.json` and `data/outputs/pilot_validation.md`. Structural validity confirms artifact/timing/source integrity only; it does not establish promotional quality or a genuine cliffhanger.

4. Inspect every semantic candidate preview before requesting blind-viewer evaluations. Run `streamlit run app.py`, open **Candidate Review**, choose Episode 1 and the semantic source, and inspect each chronological preview. The preview is reconstructed from the parsed source events.

5. Preview a deterministic sample of up to three Episode 1 semantic candidates. The selection uses available hook, family, and duration metadata, not quality scores or prior blind-viewer results:

```bash
python -m promoscout.evaluation.blind_viewer --source semantic --candidates data/outputs/semantic_candidates.json --sample-episodes 1 --candidates-per-episode 3 --sample-seed 42 --selection-manifest data/outputs/semantic_episode1_selection.json --pilot-report-json data/outputs/semantic_episode1_blind_viewer_pilot_report.json --pilot-report-markdown data/outputs/semantic_episode1_blind_viewer_pilot_report.md --dry-run
```

Inspect `data/outputs/semantic_episode1_selection.json`. The dry-run makes no Gemini request. It prints the selected IDs, cache status, model, and estimated requests.

6. Only after reviewing the manifest and confirming local Gemini configuration, execute the same command without `--dry-run`:

```bash
python -m promoscout.evaluation.blind_viewer --source semantic --candidates data/outputs/semantic_candidates.json --sample-episodes 1 --candidates-per-episode 3 --sample-seed 42 --selection-manifest data/outputs/semantic_episode1_selection.json --pilot-report-json data/outputs/semantic_episode1_blind_viewer_pilot_report.json --pilot-report-markdown data/outputs/semantic_episode1_blind_viewer_pilot_report.md
```

This evaluates only the explicit semantic sample, sequentially. Valid cached results are reused and each successful evaluation is saved incrementally. Failed candidates are reported separately; successful prior results are preserved. The default evaluation output is `data/outputs/blind_viewer_evaluations.json` and is merged without deleting existing evaluations.

7. Once semantic candidates and evaluations are available, generate the offline comparison:

```bash
python -m promoscout.evaluation.discovery_comparison --episode-number 1
```

This creates `data/outputs/discovery_comparison.json` and `data/outputs/discovery_comparison.md`. It compares candidate counts, unique windows, hook types, duration distributions, families, structural validity, and only saved blind-viewer evaluations that pass candidate/source/episode, reconstructed clip, and source-evidence checks. Failed or invalid evaluation results are excluded, not counted as weak. Human review decisions are displayed separately. The Streamlit Exports page offers both comparison files when they exist.

## Interpreting the comparison

The report identifies semantic discoveries not used as heuristic endpoints, heuristic endpoints not chosen by semantic discovery, and shared source events. It provides candidate-level scores and descriptive averages for standalone clarity and ending effectiveness when valid evaluations exist. It does not claim that one method is better because it uses an LLM. A small sample of two or three semantic candidates from one episode is exploratory and cannot establish statistical superiority. The blind-viewer model is itself an AI evaluator; review the source previews and make human decisions separately.

## Tests and boundaries

Tests use local fixtures and mocked LLM outputs; they make no Gemini requests. The analysis command is scoped with `--episode-number 1`. Do not omit the episode selection when running this controlled pilot. No episode data is modified and the heuristic candidate output remains separate.

The live semantic discovery and sampled evaluation still require an explicit manual run after reviewing the dry-run and deciding that the configured model/provider may receive this screenplay content. This implementation does not automatically send screenplay material to an external service.
