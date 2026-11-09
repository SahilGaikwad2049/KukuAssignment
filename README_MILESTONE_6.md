# PromoScout Milestone 6 — Blind-Viewer Evaluation

## Purpose and independence

The blind-viewer evaluator judges each candidate as a first-time viewer. It makes one independent structured-output request per candidate and does not reuse the episode analyzer or event-discovery rationales.

The model receives only a clip duration and the chronologically ordered screenplay events overlapping that candidate window. Each event contains its original event ID, type, content, clip-relative start/end times, and available speaker/delivery fields. The evaluator reconstructs this input from the original parsed episode, rather than trusting a candidate's stored preview, so every source event overlapping the window is included.

The prompt/input deliberately excludes episode number, absolute episode timestamps, scene headings, logline, continuity notes, episode or series summaries, character biographies/relationships, events outside the window, candidate hook/ending labels and text, discovery explanations, pipeline scores, and reviewer notes. Event IDs remain present for evidence validation. No frontend or evaluator code sends screenplay data anywhere except the explicitly configured LLM request during a user-invoked evaluation run.

## Evaluation dimensions and output

The Pydantic assessment separately scores standalone clarity, hook effectiveness, conflict comprehension, stakes clarity, cliffhanger effectiveness, and desire to continue on a 1–5 scale. Each score needs at least one source-grounded evidence reference. The result also records the viewer's 1–3 sentence reconstruction, perceived characters/roles, direct statements, reasonable inferences, unclear points, conflict, stakes, unanswered question, confusing elements, missing context, confidence, and a strong/mixed/weak verdict. The verdict is a reasoned judgment, not an average of the six scores.

Python rejects evidence event IDs outside the reconstructed clip and quotations that are not exact substrings of their cited source event. The evaluator does not see any preselected hook or ending label and does not treat an abrupt cut as a compelling cliffhanger by itself.

## Cache and resume behavior

Successful responses are stored in the existing content-addressed JSON cache under `data/cache/analysis/blind_viewer/` (or `PROMOSCOUT_CACHE_DIR/blind_viewer`). A key includes the clip content, candidate ID and origin, configured model identifier, prompt version, and schema version. A prompt/schema/model/content change therefore causes a cache miss. Successful candidates are merged and atomically written to the output immediately; a later candidate failure does not remove prior evaluations. Processing is sequential, and valid cache hits do not cause LLM requests.

The final output defaults to `data/outputs/blind_viewer_evaluations.json`, in an `evaluations` object keyed by `semantic:<candidate_id>` or `heuristic:<candidate_id>`. Existing entries are preserved when selected candidates are processed. Semantic candidates are the default source. Heuristic evaluation requires `--source heuristic`; `--source both` is also available.

## Commands

Run a no-request preview for episode 1:

```bash
python -m promoscout.evaluation.blind_viewer --episode-number 1 --dry-run
```

Evaluate a single semantic candidate (replace the ID with one in `semantic_candidates.json`):

```bash
python -m promoscout.evaluation.blind_viewer --episode-number 1 --candidate-id EP001_PROMO_001
```

Limit the selected semantic candidates:

```bash
python -m promoscout.evaluation.blind_viewer --episode-number 1 --limit 3 --dry-run
```

Evaluate a heuristic candidate explicitly:

```bash
python -m promoscout.evaluation.blind_viewer --source heuristic --episode-number 1 --candidate-id EP001_PROMO_001
```

Custom parsed input, candidate file, and result path can be supplied with `--parsed`, `--candidates`, and `--output`. Before a live run, set `GEMINI_API_KEY` and `GEMINI_MODEL` in `.env` or the environment. Dry-run does not construct the Gemini client or make API requests.

## Limitations

This is an LLM judgment grounded in clip text, not a study with recruited viewers. The screenplay may omit visual/audio details that were not parsed, and the evaluator cannot judge the rendered edit, performance, pacing, or sound mix beyond the supplied event descriptions. Results should be treated as an evaluation aid and reviewed by a human. The feature does not implement pairwise ranking or composite promos.
