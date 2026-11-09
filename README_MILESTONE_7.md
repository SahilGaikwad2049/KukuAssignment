# PromoScout Milestone 7 — Multi-Episode Blind-Viewer Pilot

## Candidate sampling

Sampling is deterministic for a fixed candidate artifact and seed. The available episode numbers are sorted and divided into equal-width strata; one episode is selected from each stratum using the seeded random generator. This spreads a 10-episode pilot across the available episode range rather than clustering it at the beginning.

By default, one candidate is selected from each sampled episode. If two are requested, the within-episode choice uses only metadata to prefer a new hook type and duration band where possible. It never looks at heuristic scores, prior blind-viewer scores, or verdicts. Identical episode/time windows and duplicate source/candidate IDs are removed before selection. If fewer distinct episodes are available than requested, the CLI reports both counts.

Candidate sources stay separate by default. `--source heuristic` and `--source semantic` each draw from one source. `--source both` is an explicit opt-in to a mixed pool; each selected row and result still records its source. In the current local outputs, heuristic candidates are available and semantic candidates are absent, so the first sample should use heuristic candidates.

## Dry-run and manifest

Run the requested 10-episode sample without sending requests:

```bash
python -m promoscout.evaluation.blind_viewer --source heuristic --sample-episodes 10 --candidates-per-episode 1 --sample-seed 42 --dry-run
```

The dry-run prints selected episode numbers, candidate IDs and sources, cache hits, required requests, and model ID. It does not construct a Gemini client, make requests, or modify blind-viewer evaluation output. It writes the exact sample and each selection rationale to:

`data/outputs/blind_viewer_pilot_selection.json`

Inspect that manifest before deciding to run the live command. For a 20-candidate sample:

```bash
python -m promoscout.evaluation.blind_viewer --source heuristic --sample-episodes 10 --candidates-per-episode 2 --sample-seed 42 --dry-run
```

To repeat the experiment on semantic candidates later, use `--source semantic`. The sample is selected independently of evaluation results.

## Live pilot execution

After reviewing the manifest and configuring `GEMINI_API_KEY` and `GEMINI_MODEL` locally, explicitly run:

```bash
python -m promoscout.evaluation.blind_viewer --source heuristic --sample-episodes 10 --candidates-per-episode 1 --sample-seed 42
```

The command evaluates candidates sequentially. It reuses valid content/model/prompt/schema cache entries, writes every successful result to `data/outputs/blind_viewer_evaluations.json` immediately, preserves previous evaluations, and continues after an individual candidate failure. Transient errors use bounded exponential backoff with jitter and honor a `Retry-After` header; permanent request errors such as HTTP 400 are not retried.

After the run, inspect:

- `data/outputs/blind_viewer_pilot_selection.json` — exact candidate sample and rationale.
- `data/outputs/blind_viewer_evaluations.json` — saved AI assessments.
- `data/outputs/blind_viewer_pilot_report.json` — machine-readable aggregate and episode-level results.
- `data/outputs/blind_viewer_pilot_report.md` — readable summary, score distributions, verdict counts, qualitative observations, failures, and separate human decisions.

The report includes average, median, minimum, maximum and 1–5 score counts for each of the six dimensions. Qualitative observations are derived from saved assessment explanations and missing-context/confusion fields. Results are descriptive; a small sample is not statistically significant. AI verdicts are not approvals. Human decisions remain separate in `review_decisions.json` and are surfaced independently in Candidate Review and Rankings.

## Frontend

Candidate Review and Rankings show pilot status, source/model, represented episodes, cache reuse, score distributions, and a downloadable selection manifest/report when those files exist. Candidate cards continue to show individual blind-viewer verdicts and six scores. The Exports page also provides the pilot report and manifest.

## Limitations

The report reflects LLM judgments based only on parsed clip events; it is not a recruited viewer study and cannot judge the final edited video, acting, pacing, or sound mix beyond screenplay descriptions. Sampling can only use episodes and candidates present in the selected local artifact. Failed requests appear in the report but are not treated as quality scores.
