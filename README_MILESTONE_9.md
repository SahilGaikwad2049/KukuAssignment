# PromoScout Milestone 9 — Series Intelligence and Composite Promos

## Architecture

Series analysis is hierarchical. It consumes parsed episode records, existing validated episode summaries, and episode discovery references. Missing episode summaries are generated through the existing episode analyzer, which chunks scenes/events and saves completed work to its existing incremental outputs. Series synthesis then groups episode material under the configured context limit, caches each successful chunk, and performs a final cross-chunk synthesis. Cache keys include screenplay/analysis content, model ID, prompt content/version, and schema version. The model ID comes from `GEMINI_MODEL`; no fallback model is selected.

The Pydantic series schemas represent canonical characters and aliases, sourced relationships and relationship changes, story arcs and developments, unresolved questions, chronology, continuity uncertainties, spoiler-sensitive developments, and composite promo opportunities. Narrative developments and composite beats cite `(episode_number, event_id)` references. Python validates every reference against the original parsed data and rejects unknown IDs. The model does not produce timestamps.

Composite generation is deterministic Python construction from the source event anchors selected in `series_analysis.json`. It allocates a target duration across contiguous source windows, retrieves all events overlapping each window, preserves original event timestamps and IDs, computes promo-relative times, and requires at least two episodes and 20–60 seconds total. Story connections and spoiler judgments remain model claims requiring review; uncertain connections and spoiler levels are surfaced as warnings. The generator does not call Gemini.

## Dry-runs

Series dry-run checks all 50 available episodes but does not process them, initialize Gemini, create cache directories, or write outputs:

```bash
python -m promoscout.analysis.series_analyzer --dry-run
```

To inspect a controlled three-episode scope:

```bash
python -m promoscout.analysis.series_analyzer --limit 3 --dry-run
```

The report shows episode counts, existing/missing analyses, series cache status, current composite opportunity count, configured model, and estimated request count. The current local data has 50 parsed episodes and no completed episode or series analyses. Its all-episode dry-run reports 0 existing analyses, 50 missing, a series cache miss, 0 composite opportunities, model `gemini-3.8-flash`, and an estimated request count. A dry-run does not submit any screenplay content.

Composite generation dry-run is also local and read-only:

```bash
python -m promoscout.generation.composite --dry-run
```

## Explicit execution commands

No live Gemini calls were made for this milestone. Review the estimated request count and your provider/data-sharing requirements before manually running analysis.

For development on a small subset:

```bash
python -m promoscout.analysis.series_analyzer --limit 3
```

For full 50-episode analysis, explicitly opt in:

```bash
python -m promoscout.analysis.series_analyzer --all-episodes
```

Successful episode results and series chunks remain cached if a request fails. Rerun the same command to resume. Outputs are:

- `data/processed/episode_analysis.json`
- `data/processed/event_discoveries.json`
- `data/processed/series_analysis.json`
- `data/processed/series_analysis.md`

Before constructing a composite, inspect series opportunities and their source event IDs. Then run:

```bash
python -m promoscout.generation.composite --dry-run
python -m promoscout.generation.composite --duration 30
python -m promoscout.evaluation.pilot_validator
```

Candidate output is written separately to `data/outputs/composite_candidates.json`; existing episode candidate files are not replaced. Structural validation includes composite source episodes, windows, overlapping source events, durations, relative timing, and hook/ending anchors. It does not prove narrative coherence or promotional quality.

## Blind-viewer evaluation

The evaluator reconstructs each segment from original parsed events, concatenates the segments, replaces event IDs with anonymous `CLIP_EVENT_####` IDs, and sends only event type, spoken/action/camera/sound content, speakers/delivery when present, relative promo times, and total clip duration. It does not send episode numbers, series synopsis, character biographies, story arcs, source timestamps, or discovery rationales. The local saved evaluation retains an anonymous-ID to `episode:event_id` mapping so evidence can be traced back and verified. The same six evaluation dimensions are used. To inspect a single generated composite before any request:

```bash
python -m promoscout.evaluation.blind_viewer --source composite --candidate-id COMPOSITE_ARC_ID_001 --dry-run
```

After reviewing the exact candidate and configuring access, execute the command without `--dry-run`. Replace the example ID with an actual candidate ID from the generated JSON. Composite evaluations are merged under `composite:<candidate_id>` in `data/outputs/blind_viewer_evaluations.json`.

## Frontend

Run `streamlit run app.py` and open **Composite Promos**. Each card shows source episodes, arc/theme, total duration, segment-level provenance, source timestamps, event IDs, hook and potential ending, spoiler/coherence warnings, validation status, and any saved blind-viewer result. Approve, reject, or defer a candidate with an optional note; decisions are saved in `data/outputs/review_decisions.json` under the `composite` source. Missing artifacts produce an honest empty state. Composite candidates are also available on the Exports page.

## Synthetic fixture and limitations

Automated tests create a clearly synthetic three-episode fixture and construct a 30-second composite spanning episodes 1–3. It demonstrates schema validation, source grounding, event inclusion, timing, spoiler warnings, structural validation, and blind-viewer anonymization. These fixture event texts and narrative links are not Kuku screenplay findings and are not written to project output files.

Real series-level analysis and composite candidates do not currently exist. The current composite dry-run therefore reports zero opportunities. Real cross-episode story arcs, character aliases, continuity concerns, and spoiler warnings require successful Gemini analysis and human review. Composite coherence is a model-supported proposal, not a proven property; inspect every segment in context. The model evaluator is not a human audience study, and its verdict does not approve a candidate.
