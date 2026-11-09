# PromoScout — Milestone 5: Pilot validation and QA

The pilot validator checks source integrity and structural rules locally. It
does not call Gemini, make quality scores, or decide whether a potential ending
is a genuine cliffhanger.

## What it checks

- Episode analysis and discovery records exist and conform to their Pydantic
  schemas, including permitted discovery categories and required fields.
- Every discovery and candidate reference resolves to an original event in the
  correct episode. The report includes the resolved source content and original
  timestamps for valid discoveries.
- Candidate start, hook, and ending metadata agree with their original events.
- Candidate duration is 20–60 seconds; hook lead-in is at most 3 seconds.
- Included/preview events resolve to the screenplay, cover every event that
  overlaps the clip, preserve chronological order, and use consistent original
  and relative timestamps when preview rows are supplied.
- Candidate IDs, duplicate windows, and candidate-family assignments are
  checked. Scene boundaries are not treated as clip boundaries, so an event
  that extends slightly beyond its scene is not rejected for that reason.

Errors invalidate a source reference or structural condition. Warnings flag
items such as duplicate discovery references or duplicate windows for review.
The JSON report contains machine-readable issue codes; the Markdown report
summarizes the same findings.

## What still needs human judgment

A structurally valid clip may still be a poor promotion. Hook appeal,
standalone clarity, conflict comprehension, viewer interest, and whether the
ending feels unresolved remain pending for human review or a separate semantic
evaluation. The validator never reports these as passed and never converts a
potential ending into a confirmed cliffhanger.

## Run the single-episode pilot

After setting a supported `GEMINI_MODEL` and an approved `GEMINI_API_KEY`, first
inspect episode 1 without making an API request:

```bash
python -m promoscout.analysis.episode_analyzer \
  data/processed/parsed_all_episodes.json \
  --episode-number 1 \
  --dry-run
```

Run semantic analysis for that episode only:

```bash
python -m promoscout.analysis.episode_analyzer \
  data/processed/parsed_all_episodes.json \
  --episode-number 1
```

Generate its semantic candidates locally:

```bash
python -m promoscout.generation.semantic \
  data/processed/parsed_all_episodes.json \
  data/processed/event_discoveries.json \
  --episode-number 1 \
  --output data/outputs/semantic_candidates.json
```

Validate the artifacts against the parsed episode:

```bash
python -m promoscout.evaluation.pilot_validator --episode-number 1
```

By default, the validator reads:

- `data/processed/parsed_all_episodes.json`
- `data/processed/episode_analysis.json`
- `data/processed/event_discoveries.json`
- `data/outputs/semantic_candidates.json`

It writes `data/outputs/pilot_validation.json` and
`data/outputs/pilot_validation.md`. Use `--parsed`, `--analysis`,
`--discoveries`, `--candidates`, `--json-output`, and `--markdown-output` to
override those paths. A missing or empty semantic artifact is reported as an
error in the generated report; existing screenplay and candidate artifacts are
read only.

## Human gold set

`evaluation_data/gold_set.json` contains an empty `entries` array and a clearly
marked `entry_template`; there are no invented human judgments. Copy the
template into `entries`, set `template_only` to `false`, then manually provide:

- episode number, original start/end timestamps, and source event IDs;
- `hook_quality`: `high`, `mixed`, or `low`;
- `standalone_clarity`: `clear`, `partially_clear`, or `unclear`;
- `cliffhanger_quality`: `compelling`, `ambiguous`, or `not_a_cliffhanger`;
- `overall_human_judgment`: `keep`, `revise`, or `reject`;
- comments explaining the decision.

The gold set is for human-selected examples only. Do not fill it from heuristic
or model confidence values.

## Current pilot status and verification

The repository currently has 50 parsed episodes, but
`data/processed/episode_analysis.json` is empty and the semantic discovery and
candidate outputs have not been generated. The validator can still produce an
accurate missing-artifact report, but a successful real end-to-end semantic
pilot requires running analysis and candidate generation first. Mocked-fixture
tests verify validator behavior without making external API calls.

Run local checks with:

```bash
python -m pytest -q
python -m compileall -q promoscout tests
python -m promoscout.evaluation.pilot_validator --help
```
