# PromoScout

PromoScout is a local Python and Streamlit workspace for finding, validating, reviewing, and provisionally ranking promotional clips from serialized screenplay episodes.

## Features

- Parses timestamped screenplay events while preserving source IDs and event types.
- Generates continuous heuristic candidate windows deterministically.
- Provides semantic episode analysis and event-discovery infrastructure using Gemini structured output, with caching and resumability.
- Reconstructs clip previews from original events and enforces source, hook timing, and duration checks.
- Evaluates saved clips as a first-time viewer using six evidence-grounded dimensions.
- Builds transparent AI-assisted provisional rankings only from structurally valid candidates with locally validated blind-viewer evidence.
- Includes series-level and multi-episode composite generation infrastructure, with a synthetic-only fixture in tests.
- Provides a responsive Streamlit dashboard for exploration, candidate review, composites, rankings, and exports.

## Architecture and stack

Python 3.11+, Streamlit, Pydantic, Pandas, Plotly, Pytest, and the official `google-genai` SDK. Parsed data and intermediate results use local JSON files. Python controls event retrieval, timestamps, clip duration, source validation, cache keys, and output writes. Gemini is optional and is never called by the dashboard or tests.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r req.txt
```

For optional Gemini analysis, configure `GEMINI_API_KEY` and `GEMINI_MODEL` in a local `.env`. Do not put credentials in submission artifacts or source control. No live model request is required to launch or review the saved outputs.

## Launch

```bash
streamlit run app.py
```

The dashboard contains Overview, Episode Explorer, Candidate Review, Rankings, Exports, Series Workspace, and Composite Promos. Missing semantic or composite artifacts are shown as pending/empty states.

## Ranking outputs

The final provisional rankings are in `raw_outputs/final_rankings.json`, `raw_outputs/final_rankings.csv`, and `raw_outputs/final_ranking_report.md`. They contain the available validated heuristic assessments. AI scores are provisional, and human approval remains separate. The human gold set has no completed entries.

Regenerate rankings from the local artifacts with:

```bash
python -m promoscout.ranking.final_ranker
```

To override scoring weights, pass a JSON file with all six dimension keys:

```bash
python -m promoscout.ranking.final_ranker --weights-json scoring_weights.json
```

## Demo links

- Live prototype: **[Add hosted prototype URL if one is created]**
- Walkthrough recording: **[Add walkthrough URL after recording]**

No hosted URL is claimed in this package.
