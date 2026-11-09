# PromoScout Frontend Guide

This guide explains how to navigate and present the PromoScout Streamlit frontend. It describes the controls that are implemented and clarifies which results are available in the current local dataset.

## Launch the application

From the project directory, activate the project environment if needed, then run:

```bash
streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`. Open that URL in a browser. Use the navigation in the left sidebar. On a narrow screen, Streamlit may collapse the sidebar into its menu button.

The navigation has two groups:

- **Workspace:** Overview, Episode Explorer, Candidate Review, Rankings, Exports.
- **More:** Series Workspace and Composite Promos.

## 1. Overview

The landing page gives a compact snapshot of the local pipeline:

- **Episodes:** number of parsed source episodes.
- **Source events:** parsed dialogue, actions, camera directions, and sound events.
- **Promo candidates:** semantic and heuristic candidate counts, labelled separately.
- **Human reviews:** saved approve/reject/later decisions.
- **Pipeline status:** compact status indicators for parsing, semantic analysis, candidates, and review.

If semantic results have not been generated, a notice says that semantic discovery is pending. Missing data is shown as an empty state instead of invented statistics. The status area reflects files currently available on disk; it does not start processing.

For the current project data, the parsed dataset has 50 episodes and 3,727 events, with 706 heuristic candidates. Real semantic candidate output and real composite candidate output are not currently available. The visible counts can change when local output files change.

## 2. Episode Explorer

Use this page to inspect the source screenplay:

1. Choose an episode from the **Episode** selector.
2. Review the event, scene, and estimated screenplay-duration metrics.
3. Use **Event types** to filter dialogue, action, camera, sound, or other event types.
4. Expand a row under each scene heading to read its content and inspect the original timestamp, event type, speaker/delivery metadata, and source event ID.
5. Scroll to **Narrative analysis** to view saved episode summary, characters, relationships, conflict, developments, revelations, emotional turns, unresolved questions, or narrative progression.

Narrative-analysis fields appear only when saved analysis exists. The page does not generate analysis by itself.

## 3. Candidate Review

This is the main page for examining and making decisions about individual clips.

### Find candidates

- **Candidate source:** choose Both, Semantic, or Heuristic. Source labels are retained so heuristic candidates are not mistaken for semantic discoveries.
- **Episodes:** restrict the list to selected episodes.
- **Hook types:** filter by the candidate's stored hook type.
- **Sort:** order chronologically, or by an existing heuristic score, hook score, or duration. Existing pipeline scores are discovery signals; they are not quality ratings.
- **Page:** candidate cards are paginated to keep the page manageable.

### Read a candidate card

Cards show the candidate ID, episode, source, candidate family, clip timestamps, duration, hook type, opening text, and potential ending text. The interface calls it a **potential ending** because it has not necessarily been judged to be a genuine cliffhanger.

Expand **Chronological screenplay preview** to reconstruct the events in the candidate window. The preview includes source times, relative clip times, event IDs, type, speaker/delivery when available, and event text. If source events are missing, the page reports that instead of silently filling them in.

Expand **Blind-viewer evaluation** to see a saved independent assessment, including the viewer's understanding, six dimension scores, cited evidence, confusing elements, and verdict. If there is no valid saved assessment, the page labels it pending or invalid; it does not invent a score.

### Record human review

For semantic and heuristic candidates, use the review form:

1. Optionally enter a note.
2. Select **Approve**, **Reject**, or **Later**.
3. Click **Save review decision**.

The decision is stored locally in `data/outputs/review_decisions.json`, separately for each candidate source and ID. It survives Streamlit reruns and does not modify candidate output files. Human decisions remain independent of AI scores.

The candidate page can also show the saved multi-episode blind-viewer pilot summary and its selection manifest when those files exist.

## 4. Rankings

The Rankings page presents three tabs:

- **Continuous:** single-episode candidates.
- **Composite:** cross-episode candidates.
- **Combined:** both categories together.

Only candidates with validated source structure and a valid saved blind-viewer evaluation receive an AI-assisted provisional score. The current six dimensions are standalone clarity, hook effectiveness, conflict comprehension, stakes clarity, cliffhanger effectiveness, and desire to continue. Scores are displayed with candidate provenance, represented episode(s), duration, verdict, structural status, human decision, and the reason for inclusion.

The score is a transparent provisional ranking, not objective truth or a guarantee of marketing performance. It does not approve a candidate. Human approval/rejection is shown separately. Pending candidates, failed assessments, and structurally ineligible candidates are listed in a collapsed section. An empty composite tab means no eligible real composite results are present; synthetic test fixtures are not shown as actual promos.

If the final ranking files have not been generated yet, the page explains how to generate them with:

```bash
python -m promoscout.ranking.final_ranker
```

## 5. Exports

The Exports page provides downloads for available local artifacts:

- Semantic candidates: JSON and CSV.
- Heuristic candidates: JSON and CSV.
- Composite candidates: JSON and CSV.
- Human-approved candidates: JSON and CSV.
- Human review decisions: JSON.
- Pilot validation: JSON and Markdown.
- Blind-viewer pilot report: JSON and Markdown, plus the selection manifest when available.
- Heuristic-versus-semantic comparison: JSON and Markdown when available.
- Final rankings: JSON, CSV, and Markdown report.

Unavailable files are shown as unavailable/disabled. Downloads reflect saved project data; the page does not trigger analysis or create fake results.

## 6. Series Workspace — ZIP upload and optional analysis

This page is for starting an isolated workspace from a screenplay ZIP.

### Upload and parse locally

1. Open **More → Series Workspace**.
2. Enter a series name.
3. Choose one `.zip` file in **Screenplay ZIP (.txt scripts)**.
4. Click **Upload and parse locally**.

The archive must contain screenplay `.txt` files formatted for PromoScout's existing parser, including episode numbering and screenplay event/timestamp structure. The ZIP is validated and parsed locally. Uploading/parsing does **not** call Gemini. After parsing, the UI reports the parsed episode and event counts.

Parsed workspace files are saved under `data/workspaces/` using a name/content-hash directory. The uploaded ZIP itself is not kept. The per-series parsed data remains local in the workspace and is excluded from Git.

### Optional Gemini analysis

After parsing, select a workspace. The page displays a model-ID field and whether an API key is configured; it never displays the key. To send screenplay text for semantic analysis, the reviewer must:

1. Confirm that the configured provider is Gemini and the model is set.
2. Check the explicit screenplay-sharing consent box.
3. Click **Analyze series and discover promo candidates**.

Without those explicit actions, no Gemini analysis starts. The current workspace integration supports Gemini, not arbitrary providers. Processing is sequential and uses the existing cache/resume behavior. Temporary provider failures stop the run with completed episode outputs retained for a later resume.

### Inspect workspace output

Choose an episode under **Explore episode output**. When present, the page displays the episode summary and conflict, expandable characters/relationships/developments/revelations/unresolved questions, discovered event IDs with their source event text and rationale, and semantic candidate windows. JSON downloads are offered for analysis, discoveries, and semantic candidates when those files exist.

This workspace is separate from the shared project candidate files. Processing an uploaded series does not silently replace the existing 50-episode dataset.

## 7. Composite Promos

Open **More → Composite Promos** to inspect saved cross-episode candidate output. A candidate card can show its story arc/theme, represented episodes, total duration, segments, original episode timestamps, promo-relative times, event IDs, source text, hook and potential ending, and spoiler/coherence warnings. It also shows structural validation and any saved blind-viewer result where available.

Use **Approve**, **Reject**, or **Review later** to save a human decision and optional note. The decision is stored in the same local review file without changing candidate output.

The page only displays saved composite candidates; it does not generate them. A synthetic test fixture is intentionally not presented as a real promo. At present there are no real composite candidates, so the expected state is an honest empty message until real series analysis and composite generation have produced output.

## Data, privacy, and interpretation

- The existing project pages read local JSON under `data/processed/` and `data/outputs/`.
- The frontend does not call an LLM just to browse, filter, review, rank, or export existing output.
- ZIP parsing is local. The separate Series Workspace analysis action sends screenplay data to Gemini only after the user checks the consent box and starts analysis.
- Missing artifacts remain missing; the UI displays empty states instead of fabricated data.
- A structural pass means the clip meets defined source/timing/duration checks. It does not prove that the clip is compelling.
- Blind-viewer scores are AI assessments, not human judgments. Human review is a separate decision.
- Synthetic fixtures are test material, not screenplay discoveries.

## Visual design and responsive navigation

The UI uses a near-black background, muted dark cards, white/light-grey text, and a restrained red accent. Typography uses a local-first system sans-serif stack, so no remote font or image is required. Streamlit's native sidebar navigation collapses to its menu on narrow screens; cards and controls use flexible widths and wrap long screenplay content.

## Frontend code map

- `app.py` — page registration and sidebar navigation.
- `promoscout/dashboard/pages.py` — Overview, Episode Explorer, Candidate Review, Rankings, and Exports.
- `promoscout/dashboard/series_workspace.py` — ZIP intake, local workspace persistence, and explicitly consented Gemini processing.
- `promoscout/dashboard/composite_page.py` — composite candidate preview and human review.
- `promoscout/dashboard/data_loader.py` — cached loading, validation, and source normalization for local artifacts.
- `promoscout/dashboard/review_manager.py` — atomic local reviewer-decision persistence.
- `promoscout/dashboard/styles.py` and `components.py` — shared styling and interface components.

## Dependencies

Install from the project dependency file and launch:

```bash
python -m pip install -r req.txt
streamlit run app.py
```
