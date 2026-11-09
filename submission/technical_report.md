# PromoScout — Technical Report

## Problem

The assignment asks for short promotional clips from serialized screenplay material. A useful system must find attention-grabbing moments, preserve their original source timing, produce a clip that is understandable to a new viewer, and leave editorial decisions reviewable. A regex score alone cannot establish whether an event matters in its story context.

## Approach and architecture

PromoScout parses scripts into timestamped dialogue, action, camera, and sound events with original event IDs. The current source dataset contains 50 parsed episodes and 3,727 events. Continuous candidate construction is deterministic Python: source events are retrieved by ID and overlap, opening hooks are constrained to a three-second lead-in, and clips are constrained to 20–60 seconds. Original timestamps and event references are retained in previews.

Heuristic discovery and semantic discovery are separate. The heuristic generator uses manually defined scoring signals and has produced 706 continuous candidates across 48 episodes. Semantic analysis and event tagging are implemented as a Gemini structured-output pipeline that receives original screenplay events, returns source event IDs, and uses chunk caching and incremental output. However, semantic result files are not populated, so semantic discovery has not been demonstrated on the real Kuku dataset.

Series-level analysis and composite generation are implemented as infrastructure. Series analysis is hierarchical and source-grounded. The composite builder turns cross-episode event anchors into Python-constructed segments, preserves per-episode source timing, and validates total duration and event integrity. No real series analysis or composite output exists. The three-episode composite in automated tests is explicitly synthetic and is not a screenplay finding.

## Blind-viewer evaluation

The blind-viewer model receives only events inside a candidate window, with relative clip timing, event type, content, and speaker/delivery where present. It does not receive episode summaries or story context. Its six dimensions are standalone clarity, hook effectiveness, conflict comprehension, stakes clarity, cliffhanger effectiveness, and desire to continue. Python checks candidate identity, clip event IDs, content hash, cited event IDs, and evidence quotations before using an assessment.

The saved multi-episode heuristic pilot selected 10 candidates. Eight assessments passed validation; two failed because the responses cited event IDs outside the supplied clips. Among the eight valid assessments, verdicts were seven strong and one mixed. Mean dimension scores were: clarity 3.88/5, hook 3.75/5, conflict 4.00/5, stakes 3.38/5, cliffhanger 3.62/5, and desire to continue 3.75/5. This is a small AI evaluation sample, not a human audience test.

## Final provisional ranking

The final ranker uses weights of 20% standalone clarity, 20% hook, 10% conflict, 10% stakes, 20% cliffhanger, and 20% desire to continue. The weighted 1–5 result is normalized to 0–100. Only candidates that pass structural checks and have valid saved blind-viewer evaluations receive a score. Missing assessments remain pending; failed evidence assessments remain failed and are excluded. Human review is independent.

Nine saved heuristic evaluations validated and were ranked: the eight valid pilot results plus an earlier saved Episode 1 evaluation. The top provisional scores were 82/100 for `EP001_PROMO_001`, `EP001_PROMO_011`, and `EP027_PROMO_022`. All nine have pending human decisions; none is an approved final selection. Of 706 heuristic candidates, 695 remain pending, two assessments failed, and none failed the ranker's structural checks. The separate existing `pilot_validation.json` is an older Episode 1 semantic validation with no semantic candidates and missing analysis/discovery artifacts; it is not the source of the heuristic structural passes.

## Actual limitations and next steps

Gemini returned temporary 503 high-demand errors during the pilot period. Two evaluation responses also failed evidence-reference validation. Real semantic discovery is pending; full series analysis and real composite generation have not run; the available blind-viewer sample is limited and AI-generated; the human gold set is empty; and no marketing conversion experiment was conducted. Reviewers should inspect source previews and make final decisions. The package distinguishes implemented infrastructure from functionality demonstrated on actual screenplay outputs.
