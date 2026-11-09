# Limitations and Remaining Work

- Gemini free-tier availability is variable. High-demand 503 errors interrupted some requests; bounded retry and resume support is present, but a stable service window is still required for further runs.
- The blind-viewer pilot is small: 8 valid assessments from 10 selected heuristic candidates. Two responses cited source event IDs outside their candidate clips and were excluded.
- Blind-viewer scores come from an AI evaluator, not recruited first-time viewers. No real marketing click-through, conversion, or audience experiment was conducted.
- Real semantic discovery outputs are absent. The Episode 1 semantic pilot has not completed.
- Full-series narrative analysis has not been executed on the real 50-episode dataset.
- No real composite candidate has been generated from Kuku episode events. The composite example in automated tests is synthetic and must not be described as a real result.
- The human review-decision file is absent and the gold-set entries are empty. The current ranked heuristic candidates all remain pending human review; none is approved.
- The existing Episode 1 pilot validation report has missing semantic analysis/discovery artifacts and zero semantic candidate rows. The final ranker independently checked the continuous candidates, but semantic and composite structural validation cannot be demonstrated without their outputs.
- Semantic story links, series continuity, spoiler severity, and composite narrative coherence require human inspection even when model source IDs validate.
