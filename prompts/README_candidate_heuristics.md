# Candidate heuristic notes

The continuous candidate generator is deliberately deterministic. Its rules are
only a high-recall discovery pass, not a final promo-quality judgment.

- Hook signals use dialogue punctuation, high-tension phrases, selected high-
  impact words, emotional delivery labels, and selected physical/revelatory actions.
- Ending signals use questions, trailing dialogue, suspense phrases, unresolved-
  action wording, explicit cut-to-black camera directions, and proximity to
  scene/episode boundaries.
- Candidate boundaries are copied from source event timestamps.
- The hook must occur within the first three seconds of the candidate.
- All events overlapping the candidate window are retained by ID, even when
  dialogue, camera, action, or sound events overlap one another.
- Scene boundaries are not hard clip boundaries. A screenplay event may cross
  a scene boundary; preserve its timestamp and surface the parser warning.

These heuristic scores must never be represented as final AI promo scores.
The later LLM evaluator and editor review are responsible for judging narrative
quality, standalone clarity, and whether the endpoint is a real cliffhanger.
