"""Deterministic candidate sampling and grounded blind-viewer pilot reporting."""
from __future__ import annotations

import json
import random
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


DIMENSIONS = (
    "standalone_clarity",
    "hook_effectiveness",
    "conflict_comprehension",
    "stakes_clarity",
    "cliffhanger_effectiveness",
    "desire_to_continue",
)


def _window_key(record: dict) -> tuple:
    candidate = record["candidate"]
    return (
        candidate.episode_number,
        round(candidate.start_seconds, 2),
        round(candidate.end_seconds, 2),
    )


def _duration_band(candidate) -> str:
    duration = candidate.duration_seconds
    if duration <= 30:
        return "short"
    if duration <= 45:
        return "medium"
    return "long"


def deduplicate_candidate_records(records: list[dict]) -> list[dict]:
    """Remove repeated IDs and exact episode/time windows deterministically."""
    ordered = sorted(records, key=lambda item: (
        item["candidate"].episode_number,
        item["candidate"].start_seconds,
        item["candidate"].end_seconds,
        item["source"],
        item["candidate"].candidate_id,
    ))
    seen_windows, seen_ids, unique = set(), set(), []
    for record in ordered:
        candidate = record["candidate"]
        id_key = (record["source"], candidate.candidate_id)
        window = _window_key(record)
        if id_key in seen_ids or window in seen_windows:
            continue
        seen_ids.add(id_key)
        seen_windows.add(window)
        unique.append(record)
    return unique


def _stratified_episode_sample(episodes: list[int], count: int, rng: random.Random) -> list[tuple[int, int, int]]:
    """Choose one available episode from each equal-width slice of the range."""
    selected = []
    total = len(episodes)
    for bucket in range(count):
        start = bucket * total // count
        stop = (bucket + 1) * total // count
        chosen_index = rng.randrange(start, stop)
        selected.append((episodes[chosen_index], bucket + 1, count))
    return selected


def sample_candidate_records(
    records: list[dict],
    *,
    sample_episodes: int,
    candidates_per_episode: int = 1,
    seed: int = 42,
) -> tuple[list[dict], dict]:
    """Sample candidates across the episode range without using evaluation scores."""
    if sample_episodes < 1:
        raise ValueError("sample_episodes must be at least 1")
    if candidates_per_episode < 1:
        raise ValueError("candidates_per_episode must be at least 1")
    unique = deduplicate_candidate_records(records)
    by_episode: dict[int, list[dict]] = defaultdict(list)
    for record in unique:
        by_episode[record["candidate"].episode_number].append(record)
    available_episodes = sorted(by_episode)
    actual_episodes = min(sample_episodes, len(available_episodes))
    rng = random.Random(seed)
    episode_bins = _stratified_episode_sample(available_episodes, actual_episodes, rng) if actual_episodes else []

    selected, manifest_entries = [], []
    for episode_number, bucket_number, bucket_count in episode_bins:
        pool = by_episode[episode_number]
        # Random tie values are seeded. Candidate scores are deliberately not
        # consulted; hook type and duration are used only as variation metadata.
        tie_break = {id(item): rng.random() for item in pool}
        chosen, seen_types, seen_duration_bands, seen_families = [], set(), set(), set()
        for _ in range(min(candidates_per_episode, len(pool))):
            ranked = sorted(pool, key=lambda item: (
                item["candidate"].candidate_family_id in seen_families,
                item["candidate"].hook_type in seen_types,
                _duration_band(item["candidate"]) in seen_duration_bands,
                tie_break[id(item)],
                item["candidate"].candidate_id,
            ))
            item = ranked[0]
            pool.remove(item)
            candidate = item["candidate"]
            variation = []
            if seen_families and candidate.candidate_family_id not in seen_families:
                variation.append(f"adds candidate family {candidate.candidate_family_id}")
            if seen_types and candidate.hook_type not in seen_types:
                variation.append(f"adds hook type {candidate.hook_type}")
            if seen_duration_bands and _duration_band(candidate) not in seen_duration_bands:
                variation.append(f"adds {_duration_band(candidate)} duration band")
            within_episode_reason = (
                ", ".join(variation)
                if variation
                else "seeded tie-break among remaining candidates using only hook type and duration metadata"
            )
            rationale = (
                f"Selected from episode stratum {bucket_number}/{bucket_count} "
                f"across {len(available_episodes)} available episodes using seed {seed}; "
                f"within-episode selection: {within_episode_reason}. "
                "No blind-viewer scores or heuristic quality scores were used."
            )
            row = {**item, "selection_rationale": rationale}
            selected.append(row)
            chosen.append(row)
            seen_types.add(candidate.hook_type)
            seen_duration_bands.add(_duration_band(candidate))
            seen_families.add(candidate.candidate_family_id)
            manifest_entries.append({
                "candidate_id": candidate.candidate_id,
                "episode_number": candidate.episode_number,
                "candidate_source": item["source"],
                "start_seconds": candidate.start_seconds,
                "end_seconds": candidate.end_seconds,
                "duration_seconds": candidate.duration_seconds,
                "hook_type": candidate.hook_type,
                "selection_rationale": rationale,
            })

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "candidate_sources": sorted({record["source"] for record in records}),
        "sample_seed": seed,
        "requested_episode_count": sample_episodes,
        "selected_episode_count": len(episode_bins),
        "candidates_per_episode_requested": candidates_per_episode,
        "available_episode_numbers": available_episodes,
        "selected_episode_numbers": [episode for episode, _, _ in episode_bins],
        "selected_candidate_count": len(selected),
        "selection_method": "seeded stratified sample across sorted available episodes; metadata-only within-episode variation",
        "selection_uses_evaluation_scores": False,
        "selected_candidates": manifest_entries,
    }
    return selected, manifest


def build_pilot_report(
    records: list[dict],
    evaluations_payload: dict,
    failures: list[dict],
    *,
    model_id: str,
    cached_count: int = 0,
    human_decisions: dict | None = None,
) -> dict:
    evaluation_map = evaluations_payload.get("evaluations", {}) if isinstance(evaluations_payload, dict) else {}
    human_decisions = human_decisions or {}
    failure_keys = {(row.get("source"), row.get("candidate_id")) for row in failures}
    candidate_rows = []
    successful_assessments = []
    successful_records = []
    for record in records:
        candidate = record["candidate"]
        source = record["source"]
        key = f"{source}:{candidate.candidate_id}"
        stored = evaluation_map.get(key)
        if stored and (source, candidate.candidate_id) not in failure_keys:
            assessment = stored.get("assessment", {})
            successful_assessments.append(assessment)
            successful_records.append(record)
            human = human_decisions.get(key, {}) if isinstance(human_decisions, dict) else {}
            candidate_rows.append({
                "candidate_id": candidate.candidate_id,
                "episode_number": candidate.episode_number,
                "candidate_source": source,
                "overall_viewer_verdict": assessment.get("overall_viewer_verdict"),
                "scores": {name: assessment.get(name, {}).get("score") for name in DIMENSIONS},
                "confidence": assessment.get("confidence"),
                "main_reason": assessment.get("verdict_explanation", ""),
                "human_decision": human.get("decision", "pending") if isinstance(human, dict) else "pending",
            })
        else:
            candidate_rows.append({
                "candidate_id": candidate.candidate_id,
                "episode_number": candidate.episode_number,
                "candidate_source": source,
                "overall_viewer_verdict": None,
                "scores": {name: None for name in DIMENSIONS},
                "confidence": None,
                "main_reason": None,
                "human_decision": human_decisions.get(key, {}).get("decision", "pending") if isinstance(human_decisions.get(key, {}), dict) else "pending",
                "evaluation_status": "failed" if (source, candidate.candidate_id) in failure_keys else "not_saved",
            })

    score_distributions = {}
    for dimension in DIMENSIONS:
        scores = [
            int(assessment[dimension]["score"])
            for assessment in successful_assessments
            if isinstance(assessment.get(dimension), dict) and assessment[dimension].get("score") is not None
        ]
        score_distributions[dimension] = {
            "average": round(statistics.mean(scores), 2) if scores else None,
            "median": statistics.median(scores) if scores else None,
            "minimum": min(scores) if scores else None,
            "maximum": max(scores) if scores else None,
            "distribution": {str(value): scores.count(value) for value in range(1, 6)},
        }
    verdict_counts = Counter(
        assessment.get("overall_viewer_verdict") for assessment in successful_assessments
        if assessment.get("overall_viewer_verdict") in {"strong", "mixed", "weak"}
    )

    qualitative = []
    qualitative_specs = (
        ("Confusing openings / weak hooks", lambda a: (a.get("hook_effectiveness") or {}).get("score", 5) <= 2,
         lambda a: (a.get("hook_effectiveness") or {}).get("rationale", "")),
        ("Weak standalone clarity", lambda a: (a.get("standalone_clarity") or {}).get("score", 5) <= 2,
         lambda a: (a.get("standalone_clarity") or {}).get("rationale", "")),
        ("Unclear stakes", lambda a: (a.get("stakes_clarity") or {}).get("score", 5) <= 2,
         lambda a: (a.get("stakes_clarity") or {}).get("rationale", "")),
        ("Weak or ineffective unresolved endings", lambda a: (a.get("cliffhanger_effectiveness") or {}).get("score", 5) <= 2,
         lambda a: (a.get("cliffhanger_effectiveness") or {}).get("rationale", "")),
        ("Abrupt cuts explicitly described as ineffective", lambda a: (
            (a.get("cliffhanger_effectiveness") or {}).get("score", 5) <= 3
            and "abrupt" in (a.get("cliffhanger_effectiveness") or {}).get("rationale", "").lower()
        ), lambda a: (a.get("cliffhanger_effectiveness") or {}).get("rationale", "")),
        ("Low desire to continue", lambda a: (a.get("desire_to_continue") or {}).get("score", 5) <= 2,
         lambda a: (a.get("desire_to_continue") or {}).get("rationale", "")),
        ("Missing context reported by the viewer", lambda a: bool(a.get("missing_context_the_viewer_wishes_for")),
         lambda a: "; ".join(a.get("missing_context_the_viewer_wishes_for", []))),
        ("Confusing elements reported by the viewer", lambda a: bool(a.get("confusing_elements")),
         lambda a: "; ".join(a.get("confusing_elements", []))),
    )
    for label, predicate, extract in qualitative_specs:
        findings = []
        for record, assessment in zip(successful_records, successful_assessments):
            if predicate(assessment):
                candidate = record["candidate"]
                note = extract(assessment)
                findings.append({
                    "candidate_id": candidate.candidate_id,
                    "episode_number": candidate.episode_number,
                    "finding_from_saved_assessment": note,
                })
        qualitative.append({"finding": label, "count": len(findings), "examples": findings[:5]})

    sources = sorted({record["source"] for record in records})
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": model_id,
        "candidate_source": sources[0] if len(sources) == 1 else ("mixed" if sources else None),
        "overall_results": {
            "selected_candidates": len(records),
            "successfully_evaluated": len(successful_assessments),
            "failed": len(failures),
            "reused_from_cache": min(cached_count, len(successful_assessments)),
            "episodes_represented": len({record["candidate"].episode_number for record in records}),
            "episode_numbers": sorted({record["candidate"].episode_number for record in records}),
        },
        "episode_level_results": candidate_rows,
        "score_distributions": score_distributions,
        "verdict_distribution": {value: verdict_counts.get(value, 0) for value in ("strong", "mixed", "weak")},
        "qualitative_findings": qualitative,
        "failures": failures,
        "human_judgments_are_separate": True,
        "interpretation_note": "Descriptive pilot results only; this small sample does not establish statistical significance.",
    }


def pilot_report_markdown(report: dict) -> str:
    summary = report["overall_results"]
    lines = [
        "# Blind-Viewer Pilot Report",
        "",
        f"- Candidate source: **{report.get('candidate_source') or 'none'}**",
        f"- Model: `{report.get('model_id') or '(not recorded)'}`",
        f"- Selected: **{summary['selected_candidates']}** · Successfully evaluated: **{summary['successfully_evaluated']}** · Failed: **{summary['failed']}** · Reused from cache: **{summary['reused_from_cache']}**",
        f"- Episodes represented: **{summary['episodes_represented']}** ({', '.join(map(str, summary['episode_numbers'])) or 'none'})",
        "",
        "> Descriptive pilot results only. A 10–20 candidate sample does not establish statistical significance.",
        "",
        "## Candidate results",
        "",
        "| Episode | Candidate | Source | Verdict | Scores (clarity / hook / conflict / stakes / ending / continue) | Confidence | Human decision | Reason |",
        "|---:|---|---|---|---|---:|---|---|",
    ]
    for row in report["episode_level_results"]:
        scores = " / ".join(str(row["scores"].get(name) or "—") for name in DIMENSIONS)
        reason = str(row.get("main_reason") or row.get("evaluation_status") or "—").replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {row['episode_number']} | {row['candidate_id']} | {row['candidate_source']} | {row.get('overall_viewer_verdict') or 'failed'} | {scores} | {row.get('confidence') if row.get('confidence') is not None else '—'} | {row.get('human_decision', 'pending')} | {reason} |"
        )
    lines.extend(["", "## Score distributions", "", "| Dimension | Average | Median | Min | Max | Scores 1–5 |", "|---|---:|---:|---:|---:|---|"])
    for name in DIMENSIONS:
        item = report["score_distributions"][name]
        distribution = ", ".join(f"{score}: {count}" for score, count in item["distribution"].items())
        lines.append(f"| {name.replace('_', ' ').title()} | {item['average'] if item['average'] is not None else '—'} | {item['median'] if item['median'] is not None else '—'} | {item['minimum'] if item['minimum'] is not None else '—'} | {item['maximum'] if item['maximum'] is not None else '—'} | {distribution} |")
    lines.extend(["", "## Verdict distribution", ""])
    for verdict, count in report["verdict_distribution"].items():
        lines.append(f"- {verdict.title()}: {count}")
    lines.extend(["", "## Qualitative findings from saved assessments", ""])
    for item in report["qualitative_findings"]:
        lines.append(f"### {item['finding']} ({item['count']})")
        if not item["examples"]:
            lines.append("- No matching saved assessments.")
        for example in item["examples"]:
            note = example["finding_from_saved_assessment"] or "No supporting explanation was provided."
            lines.append(f"- EP{example['episode_number']:03d} · {example['candidate_id']}: {note}")
    lines.extend(["", "## Failures", ""])
    if report["failures"]:
        for failure in report["failures"]:
            lines.append(f"- {failure.get('source')}:{failure.get('candidate_id')}: {failure.get('error')}")
    else:
        lines.append("- None")
    lines.extend(["", "Human decisions are reported separately and are not inferred from AI verdicts.", ""])
    return "\n".join(lines)


def save_pilot_report(report: dict, json_path: str | Path, markdown_path: str | Path) -> None:
    json_path, markdown_path = Path(json_path), Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_save(json_path, json.dumps(report, ensure_ascii=False, indent=2))
    _atomic_save(markdown_path, pilot_report_markdown(report))


def _atomic_save(path: Path, content: str) -> None:
    import os
    import tempfile

    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
        raise
