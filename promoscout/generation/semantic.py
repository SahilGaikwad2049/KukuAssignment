"""CLI for generating candidate windows from semantic event discoveries."""
import argparse
import json
from pathlib import Path

from promoscout.analysis.episode_analyzer import load_episodes, select_episodes
from promoscout.generation.continuous import generate_candidates_from_discoveries


def main():
    parser = argparse.ArgumentParser(description="Generate candidates from semantic event discoveries")
    parser.add_argument("episodes")
    parser.add_argument("discoveries", nargs="?", default="data/processed/event_discoveries.json")
    parser.add_argument("--output", default="data/outputs/semantic_candidates.json")
    parser.add_argument("--min-duration", type=float, default=20)
    parser.add_argument("--max-duration", type=float, default=60)
    parser.add_argument("--max-per-episode", type=int, default=25)
    parser.add_argument("--episode-number", type=int, help="Generate candidates only for this episode")
    args = parser.parse_args()
    try:
        episodes = select_episodes(load_episodes(args.episodes), episode_number=args.episode_number)
    except ValueError as exc:
        parser.error(str(exc))
    discoveries = json.loads(Path(args.discoveries).read_text(encoding="utf-8"))
    candidates = generate_candidates_from_discoveries(
        episodes, discoveries, min_duration=args.min_duration,
        max_duration=args.max_duration, max_candidates=args.max_per_episode,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"method": "semantic_event_ids", "candidates": [c.model_dump(mode="json") for c in candidates]}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Generated {len(candidates)} semantic candidate(s); saved to {output}")


if __name__ == "__main__":
    main()

