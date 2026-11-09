from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from zipfile import ZipFile

from promoscout.schemas import ParsedEpisode, Scene, ScreenplayEvent

EPISODE_RE = re.compile(r"^\s*EPISODE NUMBER:\s*(.*?)\s*$", re.IGNORECASE)
LOGLINE_RE = re.compile(r"^\s*EPISODE LOGLINE:\s*(.*?)\s*$", re.IGNORECASE)
CONTINUITY_RE = re.compile(r"^\s*CONTINUITY NOTES:\s*(.*?)\s*$", re.IGNORECASE)
SCENE_RE = re.compile(r"^\s*SCENE\s+(\d+)\s*$", re.IGNORECASE)
SCENE_HEADING_RE = re.compile(r"^\s*Scene heading:\s*(.*?)\s*$", re.IGNORECASE)
SCENE_START_RE = re.compile(r"^\s*Scene start:\s*([\d:.]+)\s*$", re.IGNORECASE)
SCENE_END_RE = re.compile(r"^\s*Scene end:\s*([\d:.]+)\s*$", re.IGNORECASE)
EVENT_HEADER_RE = re.compile(r"^\s*\*\*(.*?)\*\*\s*$")
EVENT_TIME_RE = re.compile(
    r"Start:\s*\((?P<start>[^)]+)\)\s*\|\s*End:\s*\((?P<end>[^)]+)\)",
    re.IGNORECASE,
)


def timestamp_to_seconds(value: str) -> float:
    """Convert MM:SS.ss to seconds while preserving the original elsewhere."""
    value = value.strip()
    match = re.fullmatch(r"(?P<minutes>\d+):(?P<seconds>\d{2}(?:\.\d+)?)", value)
    if not match:
        raise ValueError(f"Unsupported timestamp format: {value!r}")
    minutes = int(match.group("minutes"))
    seconds = float(match.group("seconds"))
    if seconds >= 60:
        raise ValueError(f"Seconds component must be below 60: {value!r}")
    return minutes * 60 + seconds


def _classify_label(label: str) -> tuple[str, str | None, str | None]:
    clean_label = label.strip().rstrip(":").strip()
    upper = clean_label.upper()
    if upper.startswith("CAMERA"):
        return "camera", None, None
    if upper.startswith("ACTION"):
        return "action", None, None
    if upper.startswith("SOUND") or upper.startswith("MUSIC"):
        return "sound_fx", None, None

    # Dialogue heading, e.g. LADY JULIANA (Heartbroken, Whispering)
    speaker_match = re.match(r"^(?P<speaker>[^()]+?)\s*(?P<attrs>(?:\([^)]*\)\s*)*)$", clean_label)
    if speaker_match:
        speaker = speaker_match.group("speaker").strip()
        attrs = re.findall(r"\(([^)]*)\)", speaker_match.group("attrs"))
        delivery = "; ".join(item.strip() for item in attrs if item.strip()) or None
        return "dialogue", speaker or None, delivery

    return "other", None, None


def _clean_content(lines: list[str]) -> str:
    content = "\n".join(line.strip() for line in lines if line.strip()).strip()
    quote_pairs = [("\"", "\""), ("“", "”"), ("‘", "’")]
    for opening, closing in quote_pairs:
        if len(content) >= 2 and content.startswith(opening) and content.endswith(closing):
            content = content[1:-1].strip()
            break
    return content


def parse_episode_text(text: str, source_file: str | None = None) -> ParsedEpisode:
    lines = text.lstrip("\ufeff").splitlines()
    episode_number: int | None = None
    logline: str | None = None
    continuity_notes: str | None = None
    in_script = False
    scenes: list[dict] = []
    warnings: list[str] = []
    event_counter = 0
    i = 0

    # Read episode metadata and then parse the script body.
    while i < len(lines):
        line = lines[i]
        if "--- SCRIPT START ---" in line.upper():
            in_script = True
            i += 1
            continue
        if "--- SCRIPT END ---" in line.upper():
            break

        if not in_script:
            match = EPISODE_RE.match(line)
            if match:
                number_match = re.search(r"\d+", match.group(1))
                if number_match:
                    episode_number = int(number_match.group())
            match = LOGLINE_RE.match(line)
            if match:
                logline = match.group(1).strip() or None
            match = CONTINUITY_RE.match(line)
            if match:
                continuity_notes = match.group(1).strip() or None
            i += 1
            continue

        scene_match = SCENE_RE.match(line)
        if scene_match:
            scenes.append({
                "scene_number": int(scene_match.group(1)),
                "heading": None,
                "start_timestamp": None,
                "end_timestamp": None,
                "start_seconds": None,
                "end_seconds": None,
                "events": [],
            })
            i += 1
            continue

        if not scenes:
            i += 1
            continue

        current_scene = scenes[-1]
        match = SCENE_HEADING_RE.match(line)
        if match:
            current_scene["heading"] = match.group(1).strip() or None
            i += 1
            continue
        match = SCENE_START_RE.match(line)
        if match:
            current_scene["start_timestamp"] = match.group(1)
            try:
                current_scene["start_seconds"] = timestamp_to_seconds(match.group(1))
            except ValueError as exc:
                warnings.append(f"Line {i + 1}: {exc}")
            i += 1
            continue
        match = SCENE_END_RE.match(line)
        if match:
            current_scene["end_timestamp"] = match.group(1)
            try:
                current_scene["end_seconds"] = timestamp_to_seconds(match.group(1))
            except ValueError as exc:
                warnings.append(f"Line {i + 1}: {exc}")
            i += 1
            continue

        header_match = EVENT_HEADER_RE.match(line)
        if not header_match:
            i += 1
            continue

        label = header_match.group(1).strip().rstrip(":").strip()
        block_start = i
        j = i + 1
        while j < len(lines):
            if SCENE_RE.match(lines[j]) or EVENT_HEADER_RE.match(lines[j]) or "--- SCRIPT END ---" in lines[j].upper():
                break
            j += 1

        block_lines = lines[i + 1:j]
        time_line_index = None
        time_match = None
        for k, block_line in enumerate(block_lines):
            candidate_match = EVENT_TIME_RE.search(block_line)
            if candidate_match:
                time_line_index = k
                time_match = candidate_match
                break

        if time_match is None or time_line_index is None:
            warnings.append(f"Line {block_start + 1}: event heading {label!r} has no valid Start/End timestamp")
            i = j
            continue

        start_ts = time_match.group("start").strip()
        end_ts = time_match.group("end").strip()
        try:
            start_seconds = timestamp_to_seconds(start_ts)
            end_seconds = timestamp_to_seconds(end_ts)
        except ValueError as exc:
            warnings.append(f"Line {block_start + 1}: {exc}")
            i = j
            continue

        content_lines = block_lines[time_line_index + 1:]
        content = _clean_content(content_lines)
        if not content:
            warnings.append(f"Line {block_start + 1}: event heading {label!r} has no content")
            i = j
            continue

        event_counter += 1
        event_type, speaker, delivery = _classify_label(label)
        content_line_numbers = [
            block_start + 1 + offset + 1
            for offset in range(time_line_index + 1, len(block_lines))
            if block_lines[offset].strip()
        ]
        source_line_end = max(content_line_numbers) if content_line_numbers else block_start + 1
        raw_block = "\n".join(lines[block_start:j]).strip()

        event = ScreenplayEvent(
            event_id=f"EP{episode_number or 0:03d}_SC{current_scene['scene_number']:03d}_EV{event_counter:04d}",
            event_type=event_type,
            label=label,
            speaker=speaker,
            delivery=delivery,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            content=content,
            raw_block=raw_block,
            source_line_start=block_start + 1,
            source_line_end=source_line_end,
        )
        current_scene["events"].append(event)
        i = j

    if episode_number is None:
        raise ValueError("Could not find EPISODE NUMBER in screenplay header")
    if not in_script:
        raise ValueError("Could not find --- SCRIPT START --- marker")
    if not scenes:
        raise ValueError("No scenes found in screenplay")

    parsed_scenes = []
    for scene_data in scenes:
        if scene_data["start_seconds"] is not None and scene_data["end_seconds"] is not None:
            if scene_data["end_seconds"] < scene_data["start_seconds"]:
                warnings.append(f"Scene {scene_data['scene_number']}: end timestamp precedes start timestamp")
        for event in scene_data["events"]:
            if scene_data["start_seconds"] is not None and event.start_seconds < scene_data["start_seconds"]:
                warnings.append(f"{event.event_id}: event begins before scene start")
            if scene_data["end_seconds"] is not None and event.end_seconds > scene_data["end_seconds"]:
                warnings.append(f"{event.event_id}: event ends after scene end")
        parsed_scenes.append(Scene(**scene_data))

    return ParsedEpisode(
        episode_number=episode_number,
        logline=logline,
        continuity_notes=continuity_notes,
        source_file=source_file,
        scenes=parsed_scenes,
        warnings=warnings,
    )


def parse_episode_file(file_path: str | Path) -> ParsedEpisode:
    path = Path(file_path)
    text = path.read_text(encoding="utf-8-sig")
    return parse_episode_text(text, source_file=path.name)


def parse_zip_archive(zip_path: str | Path) -> list[ParsedEpisode]:
    parsed: list[ParsedEpisode] = []
    with ZipFile(zip_path) as archive:
        members = [
            name for name in archive.namelist()
            if name.lower().endswith(".txt")
            and "__MACOSX/" not in name
            and not Path(name).name.startswith("._")
        ]
        members.sort(key=lambda name: [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", name)
        ])
        if not members:
            raise ValueError("The ZIP archive contains no .txt screenplay files")
        for member in members:
            try:
                raw = archive.read(member)
                try:
                    text = raw.decode("utf-8-sig")
                except UnicodeDecodeError:
                    text = raw.decode("cp1252")
                parsed.append(parse_episode_text(text, source_file=member))
            except Exception as exc:
                raise ValueError(f"Failed to parse {member!r}: {exc}") from exc
    return parsed


def save_episodes_json(episodes: list[ParsedEpisode], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = [episode.model_dump(mode="json") for episode in episodes]
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a PromoScout screenplay file into JSON.")
    parser.add_argument("input", help="Path to a .txt screenplay or .zip archive")
    parser.add_argument("--output", default="data/processed/parsed_episodes.json")
    args = parser.parse_args()
    input_path = Path(args.input)
    if input_path.suffix.lower() == ".zip":
        episodes = parse_zip_archive(input_path)
    else:
        episodes = [parse_episode_file(input_path)]
    save_episodes_json(episodes, args.output)
    total_events = sum(len(episode.events) for episode in episodes)
    print(f"Parsed {len(episodes)} episode(s), {total_events} event(s).")
    print(f"Saved JSON to: {args.output}")
    for episode in episodes:
        print(f"Episode {episode.episode_number}: {len(episode.scenes)} scene(s), {len(episode.events)} event(s), {len(episode.warnings)} warning(s)")
        for warning in episode.warnings[:5]:
            print(f"  WARNING: {warning}")


if __name__ == "__main__":
    main()
