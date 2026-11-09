# PromoScout — Milestone 1 parser starter

Copy the included `promoscout/` and `tests/` folders into the root of your existing PromoScout project. Merge them with your existing folders; do not replace your project root.

## Run the parser on one episode

```bash
python -m promoscout.parser.screenplay_parser "data/input/episode-001.txt" --output data/processed/parsed_episodes.json
```

The input may also be a ZIP archive containing `.txt` screenplay files:

```bash
python -m promoscout.parser.screenplay_parser "data/input/episodes.zip" --output data/processed/parsed_episodes.json
```

## Run parser tests

```bash
python -m pytest tests/test_parser.py -q
```

The parser preserves screenplay order and overlapping events. Timestamps are retained in their original string form and also converted to seconds for calculations. Malformed or missing event timestamps are reported as warnings; invalid episode structure raises an error.
