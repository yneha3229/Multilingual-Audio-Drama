"""Turn a multi-episode series into audio dramas, with a consistent cast.

Audio series run for many episodes. A listener notices at once if a character
sounds different in episode 5 than in episode 1, or if the script calls
"Inspector Rathore" just "Rathore" and he gets a new voice. So a series keeps
a cast list, the "series bible": every character's name, gender,
description and voice, fixed the first time the character speaks.

Each episode's script writer is told who already exists and must reuse those
exact names. Each episode's voice engine is given the existing voices, and
new characters get a voice no one in the series has yet.

Usage:
    python make_series.py stories/haveli
    python make_series.py stories/haveli --languages en-IN ta-IN --check

stories/haveli/ holds one text file per episode (ep01.txt, ep02.txt, ...),
processed in file-name order. Output:

    output/haveli/series.json        the series bible
    output/haveli/ep01/hi-IN/episode.wav
    output/haveli/ep01/en-IN/episode.wav
    ...
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import sarvam_client
from make_drama import OUTPUT_DIR, run_pipeline
from quality_check import summary_row
from script_writer import LANGUAGES, NARRATOR, Character


def load_bible(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"characters": []}


def voice_consistency(results: list[dict]) -> dict[str, set[str]]:
    """Every voice each character was given, across all episodes and languages.

    The point of the bible is that each set has exactly one voice in it.
    """
    voices: dict[str, set[str]] = {}
    for result in results:
        for episode in result["episodes"]:
            for name, voice in episode["manifest"]["voices"].items():
                voices.setdefault(name, set()).add(voice)
    return voices


def run_series(
    series_dir: Path,
    languages: list[str],
    check: bool = False,
    fresh: bool = False,
    log: Callable[[str], None] = print,
) -> tuple[list[tuple[str, dict]], dict]:
    """Run every episode of a series in order. Returns the results and the bible."""
    episode_files = sorted(series_dir.glob("*.txt"))
    if not episode_files:
        raise ValueError(f"No episode files (*.txt) found in {series_dir}")

    bible_path = OUTPUT_DIR / series_dir.name / "series.json"
    bible = load_bible(bible_path)
    results = []

    for episode_file in episode_files:
        log(f"Episode {episode_file.stem}")
        known = [
            Character(name=c["name"], gender=c["gender"], description=c["description"])
            for c in bible["characters"]
        ]
        cast = {c["name"]: c["voice"] for c in bible["characters"]}

        result = run_pipeline(
            episode_file.read_text(encoding="utf-8"),
            f"{series_dir.name}/{episode_file.stem}",
            languages,
            check=check,
            fresh=fresh,
            log=log,
            known=known,
            cast=cast,
        )
        results.append((episode_file.stem, result))

        # Add this episode's new characters to the bible, with their voice.
        voices = result["episodes"][0]["manifest"]["voices"]
        for character in result["script"].characters:
            if character.name not in cast:
                bible["characters"].append(
                    {
                        **character.model_dump(),
                        "voice": voices[character.name],
                        "first_episode": episode_file.stem,
                    }
                )
                log(f"New character: {character.name} ({voices[character.name]})")
        bible_path.parent.mkdir(parents=True, exist_ok=True)
        bible_path.write_text(json.dumps(bible, ensure_ascii=False, indent=2), encoding="utf-8")

    return results, bible


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Turn a multi-episode series into audio dramas.")
    parser.add_argument("series_dir", type=Path, help="folder with one .txt file per episode")
    parser.add_argument(
        "--languages", nargs="*", default=[], metavar="CODE",
        help=f"extra languages to translate into: {', '.join(LANGUAGES)}",
    )
    parser.add_argument("--fresh", action="store_true", help="rewrite every episode's script")
    parser.add_argument("--check", action="store_true", help="run the quality checker")
    args = parser.parse_args()

    try:
        results, bible = run_series(
            args.series_dir, args.languages, check=args.check, fresh=args.fresh,
            log=lambda message: print(f">> {message}"),
        )
    except (sarvam_client.SarvamError, RuntimeError, ValueError) as error:
        sys.exit(f"Error: {error}")

    print("\nCast (series bible):")
    print(f"  {NARRATOR:<20} shubh")
    for c in bible["characters"]:
        print(f"  {c['name']:<20} {c['voice']:<8} {c['gender']:<7} from {c['first_episode']}: "
              f"{c['description']}")

    print("\nEpisodes:")
    for stem, result in results:
        for episode in result["episodes"]:
            minutes, seconds = divmod(round(episode["manifest"]["duration"]), 60)
            print(f"  {stem}  {LANGUAGES[episode['code']]:<9} {minutes}m {seconds:02d}s  "
                  f"{episode['script'].title}")

    voices = voice_consistency([result for _, result in results])
    changed = {name: sorted(v) for name, v in voices.items() if len(v) > 1}
    print("\nVoices consistent across all episodes and languages: "
          + ("yes" if not changed else f"NO, {changed}"))

    if args.check:
        print("\nQuality (before -> after):")
        for stem, result in results:
            for episode in result["episodes"]:
                print(f"  {stem}  " + summary_row(episode["report"]))


if __name__ == "__main__":
    main()
