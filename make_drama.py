"""Run the whole pipeline with one command:

    story text -> script -> translations -> one audio episode per language

Usage:
    python make_drama.py stories/sample_story.txt
    python make_drama.py stories/sample_story.txt --languages hi-IN ta-IN bn-IN
    python make_drama.py stories/sample_story.txt --languages hi-IN --check
    python make_drama.py stories/sample_story.txt --fresh

Everything for a story lands in one folder:

    output/sample_story/
        script.json               the script in the story's own language
        en-IN/episode.wav         one folder per language, each with its
        hi-IN/episode.wav         script.json, clips, manifest.json, episode.wav
        ...

Finished work is reused. Running the command again only calls Sarvam for what
is new, for example an added language. Use --fresh to rewrite the script.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import sarvam_client
from quality_check import check_episode, summary_row
from script_writer import LANGUAGES, Character, Script, write_script
from translator import translate_script
from tts_engine import make_episode

OUTPUT_DIR = Path("output")


def save_script(script: Script, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(script.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_pipeline(
    story: str,
    name: str,
    languages: list[str],
    check: bool = False,
    fresh: bool = False,
    log: Callable[[str], None] = print,
    known: list[Character] | None = None,
    cast: dict[str, str] | None = None,
) -> dict:
    """Run every stage for one story. Used by the command line and the web app.

    name is the folder under output/ that holds this story's files.
    log receives one-line progress messages.
    known and cast carry characters and voices from earlier episodes of a
    series (see make_series.py).
    Returns the source script and one entry per language.
    """
    unknown = [code for code in languages if code not in LANGUAGES]
    if unknown:
        raise ValueError(f"Unsupported language {unknown}. Choose from: {', '.join(LANGUAGES)}")

    story_dir = OUTPUT_DIR / name
    script_path = story_dir / "script.json"

    # Step 1: the script (reused if it already exists)
    if script_path.exists() and not fresh:
        script = Script.model_validate_json(script_path.read_text(encoding="utf-8"))
        log(f"Using the existing script ({len(script.lines)} lines)")
    else:
        log("Writing the script...")
        script, dropped = write_script(story, known=known)
        save_script(script, script_path)
        log(f"Script written: {len(script.lines)} lines, {len(dropped)} speech tags dropped")

    # Steps 2 to 4: translate, speak, and optionally check, one language at a time
    episodes = []
    for code in [script.language] + [c for c in languages if c != script.language]:
        folder = story_dir / code
        if code != script.language:
            log(f"{LANGUAGES[code]}: translating...")
        version = translate_script(script, code, folder / "translation_cache.json")
        save_script(version, folder / "script.json")

        log(f"{LANGUAGES[code]}: generating audio...")
        make_episode(version, folder, cast)

        report = None
        if check:
            log(f"{LANGUAGES[code]}: checking quality...")
            report = check_episode(folder)
        elif (folder / "quality_report.json").exists():
            report = json.loads((folder / "quality_report.json").read_text(encoding="utf-8"))

        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        episodes.append(
            {
                "code": code,
                "folder": folder,
                "script": version,
                "manifest": manifest,
                "report": report,
            }
        )
    return {"script": script, "episodes": episodes}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Turn a story into a multilingual audio drama.")
    parser.add_argument("story_file", type=Path, help="text file containing the story")
    parser.add_argument(
        "--languages", nargs="*", default=[], metavar="CODE",
        help=f"extra languages to translate into: {', '.join(LANGUAGES)}",
    )
    parser.add_argument("--fresh", action="store_true", help="rewrite the script from the story")
    parser.add_argument(
        "--check", action="store_true",
        help="run the quality checker on every episode and re-generate bad takes",
    )
    args = parser.parse_args()

    try:
        result = run_pipeline(
            args.story_file.read_text(encoding="utf-8"),
            args.story_file.stem,
            args.languages,
            check=args.check,
            fresh=args.fresh,
            log=lambda message: print(f"\n>> {message}"),
        )
    except (sarvam_client.SarvamError, RuntimeError, ValueError) as error:
        sys.exit(f"Error: {error}")

    print("\nDone:")
    for episode in result["episodes"]:
        minutes, seconds = divmod(round(episode["manifest"]["duration"]), 60)
        print(f"  {LANGUAGES[episode['code']]:<10} {minutes}m {seconds:02d}s  "
              f"{episode['folder'] / 'episode.wav'}  ({episode['script'].title})")
    if args.check:
        print("\nQuality (before -> after):")
        for episode in result["episodes"]:
            print("  " + summary_row(episode["report"]))


if __name__ == "__main__":
    main()
