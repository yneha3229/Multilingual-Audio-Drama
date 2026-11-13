"""Translation checker: does each translated line still mean the same thing?

The audio quality checker compares speech with the translated script, so it
cannot see a bad translation. This checks the translation itself, two ways:

    1. Back-translation score. Sarvam's translation API (Mayura) translates
       each line back into the source language, and chrF measures how close
       that comes to the original line (0 to 100, higher is closer). Mayura is
       a different system from the chat model that did the translation, so
       the two are less likely to make the same mistake and hide it.
    2. LLM judge. The chat model sees the original, the translation, and the
       back-translation for every line, and marks each one:
           ok     - same meaning (different wording is fine)
           minor  - a small nuance lost or added; the story is unchanged
           wrong  - the meaning changed: who did what, a fact, a name,
                    a character's gender, or content missing or added

This only reports. It does not change any translation.

Usage:
    python translation_check.py output/sample_story/hi-IN output/haveli/ep01/en-IN
"""

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

import sarvam_client
from quality_check import normalize
from script_writer import LANGUAGES, NARRATOR, Script

BACK_MODEL = "mayura:v1"
NARRATOR_GENDER = "male"  # the narrator voice in tts_engine.py is male
MAYURA_MAX_CHARS = 1000


# --------------------------------------------------------------------------
# 1. chrF: character n-gram overlap between two texts
# --------------------------------------------------------------------------


def chrf(reference: str, hypothesis: str, max_n: int = 6, beta: float = 2.0) -> float:
    """chrF score from 0 to 100 (Popovic, 2015), on normalized text.

    Counts shared character sequences of length 1 to 6, ignoring spaces.
    Works for any script, which word-based scores like BLEU do poorly on.
    beta = 2 weighs recall (is all of the original there?) above precision.
    """
    ref = normalize(reference).replace(" ", "")
    hyp = normalize(hypothesis).replace(" ", "")
    precisions, recalls = [], []
    for n in range(1, max_n + 1):
        ref_grams = Counter(ref[i : i + n] for i in range(len(ref) - n + 1))
        hyp_grams = Counter(hyp[i : i + n] for i in range(len(hyp) - n + 1))
        if not ref_grams or not hyp_grams:
            continue
        overlap = sum((ref_grams & hyp_grams).values())
        precisions.append(overlap / sum(hyp_grams.values()))
        recalls.append(overlap / sum(ref_grams.values()))
    if not precisions:
        return 100.0 if ref == hyp else 0.0
    p, r = statistics.mean(precisions), statistics.mean(recalls)
    if p + r == 0:
        return 0.0
    return 100 * (1 + beta**2) * p * r / (beta**2 * p + r)


# --------------------------------------------------------------------------
# 2. Back-translation with Mayura
# --------------------------------------------------------------------------


def split_for_mayura(text: str, limit: int = MAYURA_MAX_CHARS) -> list[str]:
    """Mayura takes at most 1000 characters, so split long lines at sentence ends."""
    if len(text) <= limit:
        return [text]
    sentences = re.split(r"(?<=[.!?।])\s+", text)
    pieces, current = [], ""
    for sentence in sentences:
        if current and len(current) + 1 + len(sentence) > limit:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    pieces.append(current)
    return pieces


def back_translate(text: str, source: str, target: str, gender: str, cache: dict) -> str:
    key = f"{source}|{target}|{gender}|{text}"
    if key not in cache:
        parts = []
        for piece in split_for_mayura(text):
            data = sarvam_client.post(
                "/translate",
                {
                    "model": BACK_MODEL,
                    "input": piece,
                    "source_language_code": source,
                    "target_language_code": target,
                    "mode": "formal",  # literal, so meaning changes are not smoothed over
                    "speaker_gender": gender.capitalize(),
                },
            )
            parts.append(data["translated_text"].strip())
        cache[key] = " ".join(parts)
    return cache[key]


# --------------------------------------------------------------------------
# 3. The LLM judge
# --------------------------------------------------------------------------


class Verdict(BaseModel):
    id: int
    reason: str  # written first, so the verdict follows from the reasoning
    verdict: Literal["ok", "minor", "wrong"]


class Verdicts(BaseModel):
    verdicts: list[Verdict]


JUDGE_PROMPT = """You check translations of an audio-drama script. You read both languages fluently.

For every line you get the original and the translation. Compare them directly.

Write "reason" first: one short sentence in English. Then give the verdict that follows from it. Give each line one verdict:
- "ok": the translation means the same as the original. Different wording, natural idioms, and dropped or added filler words are fine.
- "minor": a small nuance is lost or added, but a listener would understand the same story.
- "wrong": the meaning changed. For example: who did what to whom is reversed, a fact or number is different, a name is wrong, a character's grammatical gender is wrong, or something important is missing or invented.

Use the character list to check gender. Remember that some languages do not mark gender in pronouns or verbs (Bengali "সে", Hindi "उसने", "उससे"), so a pronoun that is neutral in the translation is not a gender error. For "ok" the reason can just say "Same meaning."

Return one verdict per line, with the same ids, in order."""


def judge(lines: list[dict], characters: str, max_attempts: int = 3) -> list[Verdict]:
    """Ask the chat model for a verdict on every line, in one request."""
    listing = "\n".join(json.dumps(line, ensure_ascii=False) for line in lines)
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": f"Characters:\n{characters}\n\nLines:\n{listing}"},
    ]
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "verdicts", "schema": Verdicts.model_json_schema()},
    }
    expected = [line["id"] for line in lines]
    for attempt in range(1, max_attempts + 1):
        try:
            raw = sarvam_client.chat(messages, response_format=response_format,
                                     temperature=0.0, reasoning_effort="low", max_tokens=8000)
            verdicts = Verdicts.model_validate_json(raw).verdicts
        except (sarvam_client.OutOfTokens, ValidationError) as error:
            print(f"Judge attempt {attempt}/{max_attempts} failed: {error}", file=sys.stderr)
            continue
        if [v.id for v in verdicts] == expected:
            return verdicts
        print(f"Judge attempt {attempt}/{max_attempts} returned the wrong ids", file=sys.stderr)
    raise RuntimeError(f"No valid verdicts after {max_attempts} attempts.")


# --------------------------------------------------------------------------
# 4. Checking one translated script
# --------------------------------------------------------------------------


def check_translation(folder: Path) -> dict:
    """Check output/<story>/<language>/script.json against output/<story>/script.json."""
    source = Script.model_validate_json((folder.parent / "script.json").read_text(encoding="utf-8"))
    target = Script.model_validate_json((folder / "script.json").read_text(encoding="utf-8"))
    if target.language == source.language:
        raise ValueError(f"{folder} is the original, not a translation")
    if len(source.lines) != len(target.lines):
        raise ValueError(f"{folder}: the translation has a different number of lines")

    cache_file = folder / "backtranslation_cache.json"
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
    genders = {c.name: c.gender for c in source.characters} | {NARRATOR: NARRATOR_GENDER}

    rows = []
    for number, (original, translated) in enumerate(zip(source.lines, target.lines), start=1):
        back = back_translate(translated.text, target.language, source.language,
                              genders[original.speaker], cache)
        cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append({
            "id": number,
            "speaker": original.speaker,
            "original": original.text,
            "translation": translated.text,
            "back_translation": back,
            "chrf": round(chrf(original.text, back), 1),
        })

    characters = "\n".join(f"- {c.name}: {c.gender}, {c.description}" for c in source.characters)
    verdicts = judge(
        # The judge does not see the back-translation: in a first version it
        # did, and it blamed the translation for the back-translation's own
        # mistakes (7 of 8 checked "wrong" verdicts).
        [{k: row[k] for k in ("id", "speaker", "original", "translation")} for row in rows],
        characters,
    )
    for row, verdict in zip(rows, verdicts):
        row["verdict"], row["reason"] = verdict.verdict, verdict.reason

    counts = Counter(row["verdict"] for row in rows)
    report = {
        "source_language": source.language,
        "language": target.language,
        "summary": {
            "lines": len(rows),
            "mean_chrf": round(statistics.mean(row["chrf"] for row in rows), 1),
            "ok": counts["ok"],
            "minor": counts["minor"],
            "wrong": counts["wrong"],
        },
        "lines": rows,
    }
    (folder / "translation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Check translated scripts by back-translation.")
    parser.add_argument("folders", type=Path, nargs="+",
                        help="translated episode folders, e.g. output/sample_story/ta-IN")
    args = parser.parse_args()

    reports = []
    for folder in args.folders:
        print(f"\n=== {folder} ===")
        try:
            report = check_translation(folder)
        except (sarvam_client.SarvamError, RuntimeError, ValueError) as error:
            sys.exit(f"Error: {error}")
        reports.append((folder, report))
        for row in report["lines"]:
            if row["verdict"] != "ok":
                print(f"  line {row['id']} [{row['verdict']}] chrF {row['chrf']}: {row['reason']}")

    print("\nSummary:")
    for folder, report in reports:
        s = report["summary"]
        print(f"  {str(folder):<32} {LANGUAGES[report['language']]:<9} lines {s['lines']:>3}  "
              f"chrF {s['mean_chrf']:>5}  ok {s['ok']:>3}  minor {s['minor']:>2}  wrong {s['wrong']:>2}")


if __name__ == "__main__":
    main()
