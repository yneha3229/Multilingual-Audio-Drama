"""Phase 4: the quality checker.

Text-to-speech is random: the same line with the same settings can come out
clean in one take and garbled or halting in the next. This stage finds the bad
takes and replaces them, with no human listening.

For every clip in an episode:

    1. Transcribe it back to text with Sarvam speech-to-text.
    2. Compare the transcript with the script line:
         WER = word error rate, CER = character error rate (0.0 is perfect).
    3. Measure the delivery: how fast the line is spoken compared with the
       rest of the episode. Much slower means long, halting pauses.
    4. If a clip fails either check, generate new takes and keep the best one.

Then the episode is stitched again and quality_report.json records the scores
before and after.

Usage:
    python quality_check.py output/sample_story/hi-IN
"""

import argparse
import array
import hashlib
import io
import json
import statistics
import sys
import unicodedata
import wave
from pathlib import Path

import sarvam_client
import tts_engine

STT_MODEL = "saaras:v4"
MAX_CER = 0.15  # a clip fails if more than 15% of its characters are wrong
MIN_RELATIVE_RATE = 0.70  # ...or if it is spoken at under 70% of the usual speed
MIN_LETTERS_FOR_RATE = 15  # lines shorter than this are too short to judge speed
MAX_RETRIES = 2  # extra takes to try for a failing line
SLOW_PENALTY = 0.5  # how much "too slow" counts when comparing takes
MAX_STT_SECONDS = 28  # Sarvam's limit is 30 s; keep a margin


# --------------------------------------------------------------------------
# 1. Comparing text: WER and CER
# --------------------------------------------------------------------------


def normalize(text: str) -> str:
    """Remove differences that are not pronunciation errors.

    The speech-to-text model writes things its own way: no exclamation marks,
    "यहां" where the script has "यहाँ", ASCII digits where a Bengali script has
    Bengali digits. None of those mean the audio was wrong, so they are
    evened out before comparing.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    cleaned = []
    for ch in text:
        category = unicodedata.category(ch)
        if ch == "ँ":  # Devanagari chandrabindu -> anusvara
            cleaned.append("ं")
        elif ch == "़":  # Devanagari nukta (ज़ vs ज): drop
            continue
        elif category == "Nd":  # any script's digit -> 0-9
            cleaned.append(str(unicodedata.digit(ch)))
        elif category[0] in "LMN":  # letters, vowel signs, numbers
            cleaned.append(ch)
        else:  # punctuation, symbols, whitespace
            cleaned.append(" ")
    return " ".join("".join(cleaned).split())


def edit_distance(reference: list, hypothesis: list) -> int:
    """Fewest insertions, deletions and substitutions to turn one into the other."""
    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (ref_item != hyp_item),  # substitution
                )
            )
        previous = current
    return previous[-1]


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate: word-level mistakes divided by words in the reference."""
    ref, hyp = normalize(reference).split(), normalize(hypothesis).split()
    return edit_distance(ref, hyp) / max(len(ref), 1)


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate, ignoring spaces.

    Fairer than WER for Indian languages, where one long word can carry a
    whole phrase and "சர்ச்கேட்ல" vs "சர்ச் கேட்ல" is a spacing choice, not an error.
    """
    ref = list(normalize(reference).replace(" ", ""))
    hyp = list(normalize(hypothesis).replace(" ", ""))
    return edit_distance(ref, hyp) / max(len(ref), 1)


# --------------------------------------------------------------------------
# 2. Measuring one clip
# --------------------------------------------------------------------------


WINDOW = 0.02  # seconds per loudness measurement


def quiet_windows(path: Path) -> tuple[list[bool], int, int]:
    """Split a clip into 20 ms windows and mark the silent ones.

    Returns (quiet flags, first voiced window, last voiced window).
    """
    with wave.open(str(path), "rb") as clip:
        rate = clip.getframerate()
        samples = array.array("h", clip.readframes(clip.getnframes()))
    window = int(rate * WINDOW)
    loudness = [
        (sum(s * s for s in samples[i : i + window]) / window) ** 0.5
        for i in range(0, len(samples) - window, window)
    ]
    if not loudness:
        return [], 0, -1
    threshold = 0.02 * sorted(loudness)[int(len(loudness) * 0.9)]
    quiet = [level < threshold for level in loudness]
    if False not in quiet:
        return quiet, 0, -1
    return quiet, quiet.index(False), len(quiet) - 1 - quiet[::-1].index(False)


def letter_count(text: str) -> int:
    return len(normalize(text).replace(" ", ""))


def speaking_rate(path: Path, text: str, pace: float) -> float:
    """Characters spoken per second, with the requested pace factored out.

    Only the voiced part of the clip counts: the silence the TTS model puts
    at the start and end would otherwise make every short line look slow.
    A sad line is asked to be slower (pace 0.85); dividing by pace means it
    is not punished for doing what it was told.
    """
    _, first, last = quiet_windows(path)
    seconds = max(last - first + 1, 1) * WINDOW
    return letter_count(text) / (seconds * pace)


def pause_seconds(path: Path, min_pause: float = 0.25) -> float:
    """Total length of silent gaps inside the clip (not at its start or end)."""
    quiet, first, last = quiet_windows(path)
    if last < first:
        return 0.0

    total, run = 0.0, 0
    for is_quiet in quiet[first : last + 1]:
        if is_quiet:
            run += 1
        else:
            if run * 0.02 >= min_pause:
                total += run * 0.02
            run = 0
    return total


def split_audio(samples: array.array, rate: int, max_seconds: float = MAX_STT_SECONDS) -> list:
    """Cut audio into pieces no longer than max_seconds, cutting at pauses.

    Each cut is made at the quietest moment in the last third of the allowed
    length, so a cut lands between words rather than in the middle of one.
    """
    window = rate // 50  # 20 ms
    limit = int(max_seconds * rate)
    pieces, start = [], 0
    while len(samples) - start > limit:
        search_from = start + int(limit * 2 / 3)
        search_to = start + limit - window
        quietest, best_level = search_to, None
        for i in range(search_from, search_to, window):
            level = sum(s * s for s in samples[i : i + window])
            if best_level is None or level < best_level:
                quietest, best_level = i, level
        cut = quietest + window // 2
        pieces.append(samples[start:cut])
        start = cut
    pieces.append(samples[start:])
    return pieces


def _wav_bytes(samples: array.array, rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(samples.tobytes())
    return buffer.getvalue()


def transcribe(path: Path, language: str, cache: dict) -> str:
    """Speech-to-text for one clip. cache avoids paying twice for the same audio.

    Sarvam's speech-to-text accepts at most 30 seconds, and a long narration
    line in a wordy language can run past that, so long clips are cut at
    pauses and the pieces' transcripts are joined.
    """
    audio = path.read_bytes()
    key = hashlib.sha1(audio).hexdigest()
    if key not in cache:
        with wave.open(str(path), "rb") as clip:
            rate = clip.getframerate()
            samples = array.array("h", clip.readframes(clip.getnframes()))
        pieces = [audio] if len(samples) <= MAX_STT_SECONDS * rate else [
            _wav_bytes(piece, rate) for piece in split_audio(samples, rate)
        ]
        transcripts = []
        for number, piece in enumerate(pieces, start=1):
            data = sarvam_client.post(
                "/speech-to-text",
                {"model": STT_MODEL, "language_code": language},
                files={"file": (f"{path.stem}_part{number}.wav", piece, "audio/wav")},
            )
            transcripts.append(data["transcript"].strip())
        cache[key] = " ".join(transcripts)
    return cache[key]


def measure(path: Path, entry: dict, language: str, cache: dict) -> dict:
    """All the numbers for one take of one line."""
    transcript = transcribe(path, language, cache)
    return {
        "transcript": transcript,
        "wer": round(wer(entry["text"], transcript), 4),
        "cer": round(cer(entry["text"], transcript), 4),
        "rate": round(speaking_rate(path, entry["text"], entry["pace"]), 2),
        "pauses": round(pause_seconds(path), 2),
        "letters": letter_count(entry["text"]),
    }


def is_slow(result: dict, usual_rate: float) -> bool:
    """Too slow compared with similar lines. Very short lines ("Buyers?") are
    not judged: a single word is naturally drawn out."""
    if result["letters"] < MIN_LETTERS_FOR_RATE:
        return False
    return result["rate"] < MIN_RELATIVE_RATE * usual_rate


def judge(result: dict, usual_rate: float) -> list[str]:
    """Return the reasons a take fails. An empty list means it passes."""
    reasons = []
    if result["cer"] > MAX_CER:
        reasons.append("misread")
    if is_slow(result, usual_rate):
        reasons.append("slow")
    return reasons


def badness(result: dict, usual_rate: float) -> float:
    """One number for comparing takes of the same line. Lower is better."""
    return result["cer"] + (SLOW_PENALTY if is_slow(result, usual_rate) else 0.0)


def usual_rates(entries: list[dict], results: list[dict]) -> dict[str, float]:
    """The typical speaking rate for narration and for dialogue, separately.

    The narrator reads in a steady, quick voice; characters act, and speak
    more slowly. Comparing a character's line with the narrator's speed
    flags perfectly good acting as "slow".
    """
    groups: dict[str, list[float]] = {"narration": [], "dialogue": []}
    for entry, result in zip(entries, results):
        if result["letters"] >= MIN_LETTERS_FOR_RATE:
            groups[rate_group(entry)].append(result["rate"])
    everything = groups["narration"] + groups["dialogue"] or [r["rate"] for r in results]
    overall = statistics.median(everything)
    return {
        name: statistics.median(rates) if len(rates) >= 3 else overall
        for name, rates in groups.items()
    }


def rate_group(entry: dict) -> str:
    return "narration" if entry["speaker"] == "Narrator" else "dialogue"


# --------------------------------------------------------------------------
# 3. Checking and repairing a whole episode
# --------------------------------------------------------------------------


def check_episode(out_dir: Path, max_retries: int = MAX_RETRIES) -> dict:
    """Check every clip of an episode, re-generate the failing ones, and
    write quality_report.json. Returns the report."""
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    language = manifest["language"]
    cache_file = out_dir / "stt_cache.json"
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}

    def save_cache():
        cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    takes_dir = out_dir / "clips" / "takes"
    rejected_dir = out_dir / "clips" / "rejected"

    def first_take(entry: dict) -> Path:
        """The very first take of a line. If an earlier run of this checker
        replaced it, that original now sits in clips/rejected."""
        clip = out_dir / entry["clip"]
        original = rejected_dir / clip.name
        return original if original.exists() else clip

    # Pass 1: measure every line's first take. "Before" always means the first
    # take, so running the checker twice does not hide what it fixed earlier.
    entries = manifest["lines"]
    before = []
    for entry in entries:
        before.append(measure(first_take(entry), entry, language, cache))
        save_cache()
    rates = usual_rates(entries, before)

    # Pass 2: retry the failing lines and keep the best take.
    report_lines = []
    replaced = False

    for entry, first in zip(entries, before):
        usual_rate = rates[rate_group(entry)]
        clip = out_dir / entry["clip"]
        reasons = judge(first, usual_rate)
        # Start from the clip currently in the episode (free: it is in the cache).
        best, best_path, tries = measure(clip, entry, language, cache), clip, 0

        while judge(best, usual_rate) and tries < max_retries:
            tries += 1
            take_path = takes_dir / f"{clip.stem}_take{tries + 1}.wav"
            takes_dir.mkdir(parents=True, exist_ok=True)
            take_path.write_bytes(
                tts_engine.synthesize(
                    entry["text"], entry["voice"], language, entry["pace"], entry["temperature"]
                )
            )
            result = measure(take_path, entry, language, cache)
            save_cache()
            if badness(result, usual_rate) < badness(best, usual_rate):
                best, best_path = result, take_path

        if best_path != clip:
            # Keep the old take for before/after comparison, then swap in the new one.
            rejected_dir.mkdir(parents=True, exist_ok=True)
            if not (rejected_dir / clip.name).exists():  # never overwrite the first take
                clip.replace(rejected_dir / clip.name)
            best_path.replace(clip)
            replaced = True

        status = "ok" if not reasons else ("fixed" if not judge(best, usual_rate) else "still failing")
        print(f"  {entry['line']:>2} {entry['speaker']:<10} CER {first['cer']:>5.1%}"
              f"  rate {first['rate']:>4.1f}  {status}"
              + (f" ({', '.join(reasons)}; {tries} retries; CER now {best['cer']:.1%},"
                 f" rate {best['rate']:.1f})" if reasons else ""))
        report_lines.append(
            {
                "line": entry["line"],
                "speaker": entry["speaker"],
                "text": entry["text"],
                "status": status,
                "problems": reasons,
                "retries": tries,
                "before": first,
                "after": best,
            }
        )

    if replaced:
        tts_engine.stitch_episode(manifest, out_dir)

    def mean(stage: str, metric: str) -> float:
        return round(statistics.mean(line[stage][metric] for line in report_lines), 4)

    def count(status: str) -> int:
        return sum(line["status"] == status for line in report_lines)

    report = {
        "language": language,
        "settings": {
            "stt_model": STT_MODEL,
            "max_cer": MAX_CER,
            "min_relative_rate": MIN_RELATIVE_RATE,
            "min_letters_for_rate": MIN_LETTERS_FOR_RATE,
            "max_retries": max_retries,
        },
        "summary": {
            "lines": len(report_lines),
            "usual_rate": {group: round(rate, 2) for group, rate in rates.items()},
            "failing_before": len(report_lines) - count("ok"),
            "fixed": count("fixed"),
            "failing_after": count("still failing"),
            "wer_before": mean("before", "wer"),
            "wer_after": mean("after", "wer"),
            "cer_before": mean("before", "cer"),
            "cer_after": mean("after", "cer"),
        },
        "lines": report_lines,
    }
    (out_dir / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def summary_row(report: dict) -> str:
    s = report["summary"]
    return (
        f"{report['language']:<6} lines {s['lines']:>3}   failing {s['failing_before']:>2} -> "
        f"{s['failing_after']:<2}   WER {s['wer_before']:>5.1%} -> {s['wer_after']:<5.1%}   "
        f"CER {s['cer_before']:>5.1%} -> {s['cer_after']:<5.1%}"
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Check an episode's clips and fix bad takes.")
    parser.add_argument("episode_dirs", type=Path, nargs="+",
                        help="episode folders, e.g. output/sample_story/hi-IN")
    parser.add_argument("--retries", type=int, default=MAX_RETRIES,
                        help="extra takes to try for a failing line (0 = only measure)")
    args = parser.parse_args()

    reports = []
    for folder in args.episode_dirs:
        print(f"\n=== {folder} ===")
        try:
            reports.append(check_episode(folder, args.retries))
        except (sarvam_client.SarvamError, FileNotFoundError, ValueError) as error:
            sys.exit(f"Error: {error}")

    print("\nSummary (before -> after):")
    for report in reports:
        print("  " + summary_row(report))


if __name__ == "__main__":
    main()
