"""Phase 2: the voice engine.

Turns a script (from script_writer.py) into one audio episode:

    script JSON -> pick a voice per character -> one audio clip per line
                -> join the clips with short pauses -> episode.wav

Audio is requested as WAV and joined with Python's built-in wave module, so
this stage needs no extra libraries and no ffmpeg.

Usage:
    python tts_engine.py output/sample_story_script.json
"""

import argparse
import base64
import hashlib
import io
import json
import sys
import wave
from pathlib import Path

import sarvam_client
from script_writer import NARRATOR, Script

TTS_MODEL = "bulbul:v3"
SAMPLE_RATE = 22050

# Bulbul v3 voices. Sarvam's docs list the names but not the genders, so this
# grouping is inferred from the names. Listen to a clip and move any that are
# wrong. Order matters: characters get voices from the front of each list.
NARRATOR_VOICE = "shubh"
MALE_VOICES = ["aditya", "rahul", "rohan", "amit", "dev", "varun", "kabir", "ratan", "anand"]
FEMALE_VOICES = ["priya", "neha", "ritu", "pooja", "simran", "kavya", "ishita", "shreya"]

# Bulbul has no "angry" or "sad" setting. We approximate emotion with:
#   pace        - speaking speed (0.5 to 2.0, 1.0 is normal)
#   temperature - expressiveness (0.01 to 1.0, Sarvam's default is 0.6)
EMOTION_STYLE = {
    "neutral":    {"pace": 1.00, "temperature": 0.5},
    "calm":       {"pace": 0.95, "temperature": 0.5},
    "happy":      {"pace": 1.05, "temperature": 0.7},
    "excited":    {"pace": 1.15, "temperature": 0.8},
    "angry":      {"pace": 1.10, "temperature": 0.8},
    "sad":        {"pace": 0.85, "temperature": 0.7},
    "fearful":    {"pace": 1.10, "temperature": 0.8},
    "whispering": {"pace": 0.85, "temperature": 0.6},
}

PAUSE_SAME_SPEAKER = 0.30  # seconds of silence between lines
PAUSE_NEW_SPEAKER = 0.55


# --------------------------------------------------------------------------
# 1. Casting: one fixed voice per character
# --------------------------------------------------------------------------


def assign_voices(script: Script, cast: dict[str, str] | None = None) -> dict[str, str]:
    """Give the narrator and every character a voice, in order of appearance.

    cast holds voices already given out in earlier episodes of a series.
    Those characters keep their voice, and new characters get a voice nobody
    in the series has yet.

    The same script always produces the same casting, so a character sounds
    the same across the episode and across re-runs.
    """
    pools = {"male": MALE_VOICES, "female": FEMALE_VOICES}
    voices = {NARRATOR: NARRATOR_VOICE, **(cast or {})}
    for character in script.characters:
        if character.name in voices:
            continue
        pool = pools[character.gender]
        taken = set(voices.values())
        free = [voice for voice in pool if voice not in taken]
        # If a series has more characters than voices, start reusing them.
        same_gender = sum(voice in pool for voice in voices.values())
        voices[character.name] = free[0] if free else pool[same_gender % len(pool)]
    # Only return the voices this script needs.
    needed = {NARRATOR} | {c.name for c in script.characters}
    return {name: voice for name, voice in voices.items() if name in needed}


# --------------------------------------------------------------------------
# 2. One line -> one audio clip
# --------------------------------------------------------------------------


def synthesize(text: str, voice: str, language: str, pace: float, temperature: float) -> bytes:
    """Ask Sarvam to speak one line. Returns the bytes of a WAV file."""
    data = sarvam_client.post(
        "/text-to-speech",
        {
            "model": TTS_MODEL,
            "text": text,
            "language_code": language,
            "speaker": voice,
            "pace": pace,
            "temperature": temperature,
            "speech_sample_rate": SAMPLE_RATE,
            "output_audio_codec": "wav",
        },
    )
    # The audio arrives as text (base64). Decode it back into real bytes.
    return base64.b64decode("".join(data["audios"]))


def clip_path(clips_dir: Path, number: int, speaker: str, request: dict) -> Path:
    """File name for a clip. The short hash changes whenever the text, voice,
    or style changes, so an old clip is never reused for a different request."""
    fingerprint = hashlib.sha1(json.dumps(request, sort_keys=True).encode()).hexdigest()[:8]
    safe_speaker = "".join(ch if ch.isalnum() else "_" for ch in speaker.lower())
    return clips_dir / f"{number:03d}_{safe_speaker}_{fingerprint}.wav"


def read_wav(path: Path) -> tuple[bytes, float]:
    """Return a clip's raw audio samples and its length in seconds."""
    with wave.open(str(path), "rb") as clip:
        if (clip.getframerate(), clip.getnchannels(), clip.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise ValueError(f"{path.name} is not {SAMPLE_RATE} Hz mono 16-bit audio")
        frames = clip.readframes(clip.getnframes())
        return frames, clip.getnframes() / SAMPLE_RATE


def silence(seconds: float) -> bytes:
    return b"\x00\x00" * int(SAMPLE_RATE * seconds)  # 2 zero bytes per sample


# --------------------------------------------------------------------------
# 3. The whole episode
# --------------------------------------------------------------------------


def make_episode(script: Script, out_dir: Path, cast: dict[str, str] | None = None) -> dict:
    """Generate every line's clip, join them, and write episode.wav.

    cast holds voices from earlier episodes of a series, if any.
    Returns a manifest describing each line (voice, pace, clip file, timing).
    Phase 4 (the quality checker) reads this manifest.
    """
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    voices = assign_voices(script, cast)

    entries = []
    for number, line in enumerate(script.lines, start=1):
        style = EMOTION_STYLE[line.emotion]
        request = {
            "text": line.text,
            "voice": voices[line.speaker],
            "language": script.language,
            "pace": style["pace"],
            "temperature": style["temperature"],
        }
        path = clip_path(clips_dir, number, line.speaker, request)

        if path.exists():
            status = "cached"  # already generated on an earlier run
        else:
            path.write_bytes(synthesize(**request))
            status = "generated"

        _, duration = read_wav(path)
        print(f"  {number:>2}/{len(script.lines)} {status:<9} {line.speaker} "
              f"({request['voice']}, {line.emotion}, {duration:.1f}s)")
        entries.append(
            {
                "line": number,
                "speaker": line.speaker,
                "voice": request["voice"],
                "emotion": line.emotion,
                "pace": request["pace"],
                "temperature": request["temperature"],
                "text": line.text,
                "clip": str(path.relative_to(out_dir).as_posix()),
            }
        )

    manifest = {
        "title": script.title,
        "language": script.language,
        "voices": voices,
        "episode": "episode.wav",
        "lines": entries,
    }
    stitch_episode(manifest, out_dir)
    return manifest


def stitch_episode(manifest: dict, out_dir: Path) -> None:
    """Join the manifest's clips into episode.wav and save the manifest.

    Fills in each line's start time and duration. The quality checker calls
    this again after it replaces a clip with a better take.
    """
    episode_audio = io.BytesIO()
    position = 0.0  # seconds from the start of the episode
    previous_speaker = None

    for entry in manifest["lines"]:
        frames, duration = read_wav(out_dir / entry["clip"])
        if previous_speaker is not None:
            same = entry["speaker"] == previous_speaker
            pause = PAUSE_SAME_SPEAKER if same else PAUSE_NEW_SPEAKER
            episode_audio.write(silence(pause))
            position += pause
        episode_audio.write(frames)
        entry["start"] = round(position, 2)
        entry["duration"] = round(duration, 2)
        position += duration
        previous_speaker = entry["speaker"]

    with wave.open(str(out_dir / manifest["episode"]), "wb") as episode:
        episode.setnchannels(1)
        episode.setsampwidth(2)
        episode.setframerate(SAMPLE_RATE)
        episode.writeframes(episode_audio.getvalue())

    manifest["duration"] = round(position, 2)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# --------------------------------------------------------------------------
# 4. Command line
# --------------------------------------------------------------------------


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Turn a script JSON into an audio episode.")
    parser.add_argument("script_file", type=Path, help="script JSON made by script_writer.py")
    parser.add_argument("-o", "--output", type=Path, help="folder for the episode and clips")
    args = parser.parse_args()

    script = Script.model_validate_json(args.script_file.read_text(encoding="utf-8"))
    if args.output:
        out_dir = args.output
    elif args.script_file.parent.name == script.language:
        out_dir = args.script_file.parent  # e.g. output/sample_story/hi-IN/script.json
    else:
        default_name = args.script_file.stem.removesuffix("_script")
        out_dir = args.script_file.parent / default_name / script.language

    print(f"{script.title} ({script.language}), {len(script.lines)} lines")
    try:
        manifest = make_episode(script, out_dir)
    except (sarvam_client.SarvamError, ValueError) as error:
        sys.exit(f"Error: {error}")

    print("\nCast:")
    for name, voice in manifest["voices"].items():
        print(f"  {name}: {voice}")
    minutes, seconds = divmod(round(manifest["duration"]), 60)
    print(f"\nEpisode length: {minutes}m {seconds:02d}s")
    print(f"Saved to {out_dir / manifest['episode']}")


if __name__ == "__main__":
    main()
