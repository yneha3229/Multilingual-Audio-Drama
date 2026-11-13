"""Background music: score an episode with Stable Audio 3 and mix it under the voices.

    script.json --LLM--> music prompt --Stable Audio 3--> music.wav
    episode.wav + music.wav + manifest timings --mix--> episode_with_music.wav

1. Prompt. The chat model reads the episode and writes one short English
   prompt for a text-to-music model (genre, instruments, tempo, mood). Code
   checks it: English only, no vocals, no character names, sensible length.
   On a bad prompt the model is shown the problem and asked again.
2. Music. music_worker.py runs Stable Audio 3 Small-Music in its own
   environment (it needs PyTorch, which the main project does not). The file
   is named by a hash of the prompt, length and seed, so it is made once.
3. Mix. The manifest already says when every line starts and how long it
   lasts, so the music can drop under speech and come back up in the gaps
   without analysing the audio. A short intro and outro let the music play
   on its own, with fades at both ends.

episode.wav is never changed: the quality checker and its numbers use the dry
voices. The mix goes to a new file, episode_with_music.wav.

Usage:
    python background_music.py output/haveli/ep01/hi-IN
    python background_music.py output/haveli/ep01/hi-IN --seed 7 --new-prompt
    python background_music.py output/haveli/ep01/hi-IN --prompt "slow sitar drone, 60 BPM"
"""

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import wave
from array import array
from pathlib import Path

from pydantic import BaseModel, ValidationError

import sarvam_client
from script_writer import Script
from tts_engine import SAMPLE_RATE

MAX_MODEL_SECONDS = 120  # Stable Audio 3 Small-Music's limit
INTRO_SECONDS = 4.0  # music alone before the first line
OUTRO_SECONDS = 4.0  # music alone after the last line
MIN_OUTRO_SECONDS = 2.0  # shortest outro before we loop the music instead
FADE_IN_SECONDS = 1.5
FADE_OUT_SECONDS = 3.0
LOOP_CROSSFADE_SECONDS = 2.0  # used only when an episode is longer than one track

# Music level relative to the speech level (RMS over the spoken lines).
GAP_DB = -12.0  # intro, outro and long pauses
UNDER_SPEECH_DB = -24.0  # while someone is talking
ATTACK_SECONDS = 0.3  # the dip is complete by the time a line starts
RELEASE_SECONDS = 0.8  # slow return, so short pauses between lines do not pump
BLOCK_SECONDS = 0.01  # the gain envelope's resolution

MIN_PROMPT_WORDS, MAX_PROMPT_WORDS = 8, 60
BANNED_WORDS = ("vocal", "vocals", "voice", "voices", "singing", "singer", "sung",
                "lyrics", "lyric", "choir", "chant", "chanting", "song", "rap")
MAX_STORY_CHARS = 6000
DEFAULT_SEED = 42

SYSTEM_PROMPT = """You are the sound designer for an Indian audio drama app.
Read the episode and write ONE prompt for a text-to-music model. The music
plays quietly under the voices for the whole episode.

Rules for the prompt:
- English only, 15 to 45 words, comma-separated descriptions.
- Background score only. Never mention voices, vocals, singing, lyrics,
  choirs or songs: the music model must not produce any voice.
- Name the style, 2 to 4 instruments, a tempo in BPM and the mood.
  Drama underscore is usually slow, 55 to 90 BPM.
- Keep it steady and understated: no drops, no heavy drums, no sudden
  changes, nothing that fights the dialogue.
- Choose the instruments for the MOOD and genre first, not the setting.
  Mystery, suspense or horror: dark ambient pads, low sustained strings,
  sparse piano, deep drones, soft low pulses. Sad: solo piano, cello, warm
  pads. Warm family or romance: soft strings, light piano, gentle guitar.
- Indian instruments only if they suit that mood, and then as quiet
  texture (a tanpura or low drone), never a melodic lead like a sitar
  solo: a lead melody fights the dialogue and sounds like a travel film.
- No character names, places or plot events. The music model only
  understands sounds.

Also give "mood": a few words naming the mood you chose."""


class MusicPrompt(BaseModel):
    mood: str
    prompt: str


# --------------------------------------------------------------------------
# 1. Prompt
# --------------------------------------------------------------------------


def story_text(script: Script) -> str:
    """The episode as plain text for the LLM, cut to a safe length."""
    text = f"Title: {script.title}\n\n" + "\n".join(
        f"{line.speaker}: {line.text}" for line in script.lines
    )
    return text[:MAX_STORY_CHARS]


def check_prompt(prompt: str, names: list[str]) -> list[str]:
    """Return what is wrong with a music prompt (empty list if it is fine)."""
    problems = []
    letters = [c for c in prompt if c.isalpha()]
    if letters and sum(c.isascii() for c in letters) / len(letters) < 0.95:
        problems.append("Write the prompt in English only.")
    words = re.findall(r"[A-Za-z]+", prompt.lower())
    if not MIN_PROMPT_WORDS <= len(words) <= MAX_PROMPT_WORDS:
        problems.append(
            f"Use {MIN_PROMPT_WORDS} to {MAX_PROMPT_WORDS} words (you used {len(words)})."
        )
    banned = sorted(set(words) & set(BANNED_WORDS))
    if banned:
        problems.append(
            f"Remove {', '.join(banned)}: do not mention voices or singing at all."
        )
    for name in names:
        if name.isascii() and re.search(rf"\b{re.escape(name.lower())}\b", prompt.lower()):
            problems.append(f"Remove the character name {name}.")
    return problems


def write_music_prompt(script: Script, max_attempts: int = 3) -> MusicPrompt:
    """Ask the chat model for a music prompt, re-asking with feedback on errors."""
    names = [character.name for character in script.characters]
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "music_prompt", "schema": MusicPrompt.model_json_schema()},
    }
    first_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": story_text(script)},
    ]
    messages = list(first_messages)

    for attempt in range(1, max_attempts + 1):
        try:
            raw = sarvam_client.chat(
                messages,
                response_format=response_format,
                temperature=0.4,
                reasoning_effort="low",
                max_tokens=1000,
            )
        except sarvam_client.OutOfTokens:
            print(f"Attempt {attempt}/{max_attempts} ran out of tokens.", file=sys.stderr)
            messages = list(first_messages)
            continue
        try:
            result = MusicPrompt.model_validate_json(raw)
            problems = check_prompt(result.prompt, names)
        except ValidationError as error:
            problems = [f"The JSON does not match the schema: {error}"]
        if not problems:
            return result
        print(f"Attempt {attempt}/{max_attempts} was invalid: {problems}", file=sys.stderr)
        messages = first_messages + [
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": "That prompt has problems:\n- " + "\n- ".join(problems)
                + "\n\nReturn the complete corrected JSON.",
            },
        ]

    raise RuntimeError(f"No valid music prompt after {max_attempts} attempts.")


# --------------------------------------------------------------------------
# 2. Music
# --------------------------------------------------------------------------


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def tflite_dir() -> Path:
    """Stable Audio 3's optimised CPU build (int8, about 1.5 GB of RAM)."""
    configured = os.getenv("STABLE_AUDIO_TFLITE_DIR")
    return Path(configured) if configured else Path.home() / "stable-audio-3" / "optimized" / "tflite"


def backend() -> str:
    """'tflite' if the optimised CPU build is installed, otherwise 'pytorch'.

    The PyTorch build needs about 4.5 GB of RAM for a 2-minute track, which
    makes a 6 GB laptop swap to disk for most of the run. The TFLite build
    runs the same model with int8 weights in about 1.5 GB.
    """
    return "tflite" if venv_python(tflite_dir() / ".venv").exists() else "pytorch"


def worker_python() -> Path:
    """The Python of the Stable Audio 3 (PyTorch) environment."""
    configured = os.getenv("STABLE_AUDIO_PYTHON")
    path = Path(configured) if configured else venv_python(Path.home() / "stable-audio-3" / ".venv")
    if not path.exists():
        raise RuntimeError(
            f"Stable Audio 3 is not installed at {path}. Set it up with:\n"
            "    git clone https://github.com/Stability-AI/stable-audio-3 ~/stable-audio-3\n"
            "    cd ~/stable-audio-3/optimized/tflite; ./install.bat --download sm-music\n"
            "(low-memory CPU build), or for the PyTorch build:\n"
            "    cd ~/stable-audio-3; python -m venv .venv; .venv/Scripts/python -m pip install -e .\n"
            "or set STABLE_AUDIO_PYTHON in .env to that environment's python."
        )
    return path


def music_file(folder: Path, prompt: str, seconds: float, seed: int, engine: str = "pytorch") -> Path:
    # int8 TFLite gives a different (not worse) take than PyTorch for the same seed
    model = "small-music" if engine == "pytorch" else f"small-music-{engine}"
    key = f"{model}|{prompt}|{seconds:.1f}|{seed}"
    return folder / f"music_{hashlib.sha256(key.encode()).hexdigest()[:10]}.wav"


def to_voice_format(source: Path, target: Path) -> None:
    """44.1 kHz stereo -> 22050 Hz mono 16-bit (the voices' format), peak-normalised."""
    import numpy as np  # installed with streamlit

    with wave.open(str(source), "rb") as clip:
        rate, channels = clip.getframerate(), clip.getnchannels()
        data = np.frombuffer(clip.readframes(clip.getnframes()), dtype="<i2")
    mono = data.reshape(-1, channels).astype(np.float64).mean(axis=1)
    if rate == 2 * SAMPLE_RATE:
        # low-pass at 10 kHz (below the new 11 kHz limit), then keep every 2nd sample
        n = np.arange(-32, 33)
        cutoff = 10_000 / rate
        taps = 2 * cutoff * np.sinc(2 * cutoff * n) * np.hamming(len(n))
        mono = np.convolve(mono, taps / taps.sum(), mode="same")[::2]
    elif rate != SAMPLE_RATE:
        raise ValueError(f"{source} is {rate} Hz; expected 44100 or {SAMPLE_RATE}.")
    peak = np.abs(mono).max()
    if peak > 0:
        mono = mono * (0.9 * 32767 / peak)
    write_wav(target, array("h", np.round(mono).astype("<i2").tobytes()))


def generate_music(folder: Path, prompt: str, seconds: float, seed: int, log=print) -> Path:
    """Make the music track once; later runs reuse the file."""
    engine = backend()
    path = music_file(folder, prompt, seconds, seed, engine)
    if path.exists():
        log(f"Reusing {path.name}")
        return path
    log(f"Generating {seconds:.0f}s of music with Stable Audio 3 ({engine}, seed {seed})...")
    if engine == "tflite":
        raw = path.with_suffix(".44k.wav")
        command = [
            str(venv_python(tflite_dir() / ".venv")), str(tflite_dir() / "scripts" / "sa3_tflite.py"),
            # name the decoder, or the script stops to ask which one to use
            "--prompt", prompt, "--dit", "sm-music", "--decoder", "same-s",
            "--seconds", f"{seconds:.1f}",
            "--seed", str(seed), "--threads", str(os.cpu_count() or 2), "--out", str(raw.resolve()),
        ]
        # no stdin: if the script ever asks a question, fail instead of hanging
        subprocess.run(command, check=True, cwd=tflite_dir(), stdin=subprocess.DEVNULL)
        to_voice_format(raw, path)
        raw.unlink()
    else:
        command = [
            str(worker_python()), str(Path(__file__).with_name("music_worker.py")),
            "--prompt", prompt, "--seconds", f"{seconds:.1f}", "--seed", str(seed),
            "--rate", str(SAMPLE_RATE), "--out", str(path),
        ]
        subprocess.run(command, check=True)
    return path


# --------------------------------------------------------------------------
# 3. Mix
# --------------------------------------------------------------------------


def read_wav(path: Path) -> array:
    with wave.open(str(path), "rb") as clip:
        if (clip.getframerate(), clip.getnchannels(), clip.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise ValueError(f"{path} is not {SAMPLE_RATE} Hz mono 16-bit")
        samples = array("h")
        samples.frombytes(clip.readframes(clip.getnframes()))
    if sys.byteorder == "big":
        samples.byteswap()
    return samples


def write_wav(path: Path, samples: array) -> None:
    if sys.byteorder == "big":
        samples = array("h", samples)
        samples.byteswap()
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(samples.tobytes())


def rms(samples, ranges=None) -> float:
    """Root mean square level, over the given (start, end) sample ranges or all."""
    ranges = ranges or [(0, len(samples))]
    total, count = 0.0, 0
    for start, end in ranges:
        for value in samples[start:end]:
            total += value * value
        count += max(0, min(end, len(samples)) - start)
    return math.sqrt(total / count) if count else 0.0


def fit_length(music: array, length: int, crossfade: int) -> list[float]:
    """Cut the music to length, looping with a crossfade if it is too short."""
    out = [float(v) for v in music[:length]]
    if len(music) <= crossfade:
        raise ValueError("The music track is shorter than the loop crossfade.")
    while len(out) < length:
        # fade the end of what we have into the start of the track
        for i in range(crossfade):
            mix = i / crossfade
            out[-crossfade + i] = out[-crossfade + i] * (1 - mix) + music[i] * mix
        out.extend(float(v) for v in music[crossfade:])
    return out[:length]


def speech_ranges(manifest: dict, offset: float) -> list[tuple[float, float]]:
    """(start, end) seconds of every line, shifted by the intro."""
    return [
        (offset + line["start"], offset + line["start"] + line["duration"])
        for line in manifest["lines"]
    ]


def gain_envelope(ranges: list[tuple[float, float]], total_seconds: float) -> list[float]:
    """Music gain in dB (relative to the speech level), one value per block.

    The music dips to UNDER_SPEECH_DB during speech and sits at GAP_DB
    otherwise. The dip starts ATTACK_SECONDS early so it is complete when the
    line starts, and recovers over RELEASE_SECONDS.
    """
    blocks = math.ceil(total_seconds / BLOCK_SECONDS)
    target = [GAP_DB] * blocks
    for start, end in ranges:
        first = max(0, int((start - ATTACK_SECONDS) / BLOCK_SECONDS))
        last = min(blocks, math.ceil(end / BLOCK_SECONDS))
        for b in range(first, last):
            target[b] = UNDER_SPEECH_DB

    swing = GAP_DB - UNDER_SPEECH_DB
    down = swing * BLOCK_SECONDS / ATTACK_SECONDS
    up = swing * BLOCK_SECONDS / RELEASE_SECONDS
    envelope, level = [], GAP_DB
    for want in target:
        if want < level:
            level = max(want, level - down)
        elif want > level:
            level = min(want, level + up)
        envelope.append(level)
    return envelope


def mix_episode(folder: Path, music_path: Path) -> dict:
    """Write episode_with_music.wav and return the levels used."""
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    speech = read_wav(folder / manifest["episode"])
    music = read_wav(music_path)

    intro = int(INTRO_SECONDS * SAMPLE_RATE)
    outro = int(OUTRO_SECONDS * SAMPLE_RATE)
    spare = len(music) - intro - len(speech)
    if int(MIN_OUTRO_SECONDS * SAMPLE_RATE) <= spare < outro:
        # only just over one track: shorten the outro rather than loop a sliver
        outro = spare
    total = intro + len(speech) + outro
    bed = fit_length(music, total, int(LOOP_CROSSFADE_SECONDS * SAMPLE_RATE))

    line_samples = [
        (int(line["start"] * SAMPLE_RATE), int((line["start"] + line["duration"]) * SAMPLE_RATE))
        for line in manifest["lines"]
    ]
    speech_level = rms(speech, line_samples)
    music_level = rms(bed)
    if music_level == 0:
        raise ValueError(f"{music_path} is silent.")
    match = speech_level / music_level  # makes the music as loud as the speech

    envelope = gain_envelope(speech_ranges(manifest, INTRO_SECONDS), total / SAMPLE_RATE)
    gains = [match * 10 ** (db / 20) for db in envelope]
    block = BLOCK_SECONDS * SAMPLE_RATE  # 220.5 samples, so keep it a float
    last_block = len(gains) - 1
    fade_in = int(FADE_IN_SECONDS * SAMPLE_RATE)
    fade_out = int(FADE_OUT_SECONDS * SAMPLE_RATE)

    out = array("h", bytes(2 * total))
    for i in range(total):
        value = bed[i] * gains[min(int(i / block), last_block)]
        if i < fade_in:
            value *= i / fade_in
        elif i >= total - fade_out:
            value *= (total - i) / fade_out
        if intro <= i < intro + len(speech):
            value += speech[i - intro]
        out[i] = max(-32768, min(32767, round(value)))

    write_wav(folder / "episode_with_music.wav", out)
    return {
        "speech_offset_seconds": INTRO_SECONDS,
        "seconds": round(total / SAMPLE_RATE, 2),
        "music_db_in_gaps": GAP_DB,
        "music_db_under_speech": UNDER_SPEECH_DB,
        "looped": len(music) < total,
    }


# --------------------------------------------------------------------------
# 4. Whole step
# --------------------------------------------------------------------------


def add_music(
    folder: Path,
    seed: int = DEFAULT_SEED,
    prompt: str | None = None,
    new_prompt: bool = False,
    log=print,
) -> dict:
    """Prompt, generate and mix background music for one episode folder."""
    folder = Path(folder)
    script = Script.model_validate_json((folder / "script.json").read_text(encoding="utf-8"))
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    info_path = folder / "music.json"
    old = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}

    if prompt:
        chosen = MusicPrompt(mood="given by hand", prompt=prompt)
    elif old.get("prompt") and not new_prompt:
        chosen = MusicPrompt(mood=old["mood"], prompt=old["prompt"])
        log(f"Reusing the music prompt: {chosen.prompt}")
    else:
        chosen = write_music_prompt(script)
        log(f"Music prompt ({chosen.mood}): {chosen.prompt}")

    needed = INTRO_SECONDS + manifest["duration"] + OUTRO_SECONDS
    seconds = min(MAX_MODEL_SECONDS, math.ceil(needed))
    music_path = generate_music(folder, chosen.prompt, seconds, seed, log)
    levels = mix_episode(folder, music_path)

    info = {
        "model": "stabilityai/stable-audio-3-small-music",
        "engine": backend(),
        "mood": chosen.mood,
        "prompt": chosen.prompt,
        "seed": seed,
        "music_file": music_path.name,
        "output": "episode_with_music.wav",
        **levels,
    }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {folder / 'episode_with_music.wav'}")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Add background music to episodes.")
    parser.add_argument("folders", nargs="+", type=Path, help="episode folders, e.g. output/haveli/ep01/hi-IN")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="change it for a different take")
    parser.add_argument("--prompt", help="use this music prompt instead of asking the LLM")
    parser.add_argument("--new-prompt", action="store_true", help="ask the LLM again even if a prompt is saved")
    args = parser.parse_args()
    for folder in args.folders:
        add_music(folder, seed=args.seed, prompt=args.prompt, new_prompt=args.new_prompt)


if __name__ == "__main__":
    main()
