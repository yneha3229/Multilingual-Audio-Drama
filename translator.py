"""Phase 3: the translator.

Translates a script into another Indian language. Only the title and each
line's text change. The speakers, emotions, and line order stay the same, so
the voice engine gives every character the same voice in every language.

Why an LLM and not Sarvam's translation API (Mayura)? We tried Mayura first.
It translates one line at a time with no idea who is speaking to whom, so it
got genders wrong ("Meera ... gaya"), mixed English into every sentence, and
turned the local train into a "local bus". The LLM sees the whole scene and
the character list, so it can get gender and respect levels right.

Usage:
    python translator.py output/sample_story/script.json hi-IN ta-IN
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

import sarvam_client
from script_writer import LANGUAGES, Line, Script

LINES_PER_REQUEST = 30  # long scripts are translated in chunks of this size
MIN_NATIVE_SCRIPT = 0.8  # at least 80% of letters must be in the target script

# Where each language's letters live in Unicode. Used to check that a
# "Hindi" translation is really written in Devanagari, and so on.
SCRIPT_RANGES = {
    "en-IN": (0x0041, 0x024F),  # Latin
    "hi-IN": (0x0900, 0x097F),  # Devanagari
    "mr-IN": (0x0900, 0x097F),  # Devanagari
    "bn-IN": (0x0980, 0x09FF),  # Bengali
    "pa-IN": (0x0A00, 0x0A7F),  # Gurmukhi
    "gu-IN": (0x0A80, 0x0AFF),  # Gujarati
    "od-IN": (0x0B00, 0x0B7F),  # Odia
    "ta-IN": (0x0B80, 0x0BFF),  # Tamil
    "te-IN": (0x0C00, 0x0C7F),  # Telugu
    "kn-IN": (0x0C80, 0x0CFF),  # Kannada
    "ml-IN": (0x0D00, 0x0D7F),  # Malayalam
}


class TranslatedLine(BaseModel):
    id: int
    text: str = Field(min_length=1)


class Translation(BaseModel):
    title: str = Field(min_length=1)
    lines: list[TranslatedLine]


def build_system_prompt(source: str, target: str) -> str:
    src, tgt = LANGUAGES[source], LANGUAGES[target]
    return f"""You translate audio-drama scripts from {src} into {tgt} for voice actors.

Rules:
1. Translate every numbered line. Return exactly one translation per line id, in order. Do not merge, split, add, or drop lines.
2. Translate only each line's "text". The "speaker" field is there so you know who is talking, to get gender and respect right. Never write the speaker's name, "he said", or a description of the voice into the translation. If the original text does not say "Raghu said", neither does yours.
3. Keep the meaning exact: who did what to whom must stay the same. "Papa trusted nobody" must not become "nobody trusted Papa".
4. Write natural, spoken {tgt} that a listener would enjoy, in the native script of {tgt}. Not stiff or bookish, and not {src} words in every sentence.
5. Write all names of people and places in the native script of {tgt}.
6. Everyday borrowed words that {tgt} speakers really use (train, station, phone, office) are fine, written in the native script.
7. Use the correct grammatical gender for every character, using the character list. Check the verbs and pronouns that refer to each character.
8. Choose the right level of respect between characters, for example an elder speaking to a young woman, or siblings speaking to each other.
9. Keep punctuation that carries feeling, such as ! and ?.
10. Also translate the title."""


def build_user_message(script: Script, start: int, lines: list[Line]) -> str:
    cast = "\n".join(f"- {c.name} ({c.gender}): {c.description}" for c in script.characters)
    # One JSON object per line keeps "who is speaking" clearly apart from the
    # text. With a plain "spoken by Meera:" heading, the model sometimes
    # translated the heading into the dialogue.
    numbered = "\n".join(
        json.dumps(
            {"id": start + offset, "speaker": line.speaker, "text": line.text},
            ensure_ascii=False,
        )
        for offset, line in enumerate(lines)
    )
    return (
        f"Title: {script.title}\n\n"
        f"Characters:\n{cast}\n- Narrator: tells the story, is not a character in it\n\n"
        f"Lines:\n{numbered}"
    )


def native_script_share(text: str, language: str) -> float:
    """What fraction of the letters in text belong to the language's script."""
    low, high = SCRIPT_RANGES[language]
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 1.0  # only digits or punctuation, nothing to judge
    return sum(low <= ord(ch) <= high for ch in letters) / len(letters)


def check_translation(raw: str, expected_ids: list[int], target: str) -> Translation:
    """Validate the LLM's reply, or raise ValueError with a message it can act on."""
    try:
        translation = Translation.model_validate_json(raw.strip())
    except ValidationError as error:
        raise ValueError(str(error)) from error

    got_ids = [line.id for line in translation.lines]
    if got_ids != expected_ids:
        raise ValueError(
            f"lines must have exactly one entry for each id {expected_ids[0]} to "
            f"{expected_ids[-1]}, in order. Got ids: {got_ids}"
        )

    problems = []
    for line in translation.lines:
        # Remove a leading "Speaker: " label if the model added one anyway.
        line.text = re.sub(r"^[^\s:]{1,20}(\s[^\s:]{1,20})?:\s+", "", line.text).strip()
        share = native_script_share(line.text, target)
        if share < MIN_NATIVE_SCRIPT:
            problems.append(
                f"line {line.id} is only {share:.0%} in {LANGUAGES[target]} script. "
                f"Write it fully in {LANGUAGES[target]} script, including names."
            )
    if problems:
        raise ValueError("\n".join(problems))
    return translation


def translate_chunk(
    script: Script, start: int, lines: list[Line], target: str, max_attempts: int = 3
) -> Translation:
    """Translate some of the script's lines, re-asking the LLM on errors."""
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "translation", "schema": Translation.model_json_schema()},
    }
    first_messages = [
        {"role": "system", "content": build_system_prompt(script.language, target)},
        {"role": "user", "content": build_user_message(script, start, lines)},
    ]
    expected_ids = list(range(start, start + len(lines)))
    messages = list(first_messages)

    for attempt in range(1, max_attempts + 1):
        try:
            raw = sarvam_client.chat(
                messages,
                response_format=response_format,
                temperature=0.2,
                reasoning_effort="low",
                max_tokens=8000,
            )
        except sarvam_client.OutOfTokens:
            print(f"Attempt {attempt}/{max_attempts} ran out of tokens.", file=sys.stderr)
            messages = list(first_messages)
            continue
        try:
            return check_translation(raw, expected_ids, target)
        except ValueError as error:
            print(f"Attempt {attempt}/{max_attempts} was invalid: {error}", file=sys.stderr)
            messages = first_messages + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": f"That translation is invalid:\n{error}\n\n"
                    "Return the complete corrected JSON.",
                },
            ]
    raise RuntimeError(f"No valid {LANGUAGES[target]} translation after {max_attempts} attempts.")


def translate_script(script: Script, target: str, cache_file: Path | None = None) -> Script:
    """Return a copy of the script translated into the target language.

    cache_file remembers the finished translation. Re-running then costs
    nothing and, more importantly, returns the exact same text, so the voice
    engine can reuse the audio clips it already made.
    """
    if target not in LANGUAGES:
        raise ValueError(f"Unsupported language {target}. Choose from: {', '.join(LANGUAGES)}")
    if target == script.language:
        return script

    # The fingerprint changes if the source script changes in any way.
    fingerprint = hashlib.sha1(script.model_dump_json().encode()).hexdigest()
    if cache_file and cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if cached.get("source_fingerprint") == fingerprint:
            return Script.model_validate(cached["translated"])

    title = None
    texts: list[str] = []
    for offset in range(0, len(script.lines), LINES_PER_REQUEST):
        chunk = script.lines[offset : offset + LINES_PER_REQUEST]
        translation = translate_chunk(script, offset + 1, chunk, target)
        title = title or translation.title
        texts += [line.text for line in translation.lines]

    translated = Script(
        title=title,
        language=target,
        characters=script.characters,  # names stay as-is: they identify the voices
        lines=[
            Line(speaker=line.speaker, text=text, emotion=line.emotion)
            for line, text in zip(script.lines, texts)
        ],
    )
    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps(
                {"source_fingerprint": fingerprint, "translated": translated.model_dump()},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return translated


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Translate a script JSON into other languages.")
    parser.add_argument("script_file", type=Path, help="script JSON made by script_writer.py")
    parser.add_argument("languages", nargs="+", help=f"target codes: {', '.join(LANGUAGES)}")
    args = parser.parse_args()

    script = Script.model_validate_json(args.script_file.read_text(encoding="utf-8"))
    for target in args.languages:
        folder = args.script_file.parent / target
        try:
            translated = translate_script(script, target, folder / "translation_cache.json")
        except (sarvam_client.SarvamError, RuntimeError, ValueError) as error:
            sys.exit(f"Error: {error}")

        folder.mkdir(parents=True, exist_ok=True)
        output = folder / "script.json"
        output.write_text(
            json.dumps(translated.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n{LANGUAGES[target]}: {translated.title}")
        for line in translated.lines:
            print(f"  [{line.emotion:>10}] {line.speaker}: {line.text}")
        print(f"Saved to {output}")


if __name__ == "__main__":
    main()
