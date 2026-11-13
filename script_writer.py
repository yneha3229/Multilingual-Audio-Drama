"""Phase 1: the script writer.

Turns story prose into a structured audio-drama script in three steps:

    1. segment_story()  - plain code splits the story into numbered segments:
                          text inside quotation marks is dialogue, the rest is
                          narration. No LLM, so no words can be lost or changed.
    2. the LLM          - only labels each segment: who speaks it, with what
                          emotion, and whether it is a bare speech tag
                          ("he said") that should be dropped.
    3. build_script()   - checks the labels and combines them with the segments
                          into the final Script.

Why not let the LLM write the whole script? We tried. It dropped sentences and
gave narration to characters about half the time. Giving the LLM the smallest
possible job (labelling) makes the output reliable.

Usage:
    python script_writer.py stories/sample_story.txt
    python script_writer.py stories/sample_story.txt -o output/my_script.json
"""

import argparse
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

import sarvam_client

# The only emotions the LLM may use. Phase 2 maps each of these to a speaking
# pace, so a fixed list keeps the two stages in agreement.
Emotion = Literal[
    "neutral", "calm", "happy", "excited", "angry", "sad", "fearful", "whispering"
]

# Every language that Sarvam can both translate into (Mayura) and speak (Bulbul).
LANGUAGES = {
    "en-IN": "English",
    "hi-IN": "Hindi",
    "bn-IN": "Bengali",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
    "mr-IN": "Marathi",
    "gu-IN": "Gujarati",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "pa-IN": "Punjabi",
    "od-IN": "Odia",
}
LanguageCode = Literal[
    "en-IN", "hi-IN", "bn-IN", "ta-IN", "te-IN", "mr-IN", "gu-IN", "kn-IN", "ml-IN",
    "pa-IN", "od-IN",
]

NARRATOR = "Narrator"
MAX_LINE_CHARS = 500  # Bulbul v3 allows 2500 per request; short lines sound better
MAX_SPEECH_TAG_WORDS = 8  # narration longer than this is never dropped


# --------------------------------------------------------------------------
# 1. The final script (this is what Phase 2 will read)
# --------------------------------------------------------------------------


class Character(BaseModel):
    name: str = Field(min_length=1)
    gender: Literal["male", "female"]  # used in Phase 2 to pick a voice
    description: str  # e.g. "elderly, soft-spoken" - also helps voice choice


class Line(BaseModel):
    speaker: str = Field(min_length=1)
    text: str = Field(min_length=1)
    emotion: Emotion


class Script(BaseModel):
    title: str = Field(min_length=1)
    language: LanguageCode  # stories come in en-IN or hi-IN; translations add the rest
    characters: list[Character]
    lines: list[Line] = Field(min_length=1)

    @model_validator(mode="after")
    def check_speakers(self):
        names = [c.name for c in self.characters]
        if len(names) != len(set(names)):
            raise ValueError("characters contains duplicate names")
        if NARRATOR in names:
            raise ValueError(f'"{NARRATOR}" must not be listed in characters')
        known = set(names) | {NARRATOR}
        for number, line in enumerate(self.lines, start=1):
            if line.speaker not in known:
                raise ValueError(f'line {number}: speaker "{line.speaker}" is not in characters')
        return self


# --------------------------------------------------------------------------
# 2. Splitting the story into segments (no LLM)
# --------------------------------------------------------------------------


class Segment(BaseModel):
    id: int
    kind: Literal["dialogue", "narration"]
    text: str


# Matches "straight quoted" or “curly quoted” speech. The brackets keep the
# quoted parts in the result of re.split instead of throwing them away.
QUOTED = re.compile(r'("[^"]*"|“[^”]*”)')


def split_title(story: str) -> tuple[str | None, str]:
    """If the first line looks like a title, return (title, rest of story)."""
    first, _, rest = story.strip().partition("\n")
    first = first.strip()
    looks_like_title = (
        rest.strip() and len(first.split()) <= 10 and first[-1:] not in '.!?"”।'
    )
    return (first, rest) if looks_like_title else (None, story)


def segment_story(story: str) -> list[Segment]:
    """Split a story into dialogue and narration segments, in order."""
    segments: list[Segment] = []
    for piece in QUOTED.split(story):
        is_dialogue = bool(QUOTED.fullmatch(piece))
        text = " ".join(piece.strip('"“”').split())  # drop quotes, tidy spaces
        text = text.strip(" ,;:")
        if not re.search(r"\w", text):
            continue  # nothing speakable, e.g. the gap between two quotes

        # 'Listen," he said' leaves pieces with no ending punctuation.
        if text[-1] not in ".!?…।":
            text += "."
        if not is_dialogue:
            text = text[0].upper() + text[1:]  # "he said." -> "He said."

        kind = "dialogue" if is_dialogue else "narration"
        segments.append(Segment(id=len(segments) + 1, kind=kind, text=text))
    return segments


# --------------------------------------------------------------------------
# 3. The LLM's job: label each segment
# --------------------------------------------------------------------------


class Label(BaseModel):
    id: int
    speaker: str
    emotion: Emotion
    drop: bool


class Labels(BaseModel):
    title: str = Field(min_length=1)
    language: Literal["en-IN", "hi-IN"]
    characters: list[Character]
    labels: list[Label]


SYSTEM_PROMPT = f"""You are the casting director for an audio drama. A story has already been split into numbered segments. Each segment is either dialogue (words a character says aloud) or narration. You label the segments. You never rewrite them.

Return one label per segment, in the same order, with the same id:

- "speaker": for a dialogue segment, the name of the character who says it. Work it out from the surrounding narration and from who is being spoken to. Use one consistent name per character. For a narration segment, always "{NARRATOR}".
- "emotion": how a voice actor should perform it. Choose by the speaker's feeling, not by punctuation. An exclamation mark alone does not mean excited.
    neutral: plain delivery, no strong feeling
    calm: gentle, steady, reassuring
    happy: pleased, warm, content
    excited: POSITIVE energy only, such as good news, joy, or eagerness. Never for shouting in a quarrel or in panic.
    angry: shouting at someone, giving harsh orders, confrontation, frustration
    sad: grief, hurt, a breaking voice
    fearful: frightened, panicked, frantic with worry, nervous
    whispering: a lowered, secretive voice
  Use the cues in nearby narration and speech tags: "shouted" means angry (or fearful if the character is panicking), "whispered" means whispering, "voice breaking" means sad.
  Narration that describes HOW a character is about to speak decides the emotion of the dialogue right after it. For example, after "He lowered his voice to a whisper." the next dialogue segment is whispering, even if its words sound calm.
  When one character's speech is interrupted only by a speech tag (for example: "Stop!" he shouted. "Leave her alone!"), both parts are the same outburst and get the same emotion.
  Narration is usually neutral or calm.
- "drop": true ONLY for a narration segment that is nothing more than a speech tag, such as "He said quietly." or "Kabir shouted." These are not read aloud in an audio drama. Everything else is false. Dialogue is always false.

Also return:
- "characters": every character who speaks at least one dialogue segment, with gender and a short description of age and personality. Do not include the {NARRATOR}.
- "language": "en-IN" if the story is in English, "hi-IN" if it is in Hindi.
- "title": the story's title if one is given, otherwise a short one you write.

The story may be one episode of a series. If a list of existing series characters is given, those characters keep exactly those names: use the name from the list as the speaker, never a variation (no added titles, nicknames, or a different script). Only invent a new name for a character who is not on the list."""


def build_user_message(
    title: str | None, segments: list[Segment], known: list[Character] | None = None
) -> str:
    listing = "\n".join(f"{s.id}. [{s.kind}] {s.text}" for s in segments)
    heading = f"Title: {title}\n\n" if title else ""
    if known:
        cast = "\n".join(f'- "{c.name}" ({c.gender}): {c.description}' for c in known)
        heading += f"Existing series characters (use these exact names):\n{cast}\n\n"
    return f"{heading}Segments:\n{listing}"


def looks_like(new_name: str, known_name: str) -> bool:
    """True if a new character name is probably a variant of a known one:
    one name is the other plus a title ("राठौर" vs "इंस्पेक्टर राठौर",
    "Mehta" vs "Mehta Sahab"), or a near-identical spelling ("Vikram" vs
    "Vickram").

    The spelling rule is strict on purpose: Hindi kinship words are short
    and close, and "दादी" (grandmother) and "दादाजी" (grandfather) are
    different people.
    """
    a, b = new_name.strip().lower(), known_name.strip().lower()
    words_a, words_b = set(a.split()), set(b.split())
    if words_a <= words_b or words_b <= words_a:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.9


def parse_labels(raw: str) -> Labels:
    """Turn the LLM's raw reply into validated Labels, or raise ValueError."""
    text = raw.strip()
    # Some models wrap JSON in ```json ... ``` even when told not to.
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        return Labels.model_validate_json(text)
    except ValidationError as error:
        raise ValueError(str(error)) from error


# --------------------------------------------------------------------------
# 4. Combining segments + labels into the final script
# --------------------------------------------------------------------------


def build_script(
    segments: list[Segment], labels: Labels, known: list[Character] | None = None
) -> tuple[Script, list[str]]:
    """Check the labels and build the Script.

    known is the cast from earlier episodes of a series. Their gender and
    description always win over what the LLM wrote this time, so a character
    cannot change voice between episodes.

    Returns the script and the list of speech tags that were dropped.
    Raises ValueError (with a message the LLM can act on) if a label is wrong.
    """
    known_by_name = {c.name: c for c in known or []}
    problems = []
    for character in labels.characters:
        if character.name in known_by_name:
            continue
        for existing in known_by_name:
            if looks_like(character.name, existing):
                problems.append(
                    f'"{character.name}" looks like the existing series character '
                    f'"{existing}". If it is the same person, use exactly "{existing}".'
                )
    if problems:
        raise ValueError("\n".join(problems))
    expected_ids = [s.id for s in segments]
    got_ids = [label.id for label in labels.labels]
    if got_ids != expected_ids:
        raise ValueError(
            f"labels must have exactly one entry for each segment id 1 to {len(segments)}, "
            f"in order. Got ids: {got_ids}"
        )

    names = {c.name for c in labels.characters} | set(known_by_name)
    lines, dropped = [], []
    for segment, label in zip(segments, labels.labels):
        if segment.kind == "dialogue":
            if label.speaker not in names:
                problems.append(
                    f'segment {segment.id} is dialogue spoken by "{label.speaker}", but '
                    f'"{label.speaker}" is not in "characters". Add "{label.speaker}" to '
                    f'"characters" with gender and description. Words that a character wrote '
                    f'(a letter, a note) count as spoken by that character, even if they '
                    f'are absent or dead.'
                )
            lines.append(Line(speaker=label.speaker, text=segment.text, emotion=label.emotion))
        elif label.drop and len(segment.text.split()) <= MAX_SPEECH_TAG_WORDS:
            dropped.append(segment.text)
        else:
            # Narration is always the Narrator's, whatever the LLM said, and
            # anything too long to be a speech tag is kept even if drop=true.
            lines.append(Line(speaker=NARRATOR, text=segment.text, emotion=label.emotion))
    if problems:
        raise ValueError("\n".join(problems))

    # Characters in order of first line, with known characters' details kept.
    described = {c.name: c for c in labels.characters} | known_by_name
    speakers = list(dict.fromkeys(line.speaker for line in lines if line.speaker != NARRATOR))
    script = Script(
        title=labels.title,
        language=labels.language,
        characters=[described[name] for name in speakers],
        lines=lines,
    )
    return merge_lines(script), dropped


def merge_lines(script: Script) -> Script:
    """Join neighbouring lines that have the same speaker and emotion.

    One audio clip per sentence sounds choppy and costs more API calls.
    """
    merged: list[Line] = []
    for line in script.lines:
        previous = merged[-1] if merged else None
        if (
            previous
            and previous.speaker == line.speaker
            and previous.emotion == line.emotion
            and len(previous.text) + 1 + len(line.text) <= MAX_LINE_CHARS
        ):
            merged[-1] = previous.model_copy(update={"text": f"{previous.text} {line.text}"})
        else:
            merged.append(line)
    return script.model_copy(update={"lines": merged})


# --------------------------------------------------------------------------
# 5. The main function
# --------------------------------------------------------------------------


def write_script(
    story: str, max_attempts: int = 3, known: list[Character] | None = None
) -> tuple[Script, list[str]]:
    """Convert a story into a validated Script, re-asking the LLM on errors.

    known is the cast from earlier episodes of a series, if any.
    Returns the script and the list of speech tags that were dropped.
    """
    title, body = split_title(story)
    segments = segment_story(body)
    if not segments:
        raise ValueError("The story is empty.")

    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "segment_labels", "schema": Labels.model_json_schema()},
    }
    first_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(title, segments, known)},
    ]
    messages = list(first_messages)

    for attempt in range(1, max_attempts + 1):
        try:
            raw = sarvam_client.chat(
                messages,
                response_format=response_format,
                temperature=0.2,  # low: labelling should be consistent, not creative
                reasoning_effort="low",
                max_tokens=4000,
            )
        except sarvam_client.OutOfTokens:
            # Occasionally the model rambles until it hits the limit. Start fresh.
            print(f"Attempt {attempt}/{max_attempts} ran out of tokens.", file=sys.stderr)
            messages = list(first_messages)
            continue
        try:
            script, dropped = build_script(segments, parse_labels(raw), known)
            if title:  # keep the story's own title rather than the model's rewording
                script = script.model_copy(update={"title": title})
            return script, dropped
        except ValueError as error:
            print(f"Attempt {attempt}/{max_attempts} was invalid: {error}", file=sys.stderr)
            # Show the model its own answer and the exact problem, then retry.
            messages = first_messages + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": f"Those labels are invalid:\n{error}\n\n"
                    "Return the complete corrected JSON.",
                },
            ]

    raise RuntimeError(f"No valid script after {max_attempts} attempts.")


# --------------------------------------------------------------------------
# 6. Command line
# --------------------------------------------------------------------------


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # so Hindi prints on Windows terminals

    parser = argparse.ArgumentParser(description="Convert a story into an audio-drama script.")
    parser.add_argument("story_file", type=Path, help="text file containing the story")
    parser.add_argument("-o", "--output", type=Path, help="where to save the script JSON")
    args = parser.parse_args()

    story = args.story_file.read_text(encoding="utf-8")
    try:
        script, dropped = write_script(story)
    except (sarvam_client.SarvamError, RuntimeError, ValueError) as error:
        sys.exit(f"Error: {error}")

    output = args.output or Path("output") / f"{args.story_file.stem}_script.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(script.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n{script.title}  ({script.language})")
    print("Characters:")
    for character in script.characters:
        print(f"  {character.name} ({character.gender}): {character.description}")
    print(f"\n{len(script.lines)} lines:")
    for line in script.lines:
        print(f"  [{line.emotion:>10}] {line.speaker}: {line.text}")
    if dropped:
        print(f"\nDropped {len(dropped)} speech tags: {' | '.join(dropped)}")
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
