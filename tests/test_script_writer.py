"""Offline tests for the script writer. They need no API key.

Run with:  python -m unittest discover tests
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from script_writer import (  # noqa: E402
    Character,
    build_script,
    looks_like,
    parse_labels,
    segment_story,
    split_title,
)

STORY = (
    "Rain hammered the roof. Meera froze.\n\n"
    '"Don\'t be afraid," he said quietly. "My name is Raghunath."\n\n'
    'Meera pulled the envelope closer. "Who sent you?"'
)
# segment_story(STORY) gives:
#   1 narration  Rain hammered the roof. Meera froze.
#   2 dialogue   Don't be afraid.
#   3 narration  He said quietly.
#   4 dialogue   My name is Raghunath.
#   5 narration  Meera pulled the envelope closer.
#   6 dialogue   Who sent you?

GOOD_LABELS = {
    "title": "The Last Local",
    "language": "en-IN",
    "characters": [
        {"name": "Meera", "gender": "female", "description": "young, anxious"},
        {"name": "Raghunath", "gender": "male", "description": "elderly, gentle"},
    ],
    "labels": [
        {"id": 1, "speaker": "Narrator", "emotion": "neutral", "drop": False},
        {"id": 2, "speaker": "Raghunath", "emotion": "calm", "drop": False},
        {"id": 3, "speaker": "Narrator", "emotion": "neutral", "drop": True},
        {"id": 4, "speaker": "Raghunath", "emotion": "calm", "drop": False},
        {"id": 5, "speaker": "Narrator", "emotion": "neutral", "drop": False},
        {"id": 6, "speaker": "Meera", "emotion": "fearful", "drop": False},
    ],
}


def labels(**changes):
    return parse_labels(json.dumps({**GOOD_LABELS, **changes}))


def relabel(segment_id, **changes):
    """GOOD_LABELS with one segment's label changed."""
    return labels(
        labels=[
            {**label, **changes} if label["id"] == segment_id else label
            for label in GOOD_LABELS["labels"]
        ]
    )


class SegmentStoryTests(unittest.TestCase):
    def test_splits_dialogue_from_narration(self):
        segments = segment_story(STORY)
        self.assertEqual(
            [(s.kind, s.text) for s in segments],
            [
                ("narration", "Rain hammered the roof. Meera froze."),
                ("dialogue", "Don't be afraid."),
                ("narration", "He said quietly."),
                ("dialogue", "My name is Raghunath."),
                ("narration", "Meera pulled the envelope closer."),
                ("dialogue", "Who sent you?"),
            ],
        )
        self.assertEqual([s.id for s in segments], [1, 2, 3, 4, 5, 6])

    def test_handles_curly_quotes(self):
        segments = segment_story("She smiled. “Namaste!”")
        self.assertEqual([s.kind for s in segments], ["narration", "dialogue"])
        self.assertEqual(segments[1].text, "Namaste!")

    def test_story_without_quotes_is_all_narration(self):
        segments = segment_story("The sun set. The village slept.")
        self.assertEqual([s.kind for s in segments], ["narration"])


class SplitTitleTests(unittest.TestCase):
    def test_detects_title_line(self):
        title, body = split_title("The Last Local\n\nRain hammered the roof.")
        self.assertEqual(title, "The Last Local")
        self.assertEqual(body.strip(), "Rain hammered the roof.")

    def test_first_sentence_is_not_a_title(self):
        title, _ = split_title("Rain hammered the roof.\n\nMeera froze.")
        self.assertIsNone(title)


class BuildScriptTests(unittest.TestCase):
    def setUp(self):
        self.segments = segment_story(STORY)

    def test_builds_script_and_drops_speech_tag(self):
        script, dropped = build_script(self.segments, labels())
        self.assertEqual(dropped, ["He said quietly."])
        self.assertEqual(
            [(line.speaker, line.text) for line in script.lines],
            [
                ("Narrator", "Rain hammered the roof. Meera froze."),
                # the two halves of his speech merge once the tag is gone
                ("Raghunath", "Don't be afraid. My name is Raghunath."),
                ("Narrator", "Meera pulled the envelope closer."),
                ("Meera", "Who sent you?"),
            ],
        )

    def test_narration_always_goes_to_narrator(self):
        script, _ = build_script(self.segments, relabel(5, speaker="Meera"))
        self.assertEqual(script.lines[2].speaker, "Narrator")

    def test_long_narration_is_never_dropped(self):
        segments = segment_story(
            "In the sudden silence of the empty coach, she could hear her own heartbeat. "
            '"Who sent you?"'
        )
        answer = labels(
            labels=[
                {"id": 1, "speaker": "Narrator", "emotion": "calm", "drop": True},
                {"id": 2, "speaker": "Meera", "emotion": "fearful", "drop": False},
            ]
        )
        script, dropped = build_script(segments, answer)
        self.assertEqual(dropped, [])
        self.assertEqual(len(script.lines), 2)

    def test_rejects_dialogue_given_to_narrator(self):
        with self.assertRaisesRegex(ValueError, "segment 6"):
            build_script(self.segments, relabel(6, speaker="Narrator"))

    def test_rejects_dialogue_given_to_unknown_character(self):
        with self.assertRaisesRegex(ValueError, 'Add "Kabir" to "characters"'):
            build_script(self.segments, relabel(2, speaker="Kabir"))

    def test_rejects_missing_label(self):
        with self.assertRaisesRegex(ValueError, "exactly one entry"):
            build_script(self.segments, labels(labels=GOOD_LABELS["labels"][:5]))

    def test_removes_characters_who_never_speak(self):
        extra = GOOD_LABELS["characters"] + [
            {"name": "Sharma ji", "gender": "male", "description": "Meera's late father"}
        ]
        script, _ = build_script(self.segments, labels(characters=extra))
        # listed in order of first line: Raghunath speaks first
        self.assertEqual([c.name for c in script.characters], ["Raghunath", "Meera"])


class SeriesCharacterTests(unittest.TestCase):
    """Characters carried over from earlier episodes of a series."""

    KNOWN = [Character(name="Raghunath", gender="male", description="elderly, gentle")]

    def setUp(self):
        self.segments = segment_story(STORY)

    def test_known_character_may_speak_without_being_listed_again(self):
        answer = labels(characters=[GOOD_LABELS["characters"][0]])  # only Meera listed
        script, _ = build_script(self.segments, answer, self.KNOWN)
        self.assertEqual([c.name for c in script.characters], ["Raghunath", "Meera"])

    def test_known_details_win_over_new_ones(self):
        changed = [GOOD_LABELS["characters"][0],
                   {"name": "Raghunath", "gender": "female", "description": "a girl"}]
        script, _ = build_script(self.segments, labels(characters=changed), self.KNOWN)
        raghunath = next(c for c in script.characters if c.name == "Raghunath")
        self.assertEqual(raghunath.gender, "male")

    def test_rejects_a_variant_of_a_known_name(self):
        known = [Character(name="Inspector Raghunath", gender="male", description="police")]
        with self.assertRaisesRegex(ValueError, 'use exactly "Inspector Raghunath"'):
            build_script(self.segments, labels(), known)

    def test_looks_like(self):
        self.assertTrue(looks_like("राठौर", "इंस्पेक्टर राठौर"))
        self.assertTrue(looks_like("Mehta", "Mehta Sahab"))
        self.assertTrue(looks_like("Vickram", "Vikram"))
        self.assertFalse(looks_like("Kabir", "Meera"))
        self.assertFalse(looks_like("दादाजी", "दादी"))  # grandfather is not grandmother

    def test_hindi_story_is_segmented(self):
        segments = segment_story('दरवाज़ा खुला। "आ गई मेरी बच्ची," दादी ने कहा। "बैठो।"')
        self.assertEqual(
            [(s.kind, s.text) for s in segments],
            [("narration", "दरवाज़ा खुला।"), ("dialogue", "आ गई मेरी बच्ची."),
             ("narration", "दादी ने कहा।"), ("dialogue", "बैठो।")],
        )


class ParseLabelsTests(unittest.TestCase):
    def test_accepts_json_inside_code_fence(self):
        parsed = parse_labels("```json\n" + json.dumps(GOOD_LABELS) + "\n```")
        self.assertEqual(parsed.title, "The Last Local")

    def test_rejects_text_that_is_not_json(self):
        with self.assertRaises(ValueError):
            parse_labels("Here are your labels!")

    def test_rejects_unknown_emotion(self):
        with self.assertRaises(ValueError):
            relabel(2, emotion="furious")


if __name__ == "__main__":
    unittest.main()
