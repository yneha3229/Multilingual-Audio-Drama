"""Offline tests for the voice engine. They need no API key.

Run with:  python -m unittest discover tests
"""

import base64
import io
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tts_engine  # noqa: E402
from script_writer import Script  # noqa: E402

SCRIPT = Script.model_validate(
    {
        "title": "Test",
        "language": "en-IN",
        "characters": [
            {"name": "Meera", "gender": "female", "description": "young"},
            {"name": "Raghunath", "gender": "male", "description": "old"},
            {"name": "Kabir", "gender": "male", "description": "brother"},
        ],
        "lines": [
            {"speaker": "Narrator", "text": "Rain fell.", "emotion": "neutral"},
            {"speaker": "Meera", "text": "Who is it?", "emotion": "fearful"},
            {"speaker": "Meera", "text": "Oh. It is you.", "emotion": "calm"},
            {"speaker": "Raghunath", "text": "Only me.", "emotion": "whispering"},
            {"speaker": "Kabir", "text": "Stop!", "emotion": "angry"},
        ],
    }
)

CLIP_SECONDS = 1.0


def fake_sarvam_response(path, payload, **_):
    """Stands in for Sarvam: returns one second of silence as base64 WAV."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as clip:
        clip.setnchannels(1)
        clip.setsampwidth(2)
        clip.setframerate(tts_engine.SAMPLE_RATE)
        clip.writeframes(tts_engine.silence(CLIP_SECONDS))
    return {"audios": [base64.b64encode(buffer.getvalue()).decode()]}


class AssignVoicesTests(unittest.TestCase):
    def test_every_speaker_gets_a_different_voice(self):
        voices = tts_engine.assign_voices(SCRIPT)
        self.assertEqual(set(voices), {"Narrator", "Meera", "Raghunath", "Kabir"})
        self.assertEqual(len(set(voices.values())), 4)

    def test_voice_matches_gender(self):
        voices = tts_engine.assign_voices(SCRIPT)
        self.assertIn(voices["Meera"], tts_engine.FEMALE_VOICES)
        self.assertIn(voices["Raghunath"], tts_engine.MALE_VOICES)

    def test_casting_is_the_same_every_time(self):
        self.assertEqual(tts_engine.assign_voices(SCRIPT), tts_engine.assign_voices(SCRIPT))

    def test_series_cast_keeps_old_voices_and_gives_new_ones_free_voices(self):
        cast = {"Kabir": "aditya", "Meera": "neha"}  # from earlier episodes
        voices = tts_engine.assign_voices(SCRIPT, cast)
        self.assertEqual(voices["Kabir"], "aditya")
        self.assertEqual(voices["Meera"], "neha")
        self.assertIn(voices["Raghunath"], tts_engine.MALE_VOICES)
        self.assertNotEqual(voices["Raghunath"], "aditya")

    def test_only_this_scripts_characters_are_returned(self):
        voices = tts_engine.assign_voices(SCRIPT, {"Someone Else": "rohan"})
        self.assertNotIn("Someone Else", voices)
        self.assertNotIn("rohan", [voices["Raghunath"], voices["Kabir"]])

    def test_narrator_voice_is_not_reused_for_a_character(self):
        self.assertNotIn(tts_engine.NARRATOR_VOICE, tts_engine.MALE_VOICES)
        self.assertNotIn(tts_engine.NARRATOR_VOICE, tts_engine.FEMALE_VOICES)


class EmotionStyleTests(unittest.TestCase):
    def test_every_emotion_has_a_valid_style(self):
        for line in SCRIPT.lines:
            self.assertIn(line.emotion, tts_engine.EMOTION_STYLE)
        for style in tts_engine.EMOTION_STYLE.values():
            self.assertTrue(0.5 <= style["pace"] <= 2.0)
            self.assertTrue(0.01 <= style["temperature"] <= 1.0)


class MakeEpisodeTests(unittest.TestCase):
    def setUp(self):
        # make_episode prints progress for each line; keep test output quiet
        quiet = mock.patch("builtins.print")
        quiet.start()
        self.addCleanup(quiet.stop)

    def test_builds_episode_with_pauses_and_manifest(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(
            tts_engine.sarvam_client, "post", side_effect=fake_sarvam_response
        ) as post:
            out_dir = Path(folder)
            manifest = tts_engine.make_episode(SCRIPT, out_dir)

            self.assertEqual(post.call_count, 5)
            # 5 one-second clips, 3 speaker changes, 1 same-speaker gap
            expected = (
                5 * CLIP_SECONDS
                + 3 * tts_engine.PAUSE_NEW_SPEAKER
                + 1 * tts_engine.PAUSE_SAME_SPEAKER
            )
            self.assertAlmostEqual(manifest["duration"], expected, places=2)
            with wave.open(str(out_dir / "episode.wav"), "rb") as episode:
                seconds = episode.getnframes() / episode.getframerate()
            self.assertAlmostEqual(seconds, expected, places=2)

            saved = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(saved["lines"]), 5)
            self.assertEqual(saved["lines"][0]["start"], 0.0)
            for entry in saved["lines"]:
                self.assertTrue((out_dir / entry["clip"]).exists())

    def test_sends_the_emotion_style_and_voice_to_sarvam(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(
            tts_engine.sarvam_client, "post", side_effect=fake_sarvam_response
        ) as post:
            tts_engine.make_episode(SCRIPT, Path(folder))
            payload = post.call_args_list[3].args[1]  # Raghunath, whispering
            self.assertEqual(payload["pace"], tts_engine.EMOTION_STYLE["whispering"]["pace"])
            self.assertEqual(payload["speaker"], tts_engine.assign_voices(SCRIPT)["Raghunath"])
            self.assertEqual(payload["language_code"], "en-IN")

    def test_second_run_reuses_clips_instead_of_calling_sarvam(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(
            tts_engine.sarvam_client, "post", side_effect=fake_sarvam_response
        ) as post:
            tts_engine.make_episode(SCRIPT, Path(folder))
            tts_engine.make_episode(SCRIPT, Path(folder))
            self.assertEqual(post.call_count, 5)


if __name__ == "__main__":
    unittest.main()
