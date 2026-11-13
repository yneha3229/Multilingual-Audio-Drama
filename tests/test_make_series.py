"""Offline tests for the series runner. They need no API key.

Run with:  python -m unittest discover tests
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import make_series  # noqa: E402
import tts_engine  # noqa: E402
from script_writer import Script  # noqa: E402

EPISODE_SCRIPTS = {
    "ep01": {"characters": [("Ananya", "female"), ("Vikram", "male")]},
    "ep02": {"characters": [("Ananya", "female"), ("Rathore", "male")]},
}


def fake_run_pipeline(story, name, languages, check=False, fresh=False, log=print,
                      known=None, cast=None):
    """Stands in for the real pipeline and records what it was given."""
    episode = name.split("/")[-1]
    characters = [
        {"name": n, "gender": g, "description": "test"}
        for n, g in EPISODE_SCRIPTS[episode]["characters"]
    ]
    script = Script.model_validate({
        "title": episode, "language": "hi-IN", "characters": characters,
        "lines": [{"speaker": c["name"], "text": "नमस्ते।", "emotion": "neutral"}
                  for c in characters],
    })
    voices = tts_engine.assign_voices(script, cast)
    fake_run_pipeline.calls.append({"episode": episode, "known": known, "cast": dict(cast)})
    manifest = {"voices": voices, "duration": 1.0}
    return {"script": script,
            "episodes": [{"code": "hi-IN", "script": script, "manifest": manifest, "report": None}]}


class RunSeriesTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        root = Path(self.folder.name)
        self.series_dir = root / "stories" / "haveli"
        self.series_dir.mkdir(parents=True)
        for episode in EPISODE_SCRIPTS:
            (self.series_dir / f"{episode}.txt").write_text("कहानी", encoding="utf-8")

        fake_run_pipeline.calls = []
        for patcher in (
            mock.patch.object(make_series, "run_pipeline", side_effect=fake_run_pipeline),
            mock.patch.object(make_series, "OUTPUT_DIR", root / "output"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.bible_path = root / "output" / "haveli" / "series.json"

    def run_series(self):
        return make_series.run_series(self.series_dir, [], log=lambda message: None)

    def test_second_episode_is_given_the_first_episodes_cast(self):
        self.run_series()
        first, second = fake_run_pipeline.calls
        self.assertEqual(first["cast"], {})
        self.assertEqual(set(second["cast"]), {"Ananya", "Vikram"})
        self.assertEqual([c.name for c in second["known"]], ["Ananya", "Vikram"])

    def test_bible_records_each_character_once_with_first_episode(self):
        _, bible = self.run_series()
        saved = json.loads(self.bible_path.read_text(encoding="utf-8"))
        self.assertEqual(saved, bible)
        self.assertEqual(
            [(c["name"], c["first_episode"]) for c in bible["characters"]],
            [("Ananya", "ep01"), ("Vikram", "ep01"), ("Rathore", "ep02")],
        )

    def test_voices_stay_the_same_and_new_characters_get_new_voices(self):
        results, bible = self.run_series()
        voices = make_series.voice_consistency([result for _, result in results])
        self.assertTrue(all(len(v) == 1 for v in voices.values()))
        by_name = {c["name"]: c["voice"] for c in bible["characters"]}
        self.assertNotEqual(by_name["Rathore"], by_name["Vikram"])

    def test_running_again_reuses_the_bible(self):
        self.run_series()
        fake_run_pipeline.calls = []
        _, bible = self.run_series()
        self.assertEqual(set(fake_run_pipeline.calls[0]["cast"]), {"Ananya", "Vikram", "Rathore"})
        self.assertEqual(len(bible["characters"]), 3)

    def test_empty_folder_is_an_error(self):
        empty = Path(self.folder.name) / "empty"
        empty.mkdir()
        with self.assertRaisesRegex(ValueError, "No episode files"):
            make_series.run_series(empty, [], log=lambda message: None)


if __name__ == "__main__":
    unittest.main()
