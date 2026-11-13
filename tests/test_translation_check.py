"""Offline tests for the translation checker. They need no API key.

Run with:  python -m unittest discover tests
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import translation_check as tc  # noqa: E402

SOURCE = {
    "title": "Test", "language": "en-IN",
    "characters": [{"name": "Meera", "gender": "female", "description": "young"}],
    "lines": [
        {"speaker": "Narrator", "text": "Rain fell.", "emotion": "neutral"},
        {"speaker": "Meera", "text": "Papa trusted nobody.", "emotion": "sad"},
    ],
}
TARGET = {**SOURCE, "language": "hi-IN", "lines": [
    {"speaker": "Narrator", "text": "बारिश हुई।", "emotion": "neutral"},
    {"speaker": "Meera", "text": "पापा पर किसी ने भरोसा नहीं किया।", "emotion": "sad"},
]}
BACK = {"बारिश हुई।": "Rain fell.", "पापा पर किसी ने भरोसा नहीं किया।": "Nobody trusted Papa."}


class ChrfTests(unittest.TestCase):
    def test_identical_text_scores_100(self):
        self.assertEqual(tc.chrf("Papa trusted nobody.", "papa trusted nobody"), 100.0)

    def test_unrelated_text_scores_low(self):
        self.assertLess(tc.chrf("Papa trusted nobody.", "The train left Bandra."), 30)

    def test_reversed_meaning_can_still_score_high(self):
        # the reason chrF alone is not enough: same words, opposite meaning
        self.assertGreater(tc.chrf("Papa trusted nobody.", "Nobody trusted Papa."), 50)

    def test_works_for_indian_scripts(self):
        self.assertEqual(tc.chrf("कबीर पुणे में था।", "कबीर पुणे में था"), 100.0)


class SplitTests(unittest.TestCase):
    def test_short_text_is_one_piece(self):
        self.assertEqual(tc.split_for_mayura("One. Two."), ["One. Two."])

    def test_long_text_is_split_at_sentences(self):
        text = " ".join(f"Sentence number {i} is here." for i in range(60))
        pieces = tc.split_for_mayura(text, limit=200)
        self.assertTrue(all(len(piece) <= 200 for piece in pieces))
        self.assertEqual(" ".join(pieces), text)


class CheckTranslationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        story = Path(self.folder.name)
        self.lang_dir = story / "hi-IN"
        self.lang_dir.mkdir()
        (story / "script.json").write_text(json.dumps(SOURCE), encoding="utf-8")
        (self.lang_dir / "script.json").write_text(json.dumps(TARGET, ensure_ascii=False),
                                                   encoding="utf-8")

    def run_check(self):
        def fake_post(path, payload, **_):
            self.assertEqual(payload["target_language_code"], "en-IN")
            return {"translated_text": BACK[payload["input"]]}

        verdicts = json.dumps({"verdicts": [
            {"id": 1, "reason": "Same meaning.", "verdict": "ok"},
            {"id": 2, "reason": "Who trusted whom is reversed.", "verdict": "wrong"},
        ]})
        with mock.patch.object(tc.sarvam_client, "post", side_effect=fake_post) as post, \
             mock.patch.object(tc.sarvam_client, "chat", return_value=verdicts) as chat:
            report = tc.check_translation(self.lang_dir)
        return report, post, chat

    def test_report_combines_score_and_verdict(self):
        report, _, chat = self.run_check()
        self.assertEqual(report["summary"]["ok"], 1)
        self.assertEqual(report["summary"]["wrong"], 1)
        self.assertEqual(report["lines"][0]["chrf"], 100.0)
        self.assertEqual(report["lines"][1]["back_translation"], "Nobody trusted Papa.")
        self.assertEqual(chat.call_count, 1)  # one judge call for the whole script
        self.assertTrue((self.lang_dir / "translation_report.json").exists())

    def test_sends_the_speakers_gender(self):
        _, post, _ = self.run_check()
        self.assertEqual(post.call_args_list[1].args[1]["speaker_gender"], "Female")

    def test_back_translations_are_cached(self):
        self.run_check()
        _, post, _ = self.run_check()
        post.assert_not_called()

    def test_rejects_the_original_language_folder(self):
        with self.assertRaisesRegex(ValueError, "original"):
            (Path(self.folder.name) / "en-IN").mkdir()
            (Path(self.folder.name) / "en-IN" / "script.json").write_text(json.dumps(SOURCE))
            tc.check_translation(Path(self.folder.name) / "en-IN")


if __name__ == "__main__":
    unittest.main()
