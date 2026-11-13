"""Offline test for the web app. It needs no API key.

Streamlit's test runner executes app.py without a browser. The pipeline is
replaced by a fake that returns a small, ready-made result.

Run with:  python -m unittest discover tests
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamlit.testing.v1 import AppTest  # noqa: E402

import tts_engine  # noqa: E402
from script_writer import Script  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")

SCRIPT = Script.model_validate(
    {
        "title": "Test Story",
        "language": "en-IN",
        "characters": [{"name": "Meera", "gender": "female", "description": "young"}],
        "lines": [
            {"speaker": "Narrator", "text": "Rain fell.", "emotion": "neutral"},
            {"speaker": "Meera", "text": "Who is it?", "emotion": "fearful"},
        ],
    }
)


def fake_result(folder: Path) -> dict:
    clips = folder / "clips"
    (clips / "rejected").mkdir(parents=True)
    entries = []
    for number, line in enumerate(SCRIPT.lines, start=1):
        clip = clips / f"{number:03d}.wav"
        import wave

        for path in (clip, clips / "rejected" / clip.name):
            with wave.open(str(path), "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(tts_engine.SAMPLE_RATE)
                out.writeframes(tts_engine.silence(0.2))
        entries.append({"line": number, "speaker": line.speaker, "voice": "shubh",
                        "emotion": line.emotion, "pace": 1.0, "temperature": 0.5,
                        "text": line.text, "clip": f"clips/{clip.name}"})
    manifest = {"title": SCRIPT.title, "language": "en-IN",
                "voices": {"Narrator": "shubh", "Meera": "priya"},
                "episode": "episode.wav", "lines": entries}
    tts_engine.stitch_episode(manifest, folder)

    take = {"transcript": "who is it", "wer": 0.0, "cer": 0.0, "rate": 12.0, "pauses": 0.0}
    report = {
        "language": "en-IN",
        "summary": {"lines": 2, "usual_rate": 12.0, "failing_before": 1, "fixed": 1,
                    "failing_after": 0, "wer_before": 0.0, "wer_after": 0.0,
                    "cer_before": 0.0, "cer_after": 0.0},
        "lines": [
            {"line": 1, "speaker": "Narrator", "text": "Rain fell.", "status": "ok",
             "problems": [], "retries": 0, "before": take, "after": take},
            {"line": 2, "speaker": "Meera", "text": "Who is it?", "status": "fixed",
             "problems": ["slow"], "retries": 1, "before": {**take, "rate": 6.0}, "after": take},
        ],
    }
    episode = {"code": "en-IN", "folder": folder, "script": SCRIPT,
               "manifest": manifest, "report": report}
    return {"script": SCRIPT, "episodes": [episode]}


class AppTests(unittest.TestCase):
    def test_page_loads_with_the_sample_story(self):
        app = AppTest.from_file(APP, default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertIn("Meera", app.text_area[0].value)

    def test_generate_shows_cast_episode_and_quality(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch(
            "make_drama.run_pipeline", return_value=fake_result(Path(folder))
        ) as pipeline:
            app = AppTest.from_file(APP, default_timeout=30).run()
            app.button[0].click().run()

            self.assertFalse(app.exception)
            self.assertEqual(pipeline.call_args.args[1], "sample_story")  # shares saved work
            self.assertEqual(pipeline.call_args.args[2], ["hi-IN"])
            page = " ".join(element.value for element in app.markdown)
            self.assertIn("Test Story", page)
            self.assertIn("Line 2, Meera", page)
            self.assertEqual([metric.value for metric in app.metric], ["2", "1", "0", "0.0%"])

    def test_pipeline_error_is_shown_not_raised(self):
        import sarvam_client

        with mock.patch("make_drama.run_pipeline",
                        side_effect=sarvam_client.SarvamError("SARVAM_API_KEY is not set.")):
            app = AppTest.from_file(APP, default_timeout=30).run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertIn("SARVAM_API_KEY", app.error[0].value)


if __name__ == "__main__":
    unittest.main()
