"""Offline tests for the quality checker. They need no API key.

Run with:  python -m unittest discover tests
"""

import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import quality_check as qc  # noqa: E402
import tts_engine  # noqa: E402

RATE = tts_engine.SAMPLE_RATE


def tone(seconds: float) -> bytes:
    """A loud square wave: stands in for speech."""
    return b"".join((b"\x00\x40" if (i // 50) % 2 else b"\x00\xc0") for i in range(int(RATE * seconds)))


def wav_bytes(frames: bytes) -> bytes:
    import io

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as clip:
        clip.setnchannels(1)
        clip.setsampwidth(2)
        clip.setframerate(RATE)
        clip.writeframes(frames)
    return buffer.getvalue()


class NormalizeTests(unittest.TestCase):
    def test_ignores_case_and_punctuation(self):
        self.assertEqual(qc.normalize("Listen to me! Nobody!"), "listen to me nobody")

    def test_hindi_spelling_variants_match(self):
        self.assertEqual(qc.normalize("यहाँ ज़ोर"), qc.normalize("यहां जोर"))

    def test_bengali_digits_match_ascii_digits(self):
        self.assertEqual(qc.normalize("১১:৪০"), qc.normalize("11:40"))


class ErrorRateTests(unittest.TestCase):
    def test_identical_text_scores_zero(self):
        self.assertEqual(qc.wer("Papa trusted nobody.", "papa trusted nobody"), 0.0)
        self.assertEqual(qc.cer("Papa trusted nobody.", "papa trusted nobody"), 0.0)

    def test_one_wrong_word_in_four(self):
        self.assertEqual(qc.wer("papa trusted nobody there", "papa trusted somebody there"), 0.25)

    def test_missing_and_extra_words_count(self):
        self.assertEqual(qc.wer("a b c d", "a b c"), 0.25)
        self.assertEqual(qc.wer("a b c d", "a b c d e"), 0.25)

    def test_cer_ignores_spacing(self):
        self.assertEqual(qc.cer("சர்ச்கேட்ல", "சர்ச் கேட்ல"), 0.0)
        self.assertGreater(qc.wer("சர்ச்கேட்ல", "சர்ச் கேட்ல"), 0.0)

    def test_cer_counts_wrong_characters(self):
        self.assertAlmostEqual(qc.cer("kabir", "kabeer"), 2 / 5)

    def test_empty_transcript_is_fully_wrong(self):
        self.assertEqual(qc.wer("hello there", ""), 1.0)


class AudioMeasureTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "clip.wav"

    def test_speaking_rate_factors_out_pace(self):
        self.path.write_bytes(wav_bytes(tone(2.0)))
        self.assertAlmostEqual(qc.speaking_rate(self.path, "abcd efgh", 1.0), 4.0, delta=0.1)
        self.assertAlmostEqual(qc.speaking_rate(self.path, "abcd efgh", 0.5), 8.0, delta=0.2)

    def test_pause_seconds_finds_internal_silence_only(self):
        frames = (
            tts_engine.silence(0.5) + tone(1.0) + tts_engine.silence(0.6) + tone(1.0)
            + tts_engine.silence(0.1) + tone(0.5) + tts_engine.silence(0.5)
        )
        self.path.write_bytes(wav_bytes(frames))
        # the 0.6 s gap counts; the 0.1 s gap is too short; the ends do not count
        self.assertAlmostEqual(qc.pause_seconds(self.path), 0.6, delta=0.05)


class LongClipTests(unittest.TestCase):
    def test_short_audio_is_not_split(self):
        samples = __import__("array").array("h", tone(5.0))
        self.assertEqual(len(qc.split_audio(samples, RATE, max_seconds=10)), 1)

    def test_long_audio_is_cut_at_the_pause(self):
        import array

        # 8 s of speech, a 0.5 s pause, then 6 s more: with a 10 s limit the
        # cut must land inside the pause, not at exactly 10 s
        samples = array.array("h", tone(8.0) + tts_engine.silence(0.5) + tone(6.0))
        pieces = qc.split_audio(samples, RATE, max_seconds=10)
        self.assertEqual(len(pieces), 2)
        self.assertTrue(8.0 <= len(pieces[0]) / RATE <= 8.5)
        self.assertEqual(sum(len(piece) for piece in pieces), len(samples))
        self.assertTrue(all(len(piece) <= 10 * RATE for piece in pieces))

    def test_long_clip_is_transcribed_in_pieces(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "long.wav"
            path.write_bytes(wav_bytes(tone(20.0) + tts_engine.silence(0.5) + tone(15.0)))
            replies = iter([{"transcript": "first half"}, {"transcript": "second half"}])
            with mock.patch.object(qc.sarvam_client, "post", side_effect=lambda *a, **k: next(replies)) as post:
                text = qc.transcribe(path, "mr-IN", {})
            self.assertEqual(post.call_count, 2)
            self.assertEqual(text, "first half second half")


class JudgeTests(unittest.TestCase):
    def test_good_take_passes(self):
        self.assertEqual(qc.judge({"cer": 0.02, "rate": 15.0, "letters": 40}, usual_rate=15.0), [])

    def test_misread_take_fails(self):
        self.assertEqual(qc.judge({"cer": 0.30, "rate": 15.0, "letters": 40}, usual_rate=15.0), ["misread"])

    def test_slow_take_fails(self):
        self.assertEqual(qc.judge({"cer": 0.0, "rate": 9.0, "letters": 40}, usual_rate=15.0), ["slow"])

    def test_very_short_line_is_never_slow(self):
        self.assertEqual(qc.judge({"cer": 0.0, "rate": 4.0, "letters": 7}, usual_rate=15.0), [])

    def test_narration_and_dialogue_have_separate_usual_rates(self):
        entries = [{"speaker": "Narrator"}] * 3 + [{"speaker": "Meera"}] * 3
        results = [{"rate": r, "letters": 40} for r in (16, 15, 17, 10, 9, 11)]
        self.assertEqual(qc.usual_rates(entries, results), {"narration": 16, "dialogue": 10})

    def test_small_group_falls_back_to_the_overall_rate(self):
        entries = [{"speaker": "Narrator"}] * 4 + [{"speaker": "Meera"}]
        results = [{"rate": r, "letters": 40} for r in (16, 15, 17, 14, 9)]
        self.assertEqual(qc.usual_rates(entries, results)["dialogue"], 15)

    def test_clean_take_beats_slow_take(self):
        slow = {"cer": 0.0, "rate": 9.0, "letters": 40}
        clean = {"cer": 0.05, "rate": 15.0, "letters": 40}
        self.assertLess(qc.badness(clean, 15.0), qc.badness(slow, 15.0))


class CheckEpisodeTests(unittest.TestCase):
    """Runs the whole check-and-retry loop with Sarvam replaced by fakes."""

    TEXTS = ["the rain fell hard all night", "who sent you here so late", "nobody sent me here beta"]

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.out_dir = Path(self.folder.name)
        (self.out_dir / "clips").mkdir()

        entries = []
        for number, text in enumerate(self.TEXTS, start=1):
            clip = f"clips/{number:03d}.wav"
            # every clip lasts 1.0 s, except line 2 which is a slow 3.0 s take
            seconds = 3.0 if number == 2 else 1.0
            (self.out_dir / clip).write_bytes(wav_bytes(tone(seconds)))
            entries.append({"line": number, "speaker": "Narrator", "voice": "shubh",
                            "emotion": "neutral", "pace": 1.0, "temperature": 0.5,
                            "text": text, "clip": clip})
        manifest = {"title": "T", "language": "en-IN", "voices": {}, "episode": "episode.wav",
                    "lines": entries}
        tts_engine.stitch_episode(manifest, self.out_dir)

        quiet = mock.patch("builtins.print")
        quiet.start()
        self.addCleanup(quiet.stop)

    def run_check(self, transcripts, new_take_seconds=1.0, retries=2):
        """transcripts: what speech-to-text 'hears', by clip length in seconds."""
        def fake_post(path, payload, files=None, **_):
            with wave.open(__import__("io").BytesIO(files["file"][1]), "rb") as clip:
                seconds = round(clip.getnframes() / clip.getframerate(), 1)
            return {"transcript": transcripts[seconds]}

        with mock.patch.object(qc.sarvam_client, "post", side_effect=fake_post), \
             mock.patch.object(qc.tts_engine, "synthesize",
                               return_value=wav_bytes(tone(new_take_seconds))) as synthesize:
            report = qc.check_episode(self.out_dir, retries)
        return report, synthesize

    def test_slow_take_is_replaced_and_episode_restitched(self):
        # speech-to-text hears every clip perfectly, so only speed can fail
        with mock.patch.object(qc, "cer", return_value=0.0), mock.patch.object(qc, "wer", return_value=0.0):
            report, synthesize = self.run_check({1.0: "x", 3.0: "x"})

        self.assertEqual([line["status"] for line in report["lines"]], ["ok", "fixed", "ok"])
        self.assertEqual(report["lines"][1]["problems"], ["slow"])
        self.assertEqual(synthesize.call_count, 1)
        self.assertEqual(report["summary"]["failing_before"], 1)
        self.assertEqual(report["summary"]["failing_after"], 0)
        # the old take is kept for comparison and the episode got shorter
        self.assertTrue((self.out_dir / "clips" / "rejected" / "002.wav").exists())
        manifest = json.loads((self.out_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["lines"][1]["duration"], 1.0)
        self.assertTrue((self.out_dir / "quality_report.json").exists())

    def test_running_again_still_reports_what_was_fixed(self):
        with mock.patch.object(qc, "cer", return_value=0.0), mock.patch.object(qc, "wer", return_value=0.0):
            self.run_check({1.0: "x", 3.0: "x"})
            report, synthesize = self.run_check({1.0: "x", 3.0: "x"})

        self.assertEqual(report["lines"][1]["status"], "fixed")
        self.assertEqual(report["summary"]["failing_before"], 1)
        self.assertEqual(report["summary"]["failing_after"], 0)
        synthesize.assert_not_called()  # already fixed, so no new takes

    def test_line_that_never_improves_is_reported_as_still_failing(self):
        with mock.patch.object(qc, "cer", return_value=0.0), mock.patch.object(qc, "wer", return_value=0.0):
            report, synthesize = self.run_check({1.0: "x", 3.0: "x"}, new_take_seconds=3.0)

        self.assertEqual(report["lines"][1]["status"], "still failing")
        self.assertEqual(synthesize.call_count, 2)  # used every retry
        self.assertEqual(report["summary"]["failing_after"], 1)
        self.assertFalse((self.out_dir / "clips" / "rejected").exists())  # nothing swapped

    def test_zero_retries_only_measures(self):
        with mock.patch.object(qc, "cer", return_value=0.0), mock.patch.object(qc, "wer", return_value=0.0):
            report, synthesize = self.run_check({1.0: "x", 3.0: "x"}, retries=0)
        synthesize.assert_not_called()
        self.assertEqual(report["lines"][1]["status"], "still failing")

    def test_second_run_does_not_pay_for_speech_to_text_again(self):
        with mock.patch.object(qc.sarvam_client, "post", return_value={"transcript": "x"}) as post, \
             mock.patch.object(qc.tts_engine, "synthesize", return_value=wav_bytes(tone(3.0))):
            qc.check_episode(self.out_dir, 0)
            calls = post.call_count
            qc.check_episode(self.out_dir, 0)
            self.assertEqual(post.call_count, calls)


if __name__ == "__main__":
    unittest.main()
