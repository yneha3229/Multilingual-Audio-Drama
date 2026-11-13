"""Offline tests for background music. They need no API key and no Stable Audio.

Run with:  python -m unittest discover tests
"""

import json
import math
import subprocess
import sys
import tempfile
import unittest
import wave
from array import array
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import background_music as bm  # noqa: E402
from script_writer import Script  # noqa: E402

RATE = bm.SAMPLE_RATE
GOOD = "slow Indian ambient underscore, soft sitar, low tanpura drone, gentle bansuri, 65 BPM, mysterious and melancholic"

SCRIPT = Script.model_validate({
    "title": "The Haveli", "language": "en-IN",
    "characters": [{"name": "Ananya", "gender": "female", "description": "young"}],
    "lines": [
        {"speaker": "Narrator", "text": "The door creaked.", "emotion": "neutral"},
        {"speaker": "Ananya", "text": "Who is there?", "emotion": "fearful"},
    ],
})


def tone(seconds: float, amplitude: int, hz: float = 220.0) -> array:
    return array("h", (round(amplitude * math.sin(2 * math.pi * hz * i / RATE))
                       for i in range(int(seconds * RATE))))


def save(path: Path, samples: array) -> None:
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(samples.tobytes())


def save_stereo_44k(path: Path, seconds: float, hz: float = 440.0, amplitude: int = 10000) -> None:
    """What the TFLite build writes: 44.1 kHz stereo 16-bit."""
    frames = array("h")
    for i in range(int(seconds * 44100)):
        value = round(amplitude * math.sin(2 * math.pi * hz * i / 44100))
        frames.extend((value, value))
    with wave.open(str(path), "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(44100)
        out.writeframes(frames.tobytes())


class ConvertTests(unittest.TestCase):
    def test_44k_stereo_becomes_voice_format_and_keeps_the_tone(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "in.wav", Path(tmp) / "out.wav"
            save_stereo_44k(source, 2.0, hz=440)
            bm.to_voice_format(source, target)
            samples = bm.read_wav(target)  # also checks 22050 Hz mono 16-bit
            self.assertEqual(len(samples), RATE * 2)
            middle = samples[RATE // 2:RATE // 2 + RATE // 2]
            crossings = sum(1 for a, b in zip(middle, middle[1:]) if a < 0 <= b)
            self.assertAlmostEqual(crossings, 220, delta=2)  # 440 Hz over 0.5 s
            self.assertAlmostEqual(max(samples), 0.9 * 32767, delta=200)

    def test_very_high_frequencies_are_filtered_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "in.wav", Path(tmp) / "out.wav"
            save_stereo_44k(source, 1.0, hz=16000)  # would alias to 6 kHz without the filter
            with wave.open(str(source), "rb") as clip:
                frames = array("h", clip.readframes(clip.getnframes()))
            # add a quiet 440 Hz tone so normalisation has something to keep
            for i in range(0, len(frames), 2):
                tone_value = round(2000 * math.sin(2 * math.pi * 440 * (i // 2) / 44100))
                frames[i] = frames[i + 1] = max(-32768, min(32767, frames[i] + tone_value))
            with wave.open(str(source), "wb") as out:
                out.setnchannels(2)
                out.setsampwidth(2)
                out.setframerate(44100)
                out.writeframes(frames.tobytes())
            bm.to_voice_format(source, target)
            samples = bm.read_wav(target)
            middle = samples[RATE // 4:RATE * 3 // 4]
            crossings = sum(1 for a, b in zip(middle, middle[1:]) if a < 0 <= b)
            self.assertAlmostEqual(crossings, 220, delta=4)  # only the 440 Hz tone is left


def make_episode(folder: Path, speech_amplitude: int = 8000) -> None:
    """2 s line, 3 s silence, 2 s line: a 7 s episode."""
    speech = tone(2, speech_amplitude) + array("h", bytes(2 * 3 * RATE)) + tone(2, speech_amplitude)
    save(folder / "episode.wav", speech)
    manifest = {"episode": "episode.wav", "duration": 7.0, "lines": [
        {"line": 1, "start": 0.0, "duration": 2.0},
        {"line": 2, "start": 5.0, "duration": 2.0},
    ]}
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (folder / "script.json").write_text(SCRIPT.model_dump_json(), encoding="utf-8")


def level_db(samples, start: float, end: float) -> float:
    part = samples[int(start * RATE):int(end * RATE)]
    return 20 * math.log10(bm.rms(part))


class PromptCheckTests(unittest.TestCase):
    def test_good_prompt_passes(self):
        self.assertEqual(bm.check_prompt(GOOD, ["Ananya"]), [])

    def test_non_english_is_rejected(self):
        problems = bm.check_prompt("धीमा सितार संगीत, उदास, रहस्यमय हवेली, 60 BPM, तानपूरा", [])
        self.assertTrue(any("English" in p for p in problems))

    def test_vocals_are_rejected_even_when_negated(self):
        # "no vocals" still puts the word in front of the music model
        problems = bm.check_prompt(GOOD + ", no vocals", [])
        self.assertTrue(any("vocals" in p for p in problems))

    def test_character_name_is_rejected(self):
        problems = bm.check_prompt(GOOD + ", Ananya's theme", ["Ananya"])
        self.assertTrue(any("Ananya" in p for p in problems))

    def test_length_limits(self):
        self.assertTrue(bm.check_prompt("sad sitar", []))
        self.assertTrue(bm.check_prompt(" ".join(["calm"] * 70), []))


class WritePromptTests(unittest.TestCase):
    def test_retries_with_the_problem_then_succeeds(self):
        bad = json.dumps({"mood": "tense", "prompt": GOOD + ", soft female vocals"})
        good = json.dumps({"mood": "mysterious", "prompt": GOOD})
        with mock.patch("sarvam_client.chat", side_effect=[bad, good]) as chat:
            result = bm.write_music_prompt(SCRIPT)
        self.assertEqual(result.prompt, GOOD)
        retry = chat.call_args_list[1].args[0]
        self.assertIn("vocals", retry[-1]["content"])
        self.assertIn("Who is there?", retry[1]["content"])  # the LLM saw the story

    def test_gives_up_after_max_attempts(self):
        bad = json.dumps({"mood": "x", "prompt": "sad"})
        with mock.patch("sarvam_client.chat", return_value=bad):
            with self.assertRaises(RuntimeError):
                bm.write_music_prompt(SCRIPT, max_attempts=2)


class EnvelopeTests(unittest.TestCase):
    def test_dips_under_speech_and_is_down_when_the_line_starts(self):
        envelope = bm.gain_envelope([(4.0, 6.0)], 12.0)
        block = lambda t: envelope[int(t / bm.BLOCK_SECONDS)]  # noqa: E731
        self.assertEqual(block(1.0), bm.GAP_DB)
        self.assertEqual(block(4.0), bm.UNDER_SPEECH_DB)  # attack happened before
        self.assertEqual(block(5.9), bm.UNDER_SPEECH_DB)
        self.assertGreater(block(6.3), bm.UNDER_SPEECH_DB)  # releasing
        self.assertEqual(block(8.0), bm.GAP_DB)

    def test_short_pause_does_not_pump_back_to_full(self):
        envelope = bm.gain_envelope([(1.0, 3.0), (3.4, 5.0)], 6.0)
        between = envelope[int(3.2 / bm.BLOCK_SECONDS)]
        self.assertLess(between, bm.GAP_DB - 6)


class FitLengthTests(unittest.TestCase):
    def test_cuts_long_music(self):
        self.assertEqual(len(bm.fit_length(array("h", [1] * 100), 40, 10)), 40)

    def test_loops_short_music_with_crossfade(self):
        out = bm.fit_length(array("h", [100] * 50), 200, 10)
        self.assertEqual(len(out), 200)
        self.assertTrue(all(abs(v - 100) < 1e-6 for v in out))  # no gap or jump


class MixTests(unittest.TestCase):
    def test_mix_levels_and_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            make_episode(folder)
            save(folder / "music.wav", tone(20, 2000, hz=330))
            levels = bm.mix_episode(folder, folder / "music.wav")

            with wave.open(str(folder / "episode_with_music.wav"), "rb") as clip:
                out = array("h", clip.readframes(clip.getnframes()))
            self.assertAlmostEqual(len(out) / RATE, 4 + 7 + 4, places=2)
            self.assertFalse(levels["looped"])

            speech_db = 20 * math.log10(8000 / math.sqrt(2))
            # intro: music alone at the gap level
            self.assertAlmostEqual(level_db(out, 2.0, 3.5) - speech_db, bm.GAP_DB, delta=0.5)
            # the 3 s pause between lines (6 s to 9 s in the mix) recovers to the gap level
            self.assertAlmostEqual(level_db(out, 7.5, 8.5) - speech_db, bm.GAP_DB, delta=0.5)
            # under the first line the speech dominates and the mix is barely louder
            self.assertAlmostEqual(level_db(out, 4.5, 5.5), speech_db, delta=0.5)
            self.assertEqual(out[0], 0)  # fade in starts from silence

    def test_slightly_short_music_shortens_the_outro_instead_of_looping(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            make_episode(folder)  # 7 s; with intro and outro it needs 15 s
            save(folder / "music.wav", tone(14, 2000))
            levels = bm.mix_episode(folder, folder / "music.wav")
            self.assertFalse(levels["looped"])
            self.assertAlmostEqual(levels["seconds"], 14.0, places=2)

    def test_short_music_is_looped(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            make_episode(folder)
            save(folder / "music.wav", tone(6, 2000))
            self.assertTrue(bm.mix_episode(folder, folder / "music.wav")["looped"])


class AddMusicTests(unittest.TestCase):
    def fake_worker(self, command, check):
        out = Path(command[command.index("--out") + 1])
        seconds = float(command[command.index("--seconds") + 1])
        save(out, tone(seconds, 2000))

    def test_end_to_end_and_reuse(self):
        reply = json.dumps({"mood": "mysterious", "prompt": GOOD})
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            make_episode(folder)
            with mock.patch("sarvam_client.chat", return_value=reply) as chat, \
                 mock.patch.object(bm, "backend", return_value="pytorch"), \
                 mock.patch.object(bm, "worker_python", return_value=Path("python")), \
                 mock.patch("subprocess.run", side_effect=self.fake_worker) as run:
                info = bm.add_music(folder, log=lambda *_: None)
                again = bm.add_music(folder, log=lambda *_: None)

            self.assertEqual(chat.call_count, 1)  # prompt saved and reused
            self.assertEqual(run.call_count, 1)  # music made once
            self.assertEqual(info, again)
            self.assertEqual(info["prompt"], GOOD)
            self.assertIn(f"--seconds 15.0", " ".join(run.call_args.args[0]))
            self.assertTrue((folder / "episode_with_music.wav").exists())
            self.assertTrue((folder / "episode.wav").exists())

    def test_tflite_backend_is_used_and_converted(self):
        def fake_tflite(command, check, cwd, stdin):
            self.assertEqual(command[command.index("--dit") + 1], "sm-music")
            self.assertEqual(command[command.index("--decoder") + 1], "same-s")
            self.assertIs(stdin, subprocess.DEVNULL)  # can never hang on a question
            out = Path(command[command.index("--out") + 1])
            seconds = float(command[command.index("--seconds") + 1])
            save_stereo_44k(out, seconds)

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with mock.patch.object(bm, "backend", return_value="tflite"), \
                 mock.patch.object(bm, "tflite_dir", return_value=folder), \
                 mock.patch("subprocess.run", side_effect=fake_tflite):
                path = bm.generate_music(folder, GOOD, 3, 42, log=lambda *_: None)
            self.assertNotEqual(path, bm.music_file(folder, GOOD, 3, 42, "pytorch"))
            with wave.open(str(path), "rb") as clip:
                self.assertEqual((clip.getframerate(), clip.getnchannels()), (RATE, 1))
                self.assertAlmostEqual(clip.getnframes() / RATE, 3, places=2)
            self.assertEqual(list(folder.glob("*.44k.wav")), [])  # temp file removed

    def test_new_seed_makes_new_music(self):
        a = bm.music_file(Path("x"), GOOD, 115, 42)
        self.assertEqual(a, bm.music_file(Path("x"), GOOD, 115, 42))
        self.assertNotEqual(a, bm.music_file(Path("x"), GOOD, 115, 7))

    def test_missing_stable_audio_explains_setup(self):
        with mock.patch.dict("os.environ", {"STABLE_AUDIO_PYTHON": "C:/nowhere/python.exe"}):
            with self.assertRaises(RuntimeError) as error:
                bm.worker_python()
        self.assertIn("git clone", str(error.exception))


if __name__ == "__main__":
    unittest.main()
