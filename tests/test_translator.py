"""Offline tests for the translator. They need no API key.

Run with:  python -m unittest discover tests
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import translator  # noqa: E402
from script_writer import Script  # noqa: E402

SCRIPT = Script.model_validate(
    {
        "title": "Test",
        "language": "en-IN",
        "characters": [
            {"name": "Meera", "gender": "female", "description": "young"},
            {"name": "Kabir", "gender": "male", "description": "brother"},
        ],
        "lines": [
            {"speaker": "Narrator", "text": "Rain fell.", "emotion": "neutral"},
            {"speaker": "Meera", "text": "I am coming.", "emotion": "fearful"},
            {"speaker": "Kabir", "text": "Stop!", "emotion": "angry"},
        ],
    }
)

HINDI = ["बारिश हो रही थी।", "मैं आ रही हूँ।", "रुको!"]


def reply(texts, title="परीक्षा", first_id=1):
    lines = [{"id": first_id + i, "text": text} for i, text in enumerate(texts)]
    return json.dumps({"title": title, "lines": lines}, ensure_ascii=False)


class NativeScriptShareTests(unittest.TestCase):
    def test_pure_hindi_is_fully_native(self):
        self.assertEqual(translator.native_script_share("मैं आ रही हूँ।", "hi-IN"), 1.0)

    def test_hinglish_scores_low(self):
        share = translator.native_script_share("एक damp kurta पहने हुए old man", "hi-IN")
        self.assertLess(share, 0.6)

    def test_digits_and_punctuation_are_ignored(self):
        self.assertEqual(translator.native_script_share("11:40!", "hi-IN"), 1.0)

    def test_every_language_has_a_script_range(self):
        self.assertEqual(set(translator.SCRIPT_RANGES), set(translator.LANGUAGES))


class CheckTranslationTests(unittest.TestCase):
    def test_accepts_good_translation(self):
        result = translator.check_translation(reply(HINDI), [1, 2, 3], "hi-IN")
        self.assertEqual([line.text for line in result.lines], HINDI)

    def test_rejects_missing_line(self):
        with self.assertRaisesRegex(ValueError, "exactly one entry"):
            translator.check_translation(reply(HINDI[:2]), [1, 2, 3], "hi-IN")

    def test_rejects_untranslated_line(self):
        with self.assertRaisesRegex(ValueError, "line 2"):
            translator.check_translation(
                reply([HINDI[0], "I am coming.", HINDI[2]]), [1, 2, 3], "hi-IN"
            )

    def test_rejects_wrong_script(self):
        with self.assertRaisesRegex(ValueError, "Tamil script"):
            translator.check_translation(reply(HINDI), [1, 2, 3], "ta-IN")

    def test_strips_speaker_label(self):
        result = translator.check_translation(
            reply(["नैरेटर: बारिश हो रही थी।", "मीरा: मैं आ रही हूँ।", "रुको!"]), [1, 2, 3], "hi-IN"
        )
        self.assertEqual([line.text for line in result.lines], HINDI)

    def test_keeps_clock_times(self):
        text = "चर्चगेट से 11:40 की लोकल खाली थी।"
        result = translator.check_translation(reply([text]), [1], "hi-IN")
        self.assertEqual(result.lines[0].text, text)


class TranslateScriptTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(translator.sarvam_client, "chat", return_value=reply(HINDI))
        self.chat = patcher.start()
        self.addCleanup(patcher.stop)

    def test_translates_text_and_keeps_everything_else(self):
        result = translator.translate_script(SCRIPT, "hi-IN")
        self.assertEqual(result.language, "hi-IN")
        self.assertEqual(result.title, "परीक्षा")
        self.assertEqual(result.characters, SCRIPT.characters)
        self.assertEqual([line.text for line in result.lines], HINDI)
        self.assertEqual(
            [(line.speaker, line.emotion) for line in result.lines],
            [(line.speaker, line.emotion) for line in SCRIPT.lines],
        )

    def test_prompt_tells_the_model_who_is_speaking(self):
        translator.translate_script(SCRIPT, "hi-IN")
        user_message = self.chat.call_args.args[0][1]["content"]
        self.assertIn("Meera (female)", user_message)
        self.assertIn('{"id": 2, "speaker": "Meera", "text": "I am coming."}', user_message)

    def test_retries_with_the_error_when_reply_is_invalid(self):
        self.chat.side_effect = [reply(HINDI[:2]), reply(HINDI)]
        with mock.patch("builtins.print"):
            result = translator.translate_script(SCRIPT, "hi-IN")
        self.assertEqual(len(result.lines), 3)
        retry_messages = self.chat.call_args_list[1].args[0]
        self.assertIn("exactly one entry", retry_messages[-1]["content"])

    def test_long_scripts_are_split_into_chunks(self):
        self.chat.side_effect = [reply(HINDI[:2]), reply(HINDI[2:], first_id=3)]
        with mock.patch.object(translator, "LINES_PER_REQUEST", 2):
            result = translator.translate_script(SCRIPT, "hi-IN")
        self.assertEqual([line.text for line in result.lines], HINDI)
        self.assertEqual(self.chat.call_count, 2)

    def test_same_language_returns_script_without_calling_sarvam(self):
        self.assertIs(translator.translate_script(SCRIPT, "en-IN"), SCRIPT)
        self.chat.assert_not_called()

    def test_rejects_unsupported_language(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            translator.translate_script(SCRIPT, "fr-FR")

    def test_second_run_uses_the_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            cache_file = Path(folder) / "hi-IN" / "translation_cache.json"
            first = translator.translate_script(SCRIPT, "hi-IN", cache_file)
            second = translator.translate_script(SCRIPT, "hi-IN", cache_file)
            self.assertEqual(self.chat.call_count, 1)
            self.assertEqual(first, second)

    def test_cache_is_ignored_when_the_source_script_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            cache_file = Path(folder) / "translation_cache.json"
            translator.translate_script(SCRIPT, "hi-IN", cache_file)
            changed = SCRIPT.model_copy(update={"title": "A new title"})
            translator.translate_script(changed, "hi-IN", cache_file)
            self.assertEqual(self.chat.call_count, 2)


if __name__ == "__main__":
    unittest.main()
