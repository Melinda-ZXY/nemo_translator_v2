import unittest

from orthography_v2 import render_orthography_html
from translator_v2_core import translate


class WithClauseParsingTests(unittest.TestCase):
    def test_modal_with_clause_keeps_subject_and_companion(self):
        result = translate("我想和你玩")

        self.assertEqual(result["nemo"], "tika mo la na takukt")
        self.assertEqual(result["parsed"]["subject"]["text"], "我")
        self.assertEqual(result["parsed"]["with"]["text"], "你")

    def test_plain_verb_uses_entity_before_with_as_subject(self):
        result = translate("我和你玩")

        self.assertEqual(result["nemo"], "takukt mo la na")
        self.assertEqual(result["parsed"]["subject"]["text"], "我")
        self.assertEqual(result["parsed"]["with"]["text"], "你")

    def test_with_subject_phrase_works_before_modal(self):
        self.assertEqual(translate("我和你想玩")["nemo"], "tika mo la na takukt")

    def test_with_subject_phrase_works_with_state(self):
        self.assertEqual(translate("我和你很开心")["nemo"], "tnuka to mo la na")

    def test_bare_with_phrase_preserves_source_order(self):
        self.assertEqual(translate("我和你")["nemo"], "mo la na")

    def test_with_phrase_can_contain_possessive_noun_phrase(self):
        self.assertEqual(translate("我和你的家")["nemo"], "mo la na tu djano")
        self.assertEqual(
            translate("我想和你的家玩")["nemo"],
            "tika mo la na tu djano takukt",
        )

    def test_standalone_possessive_remains_unchanged(self):
        self.assertEqual(translate("我的家")["nemo"], "mo tu djano")


class NewLexiconTests(unittest.TestCase):
    def test_new_chinese_terms_translate_to_nemo(self):
        expected = {
            "那里": "daa",
            "这里": "dee",
            "得意": "doka",
            "得意的": "doka",
            "上升": "ka",
            "下降": "ki",
            "没事": "padu",
            "没事的": "padu",
        }

        for chinese, nemo in expected.items():
            with self.subTest(chinese=chinese):
                self.assertEqual(translate(chinese)["nemo"], nemo)

    def test_new_terms_work_in_sentences(self):
        self.assertEqual(translate("我很得意")["nemo"], "doka to mo")
        self.assertEqual(translate("电池上升")["nemo"], "ka plana")
        self.assertEqual(translate("电池下降")["nemo"], "ki plana")
        self.assertEqual(translate("这里没事的")["nemo"], "padu dee")

    def test_new_terms_have_renderable_glyphs(self):
        for token in ("daa", "dee", "doka", "ka", "ki", "padu"):
            with self.subTest(token=token):
                html = render_orthography_html(token)
                self.assertIn(f'alt="{token}"', html)
                self.assertNotIn('<span class="v2-glyph-placeholder"', html)


if __name__ == "__main__":
    unittest.main()
