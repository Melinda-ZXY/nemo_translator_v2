import unittest

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


if __name__ == "__main__":
    unittest.main()
