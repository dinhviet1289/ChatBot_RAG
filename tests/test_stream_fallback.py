import unittest

from rag_engine_web import model_chain, parse_keys, rotation_pairs, stream_answer


class RateLimitError(Exception):
    pass


class NotFoundError(Exception):
    pass


class TestParseKeys(unittest.TestCase):
    def test_primary_key_alone_is_the_whole_rotation(self):
        self.assertEqual(parse_keys("k1", []), ["k1"])

    def test_toml_array_and_comma_string_both_work_and_dupes_are_dropped(self):
        self.assertEqual(parse_keys("k1", ["k2", "k1", "k3"]), ["k1", "k2", "k3"])
        self.assertEqual(parse_keys("k1", "k2, k1 ,k3"), ["k1", "k2", "k3"])

    def test_blank_and_missing_keys_are_ignored(self):
        self.assertEqual(parse_keys("", ["k2", "", "  "]), ["k2"])
        self.assertEqual(parse_keys(None, None), [])


class TestRotationPairs(unittest.TestCase):
    def test_every_model_of_the_first_key_is_tried_before_the_next_key(self):
        pairs = rotation_pairs(["k1", "k2"], ["m1", "m2"])
        self.assertEqual(pairs, [("m1", "k1"), ("m2", "k1"), ("m1", "k2"), ("m2", "k2")])

    def test_duplicate_keys_do_not_get_a_second_turn(self):
        pairs = rotation_pairs(["k1", "k1", "k2"], ["m1"])
        self.assertEqual(pairs, [("m1", "k1"), ("m1", "k2")])

    def test_single_key_rotation_is_just_the_model_chain(self):
        self.assertEqual(rotation_pairs(["k1"], ["m1", "m2"]), [("m1", "k1"), ("m2", "k1")])

    def test_model_chain_puts_the_configured_model_first_without_repeating_it(self):
        chain = model_chain("openai/gpt-oss-120b")
        self.assertEqual(chain[0], "openai/gpt-oss-120b")
        self.assertEqual(len(set(chain)), len(chain))
        self.assertIn("qwen/qwen3.8-27b", chain)
        self.assertEqual(model_chain("qwen/qwen3.8-27b").count("qwen/qwen3.8-27b"), 1)


class TestStreamAnswer(unittest.TestCase):
    def _factory(self, behaviours):
        def emit(items):
            for item in items:
                if isinstance(item, Exception):
                    raise item
                yield item

        def make_chain(model, key):
            class Chain:
                def stream(self, inputs):
                    outcome = behaviours[(model, key)]
                    if isinstance(outcome, Exception):
                        raise outcome
                    return emit(outcome)
            return Chain()
        return make_chain

    def test_falls_back_to_the_next_model_when_the_first_is_out_of_quota(self):
        make = self._factory({("a", "k1"): RateLimitError("cạn ngày"), ("b", "k1"): ["Xin ", "chào"]})
        seen = []
        chunks = list(stream_answer(make, [("a", "k1"), ("b", "k1")], {},
                                    report=lambda m, k, fb: seen.append((m, k, fb))))
        self.assertEqual("".join(chunks), "Xin chào")
        self.assertEqual(seen, [("b", "k1", True)])

    def test_moves_to_the_next_key_once_every_model_of_the_first_key_is_spent(self):
        make = self._factory({
            ("a", "k1"): RateLimitError("hết ngày"),
            ("b", "k1"): RateLimitError("hết ngày"),
            ("a", "k2"): ["key 2 ", "trả lời"],
        })
        seen = []
        pairs = rotation_pairs(["k1", "k2"], ["a", "b"])
        chunks = list(stream_answer(make, pairs, {}, report=lambda m, k, fb: seen.append((m, k, fb))))
        self.assertEqual("".join(chunks), "key 2 trả lời")
        self.assertEqual(seen, [("a", "k2", True)])

    def test_reports_the_configured_pair_when_it_answers_directly(self):
        make = self._factory({("a", "k1"): ["Không cần đổi"]})
        seen = []
        list(stream_answer(make, [("a", "k1")], {}, report=lambda m, k, fb: seen.append((m, k, fb))))
        self.assertEqual(seen, [("a", "k1", False)])

    def test_real_failures_are_not_swallowed_by_the_fallback(self):
        make = self._factory({("a", "k1"): ValueError("prompt sai"), ("b", "k1"): ["không đáng tới"]})
        with self.assertRaises(ValueError):
            list(stream_answer(make, [("a", "k1"), ("b", "k1")], {}))

    def test_does_not_restart_after_text_reached_the_screen(self):
        # Đã phun chữ ra rồi thì không retry: làm lại sẽ nhân đôi câu trả lời.
        make = self._factory({("a", "k1"): ["nửa câu ", RateLimitError("giữa chừng")],
                              ("b", "k1"): ["khác"]})
        stream = stream_answer(make, [("a", "k1"), ("b", "k1")], {})
        self.assertEqual(next(stream), "nửa câu ")
        with self.assertRaises(RateLimitError):
            next(stream)

    def test_raises_the_last_quota_error_when_every_pair_is_spent(self):
        make = self._factory({("a", "k1"): RateLimitError("hết 1"), ("a", "k2"): NotFoundError("hết 2")})
        with self.assertRaises(NotFoundError):
            list(stream_answer(make, [("a", "k1"), ("a", "k2")], {}))


if __name__ == "__main__":
    unittest.main()
