import unittest

from rag_engine_web import model_chain, stream_answer


class RateLimitError(Exception):
    pass


class NotFoundError(Exception):
    pass


class TestModelChain(unittest.TestCase):
    def test_configured_model_goes_first_then_the_other_free_models(self):
        chain = model_chain("openai/gpt-oss-120b")
        self.assertEqual(chain[0], "openai/gpt-oss-120b")
        self.assertEqual(len(set(chain)), len(chain))
        self.assertIn("qwen/qwen3.8-27b", chain)

    def test_configured_model_already_in_the_list_is_not_retried_twice(self):
        chain = model_chain("qwen/qwen3.8-27b")
        self.assertEqual(chain.count("qwen/qwen3.8-27b"), 1)


class TestStreamAnswer(unittest.TestCase):
    def _factory(self, behaviours):
        def emit(items):
            for item in items:
                if isinstance(item, Exception):
                    raise item
                yield item

        def make_chain(model):
            class Chain:
                def stream(self, inputs):
                    outcome = behaviours[model]
                    if isinstance(outcome, Exception):
                        raise outcome
                    return emit(outcome)
            return Chain()
        return make_chain

    def test_falls_back_to_the_next_model_when_the_first_is_out_of_quota(self):
        make = self._factory({"a": RateLimitError("cạn ngày"), "b": ["Xin ", "chào"]})
        seen = []
        chunks = list(stream_answer(make, ["a", "b"], {}, report=lambda m, fb: seen.append((m, fb))))
        self.assertEqual("".join(chunks), "Xin chào")
        self.assertEqual(seen, [("b", True)])

    def test_reports_the_configured_model_when_it_answers_directly(self):
        make = self._factory({"a": ["Không cần đổi"]})
        seen = []
        list(stream_answer(make, ["a", "b"], {}, report=lambda m, fb: seen.append((m, fb))))
        self.assertEqual(seen, [("a", False)])

    def test_real_failures_are_not_swallowed_by_the_fallback(self):
        make = self._factory({"a": ValueError("prompt sai"), "b": ["không đáng tới"]})
        with self.assertRaises(ValueError):
            list(stream_answer(make, ["a", "b"], {}))

    def test_does_not_restart_after_text_reached_the_screen(self):
        # Đã phun chữ ra rồi thì không retry: làm lại sẽ nhân đôi câu trả lời.
        make = self._factory({"a": ["nửa câu ", RateLimitError("giữa chừng")], "b": ["khác"]})
        stream = stream_answer(make, ["a", "b"], {})
        self.assertEqual(next(stream), "nửa câu ")
        with self.assertRaises(RateLimitError):
            next(stream)

    def test_raises_the_last_quota_error_when_every_model_is_spent(self):
        make = self._factory({"a": RateLimitError("hết 1"), "b": NotFoundError("hết 2")})
        with self.assertRaises(NotFoundError):
            list(stream_answer(make, ["a", "b"], {}))


if __name__ == "__main__":
    unittest.main()
