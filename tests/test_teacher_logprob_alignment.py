"""CPU checks for SGLang's clipped last-vocabulary-ID logprob label."""

import copy
import unittest

from slime.rollout.lightning_opd_blocks import teacher_response_logprobs


def response(labels, values=None):
    if values is None:
        values = [-0.25] * len(labels)
    return {"meta_info": {"input_token_logprobs": [[None, 100, ""]] + [
        [value, token, ""] for value, token in zip(values, labels, strict=True)
    ]}}


class TeacherLogprobAlignmentTests(unittest.TestCase):
    def test_clipped_boundary_label_preserves_original_tokens_and_scores(self):
        tokens = [151934, 151935, 0, 151935]
        values = [-0.1, -12.5, -0.3, -7.25]
        output = response([151934, 0, 0, 0], values)
        previous = copy.deepcopy(output)
        self.assertEqual(teacher_response_logprobs(output, tokens, 151936), values)
        self.assertEqual(tokens, [151934, 151935, 0, 151935])
        self.assertEqual(output, previous)
        # A backend with the corrected boundary also works unchanged.
        self.assertEqual(teacher_response_logprobs(response(tokens, values), tokens, 151936), values)
        self.assertEqual(teacher_response_logprobs(response([0]), [127], 128), [-0.25])

    def test_other_token_mismatches_and_feature_teacher_still_fail(self):
        for labels, tokens, vocabulary in (
            ([0], [151934], 151936),
            ([1], [151935], 151936),
            ([151935], [0], 151936),
            ([0, 151934], [151935, 100], 151936),
            ([0], [151935], None),
        ):
            with self.subTest(labels=labels, tokens=tokens, vocabulary=vocabulary):
                with self.assertRaisesRegex(ValueError, "misaligned"):
                    teacher_response_logprobs(response(labels), tokens, vocabulary)

    def test_out_of_vocabulary_tokens_and_missing_logprobs_still_fail(self):
        for token in (-1, 151936):
            with self.subTest(token=token), self.assertRaisesRegex(ValueError, "outside"):
                teacher_response_logprobs(response([0]), [token], 151936)
        for labels, tokens in (([], [151935]), ([0], [151935, 100]), ([], [])):
            with self.subTest(labels=labels), self.assertRaisesRegex(ValueError, "length mismatch"):
                teacher_response_logprobs(response(labels), tokens, 151936)

    def test_nonfinite_boundary_logprobs_still_fail(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "non-finite"):
                teacher_response_logprobs(response([0], [value]), [151935], 151936)


if __name__ == "__main__":
    unittest.main()
