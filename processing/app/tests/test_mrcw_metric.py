import unittest
from unittest.mock import Mock

import torch

from app.models.metrics.median_rank_correct_word import MRCWMetric
from app.services.tensor_parser import tensor_to_string


class TestMRCWMetric(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.metric = MRCWMetric()

        # Create a mock tokenizer
        self.mock_tokenizer = Mock()
        # Simple tokenization: map each word to its ASCII value - 65 (A=0, B=1, etc.)
        self.mock_tokenizer.encode = Mock(
            side_effect=lambda text, add_special_tokens=False: [
                ord(c.upper()) - 65 for c in text if c.isalpha()
            ]
        )

    def test_simple_mrcw_calculation(self):
        """Test MRCW calculation with a simple example."""
        # Create simple logits where token 0 (A) has highest prob, token 1 (B) second highest, etc.
        vocab_size = 26  # A-Z
        seq_len = 2

        # Position 0: A=5.0, B=4.0, C=3.0, ... Z=0.1 (A should rank 1st)
        # Position 1: B=5.0, A=4.0, C=3.0, ... Z=0.1 (B should rank 1st, A should rank 2nd)
        logits = torch.zeros(seq_len, vocab_size)

        # Position 0: A (token 0) gets highest score
        for i in range(vocab_size):
            logits[0, i] = 5.0 - i * 0.1  # A=5.0, B=4.9, C=4.8, ...

        # Position 1: B (token 1) gets highest score, A gets second highest
        logits[1, 1] = 5.0  # B
        logits[1, 0] = 4.9  # A
        for i in range(2, vocab_size):
            logits[1, i] = 4.8 - (i - 2) * 0.1

        # Convert to string format
        logits_str = tensor_to_string(logits)

        # Test case: reference is "AB"
        # Position 0: looking for A (token 0) - should be rank 1
        # Position 1: looking for B (token 1) - should be rank 1
        # Expected median rank: 1.0
        predictions = ["predicted_text"]
        references = ["AB"]
        logits_list = [logits_str]

        result = self.metric.compute(
            predictions=predictions,
            references=references,
            logits=logits_list,
            tokenizer=self.mock_tokenizer,
        )

        self.assertIn("median_rank_correct_word", result)
        self.assertEqual(result["median_rank_correct_word"], 1.0)

    def test_mrcw_with_worse_ranks(self):
        """Test MRCW calculation where reference tokens have worse ranks."""
        vocab_size = 26
        seq_len = 3

        # Create logits where reference tokens have ranks 2, 3, 1 respectively
        logits = torch.zeros(seq_len, vocab_size)

        # Position 0: A (token 0) should be rank 2 (B > A > C > ...)
        logits[0, 1] = 5.0  # B gets rank 1
        logits[0, 0] = 4.9  # A gets rank 2
        for i in range(2, vocab_size):
            logits[0, i] = 4.8 - (i - 2) * 0.1

        # Position 1: B (token 1) should be rank 3 (C > D > B > ...)
        logits[1, 2] = 5.0  # C gets rank 1
        logits[1, 3] = 4.9  # D gets rank 2
        logits[1, 1] = 4.8  # B gets rank 3
        for i in range(4, vocab_size):
            logits[1, i] = 4.7 - (i - 4) * 0.1
        logits[1, 0] = 4.7 - (vocab_size - 4) * 0.1  # A gets low rank

        # Position 2: C (token 2) should be rank 1
        logits[2, 2] = 5.0  # C gets rank 1
        for i in range(vocab_size):
            if i != 2:
                logits[2, i] = 4.9 - i * 0.1

        logits_str = tensor_to_string(logits)

        # Test case: reference is "ABC"
        # Expected ranks: 2, 3, 1
        # Expected median: 2.0
        predictions = ["predicted_text"]
        references = ["ABC"]
        logits_list = [logits_str]

        result = self.metric.compute(
            predictions=predictions,
            references=references,
            logits=logits_list,
            tokenizer=self.mock_tokenizer,
        )

        self.assertIn("median_rank_correct_word", result)
        self.assertEqual(result["median_rank_correct_word"], 2.0)

    def test_empty_logits(self):
        """Test behavior with empty logits."""
        predictions = ["predicted_text"]
        references = ["ABC"]
        logits_list = [""]  # Empty logits

        result = self.metric.compute(
            predictions=predictions,
            references=references,
            logits=logits_list,
            tokenizer=self.mock_tokenizer,
        )

        self.assertIn("median_rank_correct_word", result)
        self.assertEqual(result["median_rank_correct_word"], float("inf"))

    def test_invalid_inputs(self):
        """Test validation of inputs."""
        with self.assertRaises(ValueError):
            self.metric.compute([], ["ref"], ["logits"])

        with self.assertRaises(ValueError):
            self.metric.compute(["pred"], [], ["logits"])

        with self.assertRaises(ValueError):
            self.metric.compute(["pred"], ["ref"], [])

        with self.assertRaises(ValueError):
            self.metric.compute(["pred1", "pred2"], ["ref1"], ["logits1"])


if __name__ == "__main__":
    unittest.main()
