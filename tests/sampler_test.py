# encoding: utf-8
import itertools
import unittest

import numpy as np
import torch

from txt.write import Sampler


class StubDecoder(object):
    def __init__(self, logits):
        self.logits = itertools.cycle(logits)

    def decode(self, _):
        return torch.tensor([next(self.logits)])


class SamplerTest(unittest.TestCase):
    def test_nucleus_sampling(self):
        inf = -float("Inf")
        test_cases = (
            {
                "p": 0,
                "logits": torch.tensor([0.3, 0.1, 0.2]),
                "expected": torch.tensor([0.3, 0.1, 0.2]),
            },
            {
                "p": 0.01,
                "logits": torch.tensor([0.3, 0.1, 0.2]),
                "expected": torch.tensor([0.3, inf, inf]),
            },
            {
                "p": 1,
                "logits": torch.tensor([0.3, 0.1, 0.2]),
                "expected": torch.tensor([0.3, 0.1, 0.2]),
            },
            {
                "p": 0.2,
                "logits": torch.tensor([0.7, 0.1, 0.2]),
                "expected": torch.tensor([0.7, inf, inf]),
            },
            {
                "p": 0.71,
                "logits": torch.tensor([0.7, 0.1, 0.2]),
                "expected": torch.tensor([0.7, inf, 0.2]),
            },
            {
                "p": 0.71,
                "logits": torch.tensor([0.1, 0.7, 0.2]),
                "expected": torch.tensor([inf, 0.7, 0.2]),
            },
            {
                "p": 0.71,
                "logits": torch.tensor([0.7, 0.2, 0.1]),
                "expected": torch.tensor([0.7, 0.2, inf]),
            },
            {
                "p": 0.91,
                "logits": torch.tensor([0.7, 0.1, 0.2]),
                "expected": torch.tensor([0.7, 0.1, 0.2]),
            },
        )
        for case in test_cases:
            config = {
                "temperature": 1.0,
                "k": 0,
                "p": case["p"],
                "repetition_penalty": 1.0,
            }
            decoder = StubDecoder(case["logits"])
            sampler = Sampler(decoder, torch.device("cpu"), **config)
            filtered_logits = sampler.apply_nucleus_filter(case["logits"])
            np.testing.assert_array_equal(
                case["expected"].numpy(), filtered_logits.numpy()
            )

    def test_top_k_filter(self):
        inf = -float("Inf")
        test_cases = (
            {
                "k": 0,
                "logits": torch.tensor([0.7, 0.1, 0.2]),
                "expected": torch.tensor([0.7, 0.1, 0.2]),
            },
            {
                "k": 1,
                "logits": torch.tensor([0.7, 0.1, 0.2]),
                "expected": torch.tensor([0.7, inf, inf]),
            },
            {
                "k": 2,
                "logits": torch.tensor([0.7, 0.1, 0.2]),
                "expected": torch.tensor([0.7, inf, 0.2]),
            },
            {
                "k": 3,
                "logits": torch.tensor([0.7, 0.1, 0.2]),
                "expected": torch.tensor([0.7, 0.1, 0.2]),
            },
        )
        for case in test_cases:
            config = {
                "temperature": 1.0,
                "k": case["k"],
                "p": 0,
                "repetition_penalty": 1.0,
            }
            decoder = StubDecoder(case["logits"])
            sampler = Sampler(decoder, torch.device("cpu"), **config)
            filtered_logits = sampler.apply_top_k_filter(case["logits"])
            np.testing.assert_array_equal(
                case["expected"].numpy(), filtered_logits.numpy()
            )

    def test_k_larger_than_vocab(self):
        case = {"k": 10, "vocab_size": 5}
        config = {
            "temperature": 1.0,
            "k": case["k"],
            "p": 0,
            "repetition_penalty": 1.0,
        }
        decoder = StubDecoder([1, 2, 3])
        sampler = Sampler(decoder, torch.device("cpu"), **config)
        next_token_logits = torch.rand(case["vocab_size"]).unsqueeze(0)
        with self.assertWarns(UserWarning):
            _ = sampler.apply_top_k_filter(next_token_logits)

    def test_k_out_of_bounds(self):
        case = {"k": -1, "description": "k negative"}
        config = {
            "temperature": 1.0,
            "k": case["k"],
            "p": 0,
            "repetition_penalty": 1.0,
        }
        decoder = StubDecoder([])
        with self.assertRaises(ValueError):
            _ = Sampler(decoder, torch.device("cpu"), **config)

    def test_p_out_of_bounds(self):
        cases = [
            {"p": -0.1, "description": "p negative"},
            {"p": 1.1, "description": "p greater than 1"},
        ]
        for case in cases:
            config = {
                "temperature": 1.0,
                "k": 0,
                "p": case["p"],
                "repetition_penalty": 1.0,
            }
            decoder = StubDecoder([])
            with self.assertRaises(ValueError):
                _ = Sampler(decoder, torch.device("cpu"), **config)

    def test_zero_temperature(self):
        config = {
            "temperature": 0,
            "k": 0,
            "p": 0,
            "repetition_penalty": 1.0,
        }
        decoder = StubDecoder([])
        with self.assertRaises(ZeroDivisionError):
            _ = Sampler(decoder, torch.device("cpu"), **config)

    def test_negative_temperature(self):
        config = {
            "temperature": -1,
            "k": 0,
            "p": 0,
            "repetition_penalty": 1.0,
        }
        decoder = StubDecoder([])
        with self.assertWarns(UserWarning):
            _ = Sampler(decoder, torch.device("cpu"), **config)

    def test_zero_repetition_penalty(self):
        config = {
            "temperature": 1.0,
            "k": 0,
            "p": 0,
            "repetition_penalty": 0.0,
        }
        decoder = StubDecoder([])
        with self.assertRaises(ZeroDivisionError):
            _ = Sampler(decoder, torch.device("cpu"), **config)

    def test_negative_repetition_penalty(self):
        config = {
            "temperature": 1,
            "k": 0,
            "p": 0,
            "repetition_penalty": -1.0,
        }
        decoder = StubDecoder([])
        with self.assertWarns(UserWarning):
            _ = Sampler(decoder, torch.device("cpu"), **config)

    def test_generate(self):
        logits = [[0.8, 0.1, 0.7, 0.9]]
        decoder = StubDecoder(logits)
        writer = Sampler(decoder, torch.device("cpu"))
        generated_sequence = writer.generate_ids(num_tokens=10)
        self.assertEqual(len(generated_sequence), 10)


if __name__ == "__main__":
    unittest.main()
