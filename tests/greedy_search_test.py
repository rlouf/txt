# encoding: utf-8
import itertools
import unittest

import torch

from txt.write import GreedySearch


class StubDecoder(object):
    def __init__(self, logits):
        self.logits = itertools.cycle(logits)

    def decode(self, _):
        return torch.tensor([next(self.logits)])


class GreedySearchTest(unittest.TestCase):
    def test_init(self):
        decoder = StubDecoder([[]])
        GreedySearch(decoder)

        # intialization will default to cpu backend
        writer = GreedySearch.using(decoder)
        self.assertEqual(writer.device.type, "cpu")

        # initialize with invalid backend
        with self.assertRaises(RuntimeError):
            GreedySearch.using(decoder).on("notavalidbackend")

        # full initialization
        writer = GreedySearch.using(decoder).on("cpu")
        self.assertEqual(writer.device.type, "cpu")

    def test_iteration(self):
        logits = [[0.8, 0.1, 0.7, 0.9], [0.9, 0.1, 0.7, 0.8]]
        decoder = StubDecoder(logits)
        writer = GreedySearch.using(decoder).on("cpu")
        tokens = writer.tokens()
        self.assertEqual(next(tokens), 3)
        self.assertEqual(next(tokens), 0)

    def test_generate_ids(self):
        logits = [[0.8, 0.1, 0.7, 0.9]]
        decoder = StubDecoder(logits)
        writer = GreedySearch.using(decoder).on("cpu")

        generated_sequence = writer.generate_ids(num_tokens=10)
        self.assertEqual(len(generated_sequence), 10)
        self.assertEqual(generated_sequence, [3] * 10)

    def test_generate_ids_until(self):
        logits = [[0.8, 0.1, 0.7, 0.9], [0.9, 0.1, 0.7, 0.8]]
        decoder = StubDecoder(logits)
        writer = GreedySearch.using(decoder).on("cpu")

        # No constrain on the minimum length
        generated_sequence = writer.generate_ids_until(0)
        self.assertEqual(len(generated_sequence), 2)
        self.assertEqual(generated_sequence, [3, 0])

        # Non trivial minimum length
        generated_sequence = writer.generate_ids_until(0, min_length=3)
        self.assertEqual(len(generated_sequence), 4)
        self.assertEqual(generated_sequence, [3, 0, 3, 0])

        # Generate until max length
        generated_sequence = writer.generate_ids_until(1, max_length=6)
        self.assertEqual(len(generated_sequence), 6)
        self.assertEqual(generated_sequence, [3, 0, 3, 0, 3, 0])

        # Generate until specific sequence rather than token
        generated_sequence = writer.generate_ids_until([0, 3])
        self.assertEqual(len(generated_sequence), 3)
        self.assertEqual(generated_sequence, [3, 0, 3])

    def test_generate_ids_with_prompt(self):
        logits = [[0.8, 0.1, 0.7, 0.9]]
        decoder = StubDecoder(logits)
        writer = GreedySearch.using(decoder).on("cpu")

        prompt = [1, 234]
        generated_sequence = writer.prompt_ids(prompt).generate_ids(num_tokens=10)
        self.assertEqual(len(generated_sequence), 10)
        self.assertEqual(generated_sequence, [3] * 10)


if __name__ == "__main__":
    unittest.main()
