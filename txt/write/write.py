""" Functions that create the sequence generating algorithms.
"""
from .greedy_search import GreedySearch
from .sampler import Sampler


__all__ = ["with_greedy_search", "with_sampler"]


def with_greedy_search(model, device):
    return GreedySearch(model, device)


def with_sampler(model, device, k=9, p=0.0, temperature=1.0, repetition_penalty=1.0):
    return Sampler(model, device, k, p, temperature, repetition_penalty)
