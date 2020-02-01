# encoding: utf-8
import itertools as it
from typing import List, Generator, Union

import torch

from .writer import Writer
from txt.models import Model


__all__ = ["GreedySearch"]


class GreedySearch(Writer):
    """Generate tokens with greedy search.

    Greedy search is an auto-regressive method that consists in choosing at
    each step the token that has the highest probability given the previous
    tokens.

    Attribute
    ---------
    model: Model
        The model that will be used to generate log-probabilities for the next
        token.

    Examples
    --------

    The greedy search can be instantiated on CPU with

    >>> greedy = GreedySearch.using(model)

    Or on GPU with

    >>> greedy = GreedySearch.using(model).on("cuda")

    You can use the `generate` method to quickly generate text. Note that you
    will need to concatenate the results to the prompt if you want to keep the
    later

    >>> prompt = "This was the best of times"
    >>> greedy = GreedySearch.using(model)
    >>> generated_text = greedy.prompt(prompt).generate(10)
    >>> text = prompt + generated_text

    Use the `generate_until` method to generate text until a specified string
    is generated.

    >>> prompt = "This was the best of times"
    >>> end_token = 1
    >>> greedy = GreedySearch.using(model)
    >>> generated_text = greedy.prompt(prompt).generate_until(end_token)

    The generation logic of `GreedySearch` is managed internally by a
    `tokens()` generator. This generator yields token ids one after another,
    updating the past at each step. You can call the generator directly. For
    instance:

    >>> greedy = GreedySearch.using(model)
    >>> tokens = greedy.tokens()
    >>> for token_id in tokens:
    ...     print(token_id)
    ...     if token_id == 5:
    ...         break
    """

    def __init__(self, model: Model) -> None:
        super(GreedySearch, self).__init__(model)

    def tokens(self) -> Generator[int, None, None]:
        """Generate tokens one at a time.

        (TODO) Handle sequence that grow longer than the model's input size.

        Yields
        ------
        int:
            The next generated token.
        """
        past = self.past
        while True:
            next_token_logits = self.model.decode(past)
            past, next_token = self.pick_next_token(past, next_token_logits)
            yield next_token

    def pick_next_token(self, past: torch.tensor, logits: torch.tensor):
        """Pick the next token and update the state.
        """
        next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
        past = torch.cat((past, next_token), dim=1)
        return past, next_token.squeeze(-1).squeeze(-1).item()

    def generate_ids(self, num_tokens: int) -> List[int]:
        """Generate a sequence of tokens ids with a fixed length.

        Attribute
        ---------
        num_tokens: int
            The number of tokens to generate (in addition to the prompt
            if one is provided).

        Returns
        -------
        List[int]
            A list that contains the generated tokens.
        """
        tokens = self.tokens()
        return list(it.islice(tokens, num_tokens))

    def generate_ids_until(
        self,
        end_tokens: Union[int, List[int]],
        max_length: int = 100,
        min_length: int = 1,
    ) -> List[int]:
        """Generate a sequence until a token or a list of tokens is generated.

        Attributes
        ----------
        end_tokens: int or List[int]
            The token or sequence of tokens that stops the process when generated.
        max_length: int, optional
            The maximum length of the generated sequence. The process will stop
            at `max_length` even if the `end_tokens` has not been generated.
            Default to 100 to prevents infinite loops.
        min_length: int, optional
            The minimum length of the generated sequence. If `end_tokens` is
            generated but the sequence is shorter than `min_length` the
            generation continues. Defaults to 1, which is equivalent to no
            minimum length requirement.

        Returns
        -------
        List[int]
            A list that contains the generated tokens. It includes the
            `end_tokens` if the generation stopped before `max_length`.
        """
        sequence = []

        if isinstance(end_tokens, int):
            end_tokens = [end_tokens]
        last_token = end_tokens[-1]

        tokens = self.tokens()
        for token in tokens:
            sequence.append(token)
            if token == last_token:
                if len(sequence) < min_length:
                    continue
                if sequence[-len(end_tokens) :] == end_tokens:
                    break
            if len(sequence) == max_length:
                break

        return sequence
