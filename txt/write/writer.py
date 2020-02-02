# encoding: utf-8
from abc import ABC, abstractmethod
from typing import List, Union

import torch

from txt.models import Model


class Writer(ABC):
    """Defines a common interface for all sequence generation algorithms.

    Examples
    --------

    Any class that inherits from `Writer` can be instantiated directly by
    passing a model instance.

    >>> writer = Writer(model)

    We all provide a `using` classmethod for API sugar magic.

    >>> writer = Writer.using(model)

    Writers' parameters are loaded on CPU by default. You can load them on any
    device using:

    >>> device_name = "cuda"
    >>> writer = Writer.using(model).on(device_name)

    where `device_name` can be any device name that is understood by PyTorch.

    One can generate a fixed number of token ids using the `generate_ids`
    method, or directly text with the `generate` method. Note that the number
    of words corresponding to a sequence of tokens is at most equal to the
    number of tokens and is not easily predictable.

    >>> writer = Writer.using(model)
    >>> generated_ids = writer.generate_ids(10)
    >>> generated_text = writer.generate(10)

    It is possible to prompt the model with text or token ids.

    >>> writer = Writer.using(model)
    >>> generated_ids = writer.prompt_ids([1, 234]).generate_ids(10)
    >>> generated_text = writer.prompt("how are you?").generate(10)

    We also provide a method to generate text or token ids until a substring
    or sequence of tokens has been generated.

    >>> writer = Writer.using(model)
    >>> generated_ids = writer.generate_ids_until(0)
    >>> generated_text = writer.generate_until("the end.")

    In order to avoid sequences that are too short or too long one can pass
    `min_length` and `max_length` arguments. These are expressed in number of tokens.
    See the documentation of each writer for the default values.

    >>> writer = Writer.using(model)
    >>> generated_ids = writer.generate_ids_until(0, min_length=10, max_length=100)
    >>> generated_text = writer.generate_until("the end.", min_length=10, max_length=100)

    Attribute
    ---------
    model: Model
        The model used to generate the probabilities for the next token.
    """

    def __init__(self, model: Model) -> None:
        self.model = model
        self.device = torch.device("cpu")
        self.past = torch.tensor([[]], dtype=torch.long, device=self.device)

    @classmethod
    def using(cls, model: Model) -> "Writer":
        """Use the specified model to generate sequences.

        The model passed as an argument must implement the `decode` method.
        #(TODO) Create a `Model` interface that ensures this.

        Attribute
        ---------
        model: Model
            The model to be used with the sequence generation algorithm.

        Returns
        -------
        Writer
            The current instance of the Writer with the model attached.
        """
        return cls(model)

    def on(self, device_name: str) -> "Writer":
        """Initiate the writer's parameters on the specified device.

        Attribute
        ---------
        device_name: str
            Any device name understood by PyTorch.

        Returns
        -------
        Writer
            The current instance of the writer.
        """
        try:
            self.device = torch.device(device_name)
            self.past = self.past.to(self.device)
        except RuntimeError as e:
            raise e
        return self

    def prompt(self, prompt: str) -> "Writer":
        """Initialize the writer with a text prompt.

        The conversion from text to token ids is managed by the `ids_from_text`
        method of the `Model` class. We follow the principle of least surprise;
        by default, this method does not add any special token to the converted
        text.

        If you want to add special tokens, you need to add them explicitly in
        the input string.

        Examples
        --------

        The following won't add any special token to the string.

        >>> writer = Writer.using(model).prompt("hello, how are you?")

        If you want to add, for instance, the "[CLS]" token you need to add it
        explicitly:

        >>> writer = Writer.using(model).prompt("[CLS] hello, how are you?")

        Attribute
        ---------
        prompt: str
            The text to prompt the writer with.

        Returns
        -------
        Writer
            The current Writer instance.
        """
        prompt_ids = self.model.ids_from_text(prompt)
        self.prompt_ids(prompt_ids)
        return self

    def prompt_ids(self, prompt_ids: List[int]) -> "Writer":
        """Initialize the writer with a prompt.

        Attribute
        ---------
        prompt_ids: List[int]
            The list of tokens to prompt the writer with.

        Returns
        -------
        Writer
            The current Writer instance.
        """
        self.past = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        return self

    def generate(self, num_tokens: int) -> str:
        """Generate a fixed number of token and convert them to text.

        Note that this method will generate `num_tokens` tokens and then
        convert them to text. Since there is no simple correspondance between
        the number of tokens and the number of words, understand this argument as
        an upper bound on the number of words that will be returned.

        Attribute
        ---------
        num_tokens: int
            The number of tokens to generate.

        Returns
        -------
        str
            The generated tokens converted to text.
        """
        generated_ids = self.generate_ids(num_tokens)
        return self.model.text_from_ids(generated_ids)

    @abstractmethod
    def generate_ids(self, num_tokens: int) -> List[int]:
        """Generate a fixed number of token ids.
        """
        pass

    def generate_until(
        self, end_substring: str, max_length: int, min_length: int
    ) -> str:
        """Generate text until a given substring has been generated.

        Attributes
        ----------
        end_substring: str
            The substring we want our generated text to end with.
        max_length: int
            The maximum length of the generated sequence. The generation will
            stop when the length of the sequence reaches `max_length`, even if
            the substring has not been generated.  Required to avoid infinite
            loops.
        min_length: int
            The minimum length of the generated sequence, in number of tokens. If the substring
            is observed but the sequence is shorter than `min_length` the generation will continue.
        """
        end_tokens = self.model.ids_from_text(end_substring)
        generated_ids = self.generate_ids_until(end_tokens, max_length, min_length)
        return self.model.text_from_ids(generated_ids)

    @abstractmethod
    def generate_ids_until(
        self, end_tokens: Union[int, List[int]], max_length: int, min_length: int
    ) -> List[int]:
        """Generate token ids until a token or a sequence of token has been generated.
        """
        pass
