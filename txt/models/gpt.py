# encoding: utf-8
from typing import List

import torch
import transformers
from txt.models import Model


class GPT(Model):
    """Wrapper around HuggingFace's implementation of OpenAI's GPT with a
    language model head.
    """

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        decoder: transformers.PreTrainedModel,
    ) -> None:
        self.tokenizer = tokenizer
        self.decoder = decoder

    @classmethod
    def from_pretained(cls, model_name: str) -> "GPT":
        tokenizer = transformers.OpenAIGPTTokenizer.from_pretained(model_name)
        model = transformers.OpenAIGPTLMHeadModel.from_pretained(model_name)
        return cls(tokenizer, model)

    def decode(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        """ Returns the last token logits returned by GPT with a language model
        head.

        Argument
        --------
        input_ids: torch.tensor
            A tensor that contains the encoded prompt.
        """
        output = self.decoder(input_ids)
        return output[0][:, -1, :]

    def ids_from_text(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def text_from_ids(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)
