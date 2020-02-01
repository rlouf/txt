# encoding: utf-8
from abc import ABC, abstractmethod
from typing import List

import torch


class Model(ABC):
    @abstractmethod
    def decode(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        pass

    @abstractmethod
    def ids_from_text(self, text: str) -> List[int]:
        pass

    @abstractmethod
    def text_from_ids(self, token_ids: torch.LongTensor) -> str:
        pass
