from pathlib import PurePosixPath

import torch
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class Config:
    id2label: Dict[int, str]


class MockSequenceClassificationResult:

    def __init__(self, batch_logits: torch.FloatTensor):
        self.logits = batch_logits


class MockSequenceClassificationModel:

    def __init__(self, config: Config, logits: List[float]):
        self._logits = logits
        self.config = config
        self.device_name = None

    def __call__(self, first_texts, second_texts=None):
        batch_logits = torch.FloatTensor([self._logits] * len(first_texts))
        return MockSequenceClassificationResult(batch_logits)

    def to(self, device_name: str):
        self.device_name = device_name
        return self


class MockSequenceClassificationFactory:

    def __init__(self, mock_models: Dict[PurePosixPath,
                                         MockSequenceClassificationModel]):
        self.mock_models = mock_models

    def from_pretrained(self, model_name, cache_dir):
        # the cache_dir path already has model_name
        return self.mock_models[cache_dir]
