import torch
from typing import Dict
from dataclasses import dataclass


class MockSequenceClassification:
    @dataclass
    class Config:
        id2label: Dict[int, str]
    config = Config(id2label={
        0: 'label_1', 1: 'label_2',
        2: 'label_3', 3: 'label_4'})

    def __init__(self, **kwargs):
        self.texts = kwargs['texts']

    @classmethod
    def from_pretrained(cls, model_name, cache_dir):
        return cls

    @property
    def logits(self) -> torch.FloatTensor:
        return torch.FloatTensor([[0.1, 0.2, 0.3, 0.4]] * len(self.texts))
