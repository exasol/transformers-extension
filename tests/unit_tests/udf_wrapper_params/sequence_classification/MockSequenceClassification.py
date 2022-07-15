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

    cnt = 0
    model_based_data_size_at_each_batch = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        # initiated by tests
        if 'model_based_data_size_at_each_batch' in kwargs:
            MockSequenceClassification.cnt = 0
            MockSequenceClassification.model_based_data_size_at_each_batch = \
                kwargs['model_based_data_size_at_each_batch']

    @classmethod
    def from_pretrained(cls, model_name, cache_dir):
        return cls

    @property
    def logits(self) -> torch.FloatTensor:
        MockSequenceClassification.cnt += 1
        model_based_data_size = \
            MockSequenceClassification.model_based_data_size_at_each_batch[
                MockSequenceClassification.cnt - 1]
        return torch.FloatTensor([[0.1, 0.2, 0.3, 0.4]] *
                                 model_based_data_size)
