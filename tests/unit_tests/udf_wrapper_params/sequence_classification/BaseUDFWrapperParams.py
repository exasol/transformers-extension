from abc import ABC, abstractmethod


class BaseUDFWrapperParams(ABC):

    @abstractmethod
    def udf_wrapper():
        raise NotImplementedError

    @abstractmethod
    def _single_text(self):
        raise NotImplementedError

    @abstractmethod
    def _text_pair(self):
        raise NotImplementedError

    @property
    def single_text(self):
        return self._single_text()

    @property
    def test_pair(self):
        return self._text_pair()


