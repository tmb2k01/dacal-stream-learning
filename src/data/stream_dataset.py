from abc import ABC, abstractmethod


class BaseStreamDataset(ABC):
    @abstractmethod
    def __iter__(self):
        pass
