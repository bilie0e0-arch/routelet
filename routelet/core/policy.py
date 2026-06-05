from abc import ABC, abstractmethod

from routelet.core.request import NormalizedRequest


class PolicyBase(ABC):
    @abstractmethod
    def select_model(self, request: NormalizedRequest) -> str:
        """Return the model snapshot ID to use for this request."""
        ...
