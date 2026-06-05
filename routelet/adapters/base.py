from abc import ABC, abstractmethod

from routelet.core.request import NormalizedRequest, NormalizedResponse


class AdapterBase(ABC):
    @abstractmethod
    def call(self, model: str, request: NormalizedRequest) -> NormalizedResponse: ...
