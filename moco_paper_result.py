from abc import ABC, abstractmethod

class MocoPaperResult(ABC):
    @abstractmethod
    def generate_results(self):
        pass

    @abstractmethod
    def report_results(self):
        pass
