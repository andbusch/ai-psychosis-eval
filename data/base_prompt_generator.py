from abc import ABC, abstractmethod

class BasePromptGenerator(ABC):
    """interface for prompt generation"""
    @abstractmethod
    def parse_response():
        pass

    @abstractmethod
    def load_prompts(self) -> dict:
        pass

    @abstractmethod
    def save_responses_to_json(self, filename: str, responses: dict):
        pass