import yaml
from abc import ABC, abstractmethod
from pathlib import Path

class PromptGenerator(ABC):
    """interface for prompt generation"""
    @abstractmethod
    def load_prompts(self) -> dict:
        pass


class SinglePromptGenerator(PromptGenerator):
    """a synthetic data prompt data loader for a single user prompt"""

    def __init__(self):
        """uses a random init value to initialize prompt characteristics"""
        with open("data/prompt_config.yaml") as f:
            self._loaded_yaml = yaml.safe_load(f)

        self.system_prompt = self._loaded_yaml.get("system_prompt", "")
        self.common_traits = self._loaded_yaml.get("common_traits", [])

    def format_prompt(self, common_trait: dict)-> str:
        example_str = ', '.join(common_trait["examples"])
        prompt = f"""Common Traits:
            - Trait: {common_trait["name"]}
            Description: {common_trait["description"]}
            Examples: {example_str}
        """

        return prompt

    def load_prompts(self, num_examples) -> dict:
        """
        returns a dict with question, answer keys to be used to load as a huggingface dataset
        """
        max_examples = len(self.common_traits)
        if num_examples > max_examples:
            print(f"Warning: {num_examples} requested examples > {max_examples} max examples.")
        
        dataset_dict = {
            "question": [],
            "answer": []
        }
        for index, common_trait in enumerate(self.common_traits):
            if index >= num_examples:
                break
            
            formatted_prompt = self.format_prompt(common_trait)
            dataset_dict["question"].append(formatted_prompt)
            dataset_dict["answer"].append("")

        return dataset_dict
