import pytest
import json
import yaml
from pathlib import Path
from datasets import Dataset
from unittest.mock import patch, mock_open
from data.advanced_prompt_generator.advanced_prompt_generator import AdvancedPromptGenerator
from data.base_prompt_generator import BasePromptGenerator

@pytest.fixture
def generator_instance() -> AdvancedPromptGenerator:
    return AdvancedPromptGenerator()

# def test_init(generator_instance: AdvancedPromptGenerator):
#     #print(generator_instance.prompt_attributes)
#     print(len(generator_instance.prompt_list))
#     print(generator_instance.prompt_list[0:2])

def test_process_save_multiple_rollouts_to_json(generator_instance: AdvancedPromptGenerator, tmp_path):
    input_path = Path("tests/data/results_multiple_rollouts.jsonl")

    if not input_path.exists():
        pytest.fail(f"Test data file not found at: {input_path}")

    dataset = Dataset.from_json(str(input_path))
    generator_instance.save_responses_to_json(tmp_path, dataset)

def test_process_save_multiple_rollouts_batch_size_1(generator_instance: AdvancedPromptGenerator, tmp_path):
    input_path = Path("tests/data/results_multiple_rollouts.jsonl")

    if not input_path.exists():
        pytest.fail(f"Test data file not found at: {input_path}")

    dataset = Dataset.from_json(str(input_path))
    generator_instance.save_responses_to_json(tmp_path, dataset, batch_size=1)
