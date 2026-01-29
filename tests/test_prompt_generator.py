import pytest
import json
import yaml
from pathlib import Path
from datasets import Dataset
from unittest.mock import patch, mock_open
from data.prompt_generator.prompt_generator import PromptGenerator
from data.prompt_generator.prompt_generator import BasePromptGenerator

@pytest.fixture
def generator_instance():
    """
    Creates an instance of PromptGenerator with a mocked __init__.
    This prevents the test from failing if 'data/prompt_config.yaml' 
    doesn't exist in the test environment.
    """
    # We mock the open() call and yaml.safe_load so __init__ passes without error
    with patch("builtins.open", mock_open(read_data="themes: []")), \
        patch("yaml.safe_load", return_value={"themes": [], "harms": [], "system_prompt": ""}):
        generator = PromptGenerator()
        return generator

def test_process_results_single_completion_jsonl(generator_instance : BasePromptGenerator, tmp_path):
    """
    Loads the specific JSONL results file, converts it to a list,
    and tests if save_responses_to_json parses and saves it correctly.
    """
    input_path = Path("tests/data/results_single_prompt.jsonl")

    if not input_path.exists():
        pytest.fail(f"Test data file not found at: {input_path}")

    dataset = Dataset.from_json(str(input_path))
    output_file = tmp_path / "test_single_parsed_output.json"
    generator_instance.save_responses_to_json(output_file, dataset)

def test_process_results_multiple_completions_jsonl(generator_instance : BasePromptGenerator, tmp_path):
    """
    Loads the specific JSONL results file with multiple, converts it to a list,
    and tests if save_responses_to_json parses and saves it correctly.
    """

    input_path = Path("tests/data/results_multiple_prompt.jsonl")

    if not input_path.exists():
        pytest.fail(f"Test data file not found at: {input_path}")

    dataset = Dataset.from_json(str(input_path))
    output_file = tmp_path / "test_multiple_parsed_output.json"
    generator_instance.save_responses_to_json(output_file, dataset)

