import pytest
import json
import yaml
from pathlib import Path
from datasets import Dataset
from unittest.mock import patch, mock_open
from data.improved_prompt_generator.improved_prompt_generator import ImprovedPromptGenerator
from data.base_prompt_generator import BasePromptGenerator

@pytest.fixture
def generator_instance() -> ImprovedPromptGenerator:
    return ImprovedPromptGenerator(input_file="data/improved_prompt_generator/improved_prompt_config.yaml")

def test_read_yaml(generator_instance: ImprovedPromptGenerator):
    # assert len(generator_instance.prompt_tuples) == 240, f"Error: expected len(240), got len({len(generator_instance.prompt_tuples)})"
    print(f"Number of prompts generated: {len(generator_instance.prompt_tuples)}")

def test_random_load_prompts(generator_instance: ImprovedPromptGenerator):
    num_examples = 10
    random_seed = 1111010
    dict = generator_instance.load_prompts(num_examples, random_seed)
    #print(dict)

def test_match_pattern(generator_instance: ImprovedPromptGenerator):
    num_examples = 1
    random_seed = 1111010
    dict = generator_instance.load_prompts(num_examples, random_seed)

    example = dict['question'][0] # first question string
    metadata = generator_instance._match_pattern(example)
    assert metadata.get('theme', None), "Error parsing theme"
    assert metadata.get('harm', None), "Error parsing harm"
    assert metadata.get('condition', None), "Error parsing theme"
    assert metadata.get('style', None), "Error parsing theme"