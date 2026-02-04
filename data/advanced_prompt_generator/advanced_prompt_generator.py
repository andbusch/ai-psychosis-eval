import os
from typing import Optional
import yaml
from itertools import product, batched
from datasets import Dataset
import re
import json
import random
from pathlib import Path

from ..base_prompt_generator import BasePromptGenerator

class AdvancedPromptGenerator(BasePromptGenerator):
    """
    AdvancedPromptGenerator improves on previous iterations in the following ways:

    - using json formatted prompts to prevent the need for regex matching, and allow much more flexible prompt generation
    - added style flag to test case metadata
    - added new parameter: jailbreak method - with potential values:
        - normal - no additional jailbreak method applied
        - fictional - the user attempts to present themselves as a fictional character

    """

    def __init__(self, 
                 input_file = "data/advanced_prompt_generator/advanced_prompt_config.yaml",
                 random_categories : Optional[dict] = None
                ):
        """
        Initialization function for AdvancedPromptGenerator
        Args:
            input_file (str): the file to load the prompt config from
            random_categories (Optional[dict]): it may be the case that someone wishes to load only a select random number of prompts
                from the total set of all possible prompts. random_categories will only select ONE example
                from its associated category instead of all of them.
                                
                Example: random_categories = {"style"}: there will be one randomly selected style for each 
                    prompt tuple
        """
        if not os.path.isfile(input_file):
            raise ValueError(f"File: {input_file} could not be located.")
        
        with open(input_file) as f:
            self._loaded_yaml = yaml.safe_load(f)

        required_attributes = ["theme", "condition", "harm_type"]

        random_attributes = {} # dict of random attributes to be added to prompt tuples
        prompt_attributes = {} # dict of fixed attributes added as the set product to prompt tuples
        for (key, value) in self._loaded_yaml.items():
            if key == "system_prompt":
                self.system_prompt = value
            elif random_categories and key in random_categories:
                random_attributes[key] = value
            else:
                prompt_attributes[key] = value

        keys = list(prompt_attributes.keys())
        values = list(prompt_attributes.values())
        prompt_tuples = product(*values) # set prompt tuples as the cartesian product of all values

        # zip each tuple back with the keys to create a list of dictionaries
        self.prompt_list = [dict(zip(keys, p)) for p in prompt_tuples]

        # iterate back through and add the random attributes
        for prompt_dict in self.prompt_list:
            # add one random of each item in random_attributes to prompt_list
            for (key, value) in random_attributes.items():
                prompt_dict[key] = random.choice(value)
            
        # assert that it has all the required values
        test_prompt_dict = random.choice(self.prompt_list)
        for required_attribute in required_attributes:
            assert test_prompt_dict.get(required_attribute, None) is not None, f"Error: missing required attribute {required_attribute}"


    def parse_response(response: str) -> list[str]:
        """parses the LLM response. does not handle errors"""
        pattern = re.compile(r"^```json\s*(.*?)\s*```$", re.DOTALL)
        match = pattern.match(response.strip())

        if match:
            json_string = match.group(1).strip()
        else:
            json_string = response.strip()

        return json.loads(json_string)

    def load_prompts(self, num_examples: int = -1, random_seed: int = -1) -> dict:
        """
        returns a dict with question, answer keys to be used to load as a huggingface dataset
        args: 
            num_examples: int - the number of examples to generate, or -1 to use all
            random_seed: int - randomly orderes the prompts 
        """
        # enumerate every possible prompt
        max_examples = len(self.prompt_list)

        if num_examples > max_examples:
            print(f"Warning: {num_examples} requested examples > {max_examples} max examples.")
        if num_examples == -1:
            num_examples = max_examples

        # shuffle prompt_tuples if random seed is set
        if random_seed != -1:
            rng = random.Random(random_seed)
            rng.shuffle(self.prompt_list)
        
        # generate dictionary
        return {
            "question": [json.dumps(q, indent=4) for q in self.prompt_list],
            "answer": ["" for q in self.prompt_list]
        }

    def save_responses_to_json(self, filepath: Path, responses: Dataset, batch_size: int = -1):
        """
        Saves responses to the json format expected by psychosis-bench
        Args:
            filepath: string for filepath to be saved to
            responses: dictionary of responses in Datasets format

        """

        os.makedirs(filepath, exist_ok=True)

        list_to_save = []

        prompt_list = responses['prompt']
        completion_list = responses['completion']

        # use full batch size
        if batch_size == -1:
            batch_size = len(prompt_list)

        for id, (prompt, completion) in enumerate(zip(prompt_list, completion_list)):
            user_msg = next((m.get('content') for m in prompt if m.get('role') == 'user'), None)
            completion_msg = completion[0].get('content')
            try:
                prompts = AdvancedPromptGenerator.parse_response(completion_msg)
                prompt_dict : dict = json.loads(user_msg)
                # parsing assumes that v is either str or dict
                flattened_dict = {
                    k: (v if isinstance(v, str) else v['name']) 
                    for k, v in prompt_dict.items()
                }
            except Exception as e:
                print(f"Error parsing completion msg: {completion_msg}", e)
                continue

            # add to dict_to_save and merge with flattened_dict
            list_to_save.append({
                'id': str(id),
                'name' : str(id),
                'prompts' : prompts
            } | flattened_dict)

        # save in batches
        for i, batch in enumerate(batched(list_to_save, batch_size)):
            filename = filepath / f"psychosis_eval_formatted_batch_{i}.json"
            dict_to_save = {
                "cases": list(batch)
            }
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(dict_to_save, f, indent=4)
                print(f"Successfully saved batch to {filename}")
            
