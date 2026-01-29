import os
import yaml
import itertools
import re
import json
import random

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

    def __init__(self, input_file = "data/advanced_prompt_generator/advanced_prompt_config.yaml"):
        """uses a random init value to initialize prompt characteristics"""
        if not os.path.isfile(input_file):
            raise ValueError(f"File: {input_file} could not be located.")
        
        with open(input_file) as f:
            self._loaded_yaml = yaml.safe_load(f)

        prompt_attributes = {}
        for (key, value) in self._loaded_yaml.items():
            if key == "system_prompt":
                self.system_prompt = value
            else:
                prompt_attributes[key] = value

        keys = list(prompt_attributes.keys())
        values = list(prompt_attributes.values())
        prompt_tuples = itertools.product(*values) # set prompt tuples as the cartesian product of all values

        # zip each tuple back with the keys to create a list of dictionaries
        self.prompt_list = [dict(zip(keys, p)) for p in prompt_tuples]

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

    def save_responses_to_json(self, filename: str, responses: dict):
        """
          {
            "theme": 
            {
                "name": [the common AI-psychosis theme],
                "decription": [a description of the theme]
            },
            "harm": [the harm the user is at risk for],
            "style": {
                "name": [the specific style of the user text, or standard],
                "description": [a description of the style]
            },
            "condition" : [implicit or explicit]
            "jailbreak" : 
            {
                "name": [the name for the attempted jailbreak],
                "description": [a description of how the user attempts to jailbreak the model]    
            }
        }
        """

        dict_to_save = {
            "cases" : []
        }

        prompt_list = responses['prompt']
        completion_list = responses['completion']

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
            dict_to_save['cases'].append({
                'id': str(id),
                'name' : str(id),
                'prompts' : prompts
            } | flattened_dict)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dict_to_save, f, indent=4)
            print(f"Successfully saved completions to {filename}")
            
