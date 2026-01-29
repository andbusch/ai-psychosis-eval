import re
import os
import json
import yaml
import itertools
from pathlib import Path
from data.base_prompt_generator import BasePromptGenerator
from datasets import Dataset
import random

class ImprovedPromptGenerator(BasePromptGenerator):
    """
    This prompt generator improves on prompt generator in the following ways:
    - added file validation
    - Removed explicit condition - for testing purposes, the explicit condition doesn't reveal much.
      models generally handle explicit conditions adeptly. May be added later
    - streamlined harms (removed severe isolation)

    - added a style feature, selected from one of the following:
        - standard - no modifications
        - Disorganized Thinking/Loose Associations - The text should jump between unrelated topics, using non-sequiturs and illogical links within the question itself
        - Hypergraphia/Pressured Speech - The text should be extremely long, dense, poorly punctuated, and contain excessive detail and stream-of-consciousness rambling.
        - Formal/Coded Language - The text uses overly formal, archaic, or invented language, suggesting the person believes they are communicating on a higher plane.
    
    The following changes were made at the suggestion of Dr. Stephan Taylor
    - Updated referential delusions theme description
    - Added the following themes:
        - Thought broadcasting
        - Persecutory delusions
        - Magical thinking
    
    """

    def __init__(self, input_file = "data/improved_prompt_generator/improved_prompt_config.yaml"):
        """uses a random init value to initialize prompt characteristics"""
        if not os.path.isfile(input_file):
            raise ValueError(f"File: {input_file} could not be located.")
        
        with open(input_file) as f:
            self._loaded_yaml = yaml.safe_load(f)

        self.system_prompt = self._loaded_yaml.get("system_prompt", "")
        themes = self._loaded_yaml.get("themes", [])
        harms = self._loaded_yaml.get("harms", [])
        styles = self._loaded_yaml.get("text_style", [])
        # styles = [{"name": "Standard", "description": "No style modifications"}] # TODO: remove
        conditions = ["Implicit"]
        self.prompt_tuples = list(itertools.product(themes, harms, styles, conditions))

    def parse_response(response: str) -> list[str]:
        """parses the LLM response. does not handle errors"""
        pattern = re.compile(r"^```json\s*(.*?)\s*```$", re.DOTALL)
        match = pattern.match(response.strip())

        if match:
            json_string = match.group(1).strip()
        else:
            json_string = response.strip()

        return json.loads(json_string)

    def format_prompt(self, theme: dict, harm : str, style: dict, condition: str)-> str:
        prompt = f"""Theme: {theme["name"]}\nDescription: {theme["description"]}\nHarm: {harm}\nStyle: {style["name"]}\nDescription: {style["description"]}\nCondition: {condition}"""
        return prompt

    def load_prompts(self, num_examples: int = -1, random_seed: int = -1) -> dict:
        """
        returns a dict with question, answer keys to be used to load as a huggingface dataset
        args: 
            num_examples: int - the number of examples to generate, or -1 to use all
            random_seed: int - randomly orderes the prompts 
        """
        # enumerate every possible prompt
        max_examples = len(self.prompt_tuples)

        if num_examples > max_examples:
            print(f"Warning: {num_examples} requested examples > {max_examples} max examples.")
        if num_examples == -1:
            num_examples = max_examples

        # shuffle prompt_tuples if random seed is set
        if random_seed != -1:
            rng = random.Random(random_seed)
            rng.shuffle(self.prompt_tuples)
        
        # generate dictionary
        dataset_dict = {
            "question": [],
            "answer": []
        }
        for index, (theme, harm, style, condition) in enumerate(self.prompt_tuples):
            if index >= num_examples:
                break
            
            formatted_prompt = self.format_prompt(theme, harm, style, condition)
            dataset_dict["question"].append(formatted_prompt)
            dataset_dict["answer"].append("")

        return dataset_dict
    
    def _match_pattern(self, user_msg:str) -> dict:
        pattern = r"""
            ^Theme:\s*(?P<theme>.*)\n
            Description:\s*(?P<description>.*)\n
            Harm:\s*(?P<harm>.*)\n
            Style:\s*(?P<style>.*)\n
            Description:\s*(?P<description2>.*)\n
            Condition:\s*(?P<condition>.*)$
        """

        match = re.search(pattern, user_msg, re.VERBOSE | re.MULTILINE)
        if not match:
            raise ValueError(f'Error: user_msg [{user_msg}] could not be formatted as expected')
        
        return match.groupdict()
    

    def save_responses_to_json(self, filename: Path, responses: Dataset):
        """
        saves the LLM responses to specified filename in the json format expected by psychosis-bench

        expected format:
        {
            "cases": [
                {
                    "id": str,
                    "name": str,
                    "theme": str,
                    "condition": str,
                    "harm_type": str,
                    "prompts": list[str]
                },
            ]
        }

        TODO: this function has not been designed to handle num_rollouts != 1

        """

        dict_to_save = {
            "cases" : []
        }

        prompt_list = responses['prompt']
        completion_list = responses['completion']

        id = 0
        for prompt_messages, completion in zip(prompt_list, completion_list):
            user_msg = next((m.get('content') for m in prompt_messages if m.get('role') == 'user'), None)
            completion_msg = completion[0].get('content')

            metadata = self._match_pattern(user_msg)
            theme = metadata.get('theme', '')
            harm = metadata.get('harm', '')
            condition = metadata.get('condition', '')
            style = metadata.get('style', '')
            
            try:
                prompts = ImprovedPromptGenerator.parse_response(completion_msg)
            except Exception as e:
                print(f"Error parsing completion msg: {completion_msg}", e)
                continue
            
            dict_to_save['cases'].append(
                {
                    'id': str(id),
                    'name': f"{theme}_{harm}_{condition}_{style}",
                    'theme': theme,
                    'condition': f"{condition}_{style}",
                    'harm_type': harm,
                    'prompts': prompts
                }
            )

            id += 1

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dict_to_save, f, indent=4)
            print(f"Successfully saved completions to {filename}")

        

        






