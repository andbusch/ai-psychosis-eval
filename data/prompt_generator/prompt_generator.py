import re
import json
import yaml
import itertools
from pathlib import Path
from data.base_prompt_generator import BasePromptGenerator
from datasets import Dataset

class PromptGenerator(BasePromptGenerator):
    """a synthetic data prompt data loader for a string of user prompts"""
    def parse_response(response: str) -> list[str]:
        """parses the LLM response. does not handle errors"""
        pattern = re.compile(r"^```json\s*(.*?)\s*```$", re.DOTALL)
        match = pattern.match(response.strip())

        if match:
            json_string = match.group(1).strip()
        else:
            json_string = response.strip()

        return json.loads(json_string)

    def __init__(self):
        """uses a random init value to initialize prompt characteristics"""
        with open("data/prompt_generator/prompt_config.yaml") as f:
            self._loaded_yaml = yaml.safe_load(f)

        self.system_prompt = self._loaded_yaml.get("system_prompt", "")
        themes = self._loaded_yaml.get("themes", [])
        harms = self._loaded_yaml.get("harms", [])
        conditions = ["Explicit", "Implicit"] 
        self.prompt_tuples = list(itertools.product(themes, harms, conditions))

    def format_prompt(self, theme: str, harm : str, condition: str)-> str:
        prompt = f"""Theme: {theme["name"]}\nDescription: {theme["description"]}\nHarm: {harm}\nCondition: {condition}"""
        return prompt

    def load_prompts(self, num_examples: int = -1) -> dict:
        """
        returns a dict with question, answer keys to be used to load as a huggingface dataset
        args: 
        - num_examples: int - the number of examples to generate, or -1 to use all
        """
        # enumerate every possible prompt
        max_examples = len(self.prompt_tuples)

        if num_examples > max_examples:
            print(f"Warning: {num_examples} requested examples > {max_examples} max examples.")
        if num_examples == -1:
            num_examples = max_examples
        
        # generate dictionary
        dataset_dict = {
            "question": [],
            "answer": []
        }
        for index, (theme, harm, condition) in enumerate(self.prompt_tuples):
            if index >= num_examples:
                break
            
            formatted_prompt = self.format_prompt(theme, harm, condition)
            dataset_dict["question"].append(formatted_prompt)
            dataset_dict["answer"].append("")

        return dataset_dict

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

        pattern = r"""
            ^Theme:\s*(?P<theme>.*)\n
            Description:\s*(?P<description>.*)\n
            Harm:\s*(?P<harm>.*)\n
            Condition:\s*(?P<condition>.*)$
        """

        prompt_list = responses['prompt']
        completion_list = responses['completion']
        print(len(prompt_list))

        id = 0
        for prompt_messages, completion in zip(prompt_list, completion_list):
            user_msg = next((m.get('content') for m in prompt_messages if m.get('role') == 'user'), None)
            completion_msg = completion[0].get('content')

            match = re.search(pattern, user_msg, re.VERBOSE | re.MULTILINE)
            if not match:
                raise ValueError(f'Error: user_msg [{user_msg}] could not be formatted as expected')
            
            metadata = match.groupdict()
            theme = metadata.get('theme', '')
            harm = metadata.get('harm', '')
            condition = metadata.get('condition', '')
            try:
                prompts = PromptGenerator.parse_response(completion_msg)
            except Exception as e:
                print(f"Error parsing completion msg: {completion_msg}", e)
                continue
            
            dict_to_save['cases'].append(
                {
                    'id': str(id),
                    'name': f"{theme}_{harm}_{condition}",
                    'theme': theme,
                    'condition': condition,
                    'harm_type': harm,
                    'prompts': prompts
                }
            )

            id += 1

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dict_to_save, f, indent=4)
            print(f"Successfully saved completions to {filename}")

        

        






