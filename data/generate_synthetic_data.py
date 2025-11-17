"""
https://verifiers.readthedocs.io/en/latest/
"""

import os
from datasets import Dataset
import verifiers as vf
from openai import OpenAI

from prompt_generator import PromptGenerator, SinglePromptGenerator

def load_dataset(gen: PromptGenerator, num_examples: int) -> Dataset:
    """
    loads the dataset used to prompt the synthetic data generator
    Args:
        - gen: an instance of prompt generator class
        - num_examples: the number of prompts to be generated

    Returns:
        - dataset of size num_examples
    """
    dict = gen.load_prompts(num_examples)
    return Dataset.from_dict(dict)

def main():
    # run parameters
    model = "gpt-4.1-nano"
    num_examples = 1
    rollouts_per_example = 2

    # load prompts
    gen = SinglePromptGenerator()
    system_prompt = gen.system_prompt
    dataset = load_dataset(gen, num_examples)
    print(dataset["question"])
    return
    
    # generate data
    env = vf.SingleTurnEnv(
        dataset=dataset,
        rubric=vf.Rubric(),
        system_prompt=system_prompt,
    )

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url="https://openrouter.ai/api/v1"
    )

    results = env.evaluate(
        client = client,
        model = model,
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example
    )

    print(results)

    #env.make_dataset(results, push_to_hub=True, hub_name="abusch472/glaze-rl-data")

if __name__ == "__main__":
    main()