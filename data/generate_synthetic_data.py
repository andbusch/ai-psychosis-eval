"""
https://verifiers.readthedocs.io/en/latest/
"""

import os
import json
from datasets import Dataset
import verifiers as vf
from verifiers.utils.eval_utils import save_results, make_dataset
from openai import AsyncOpenAI
from data.base_prompt_generator import BasePromptGenerator

from data.prompt_generator.prompt_generator import PromptGenerator
from data.improved_prompt_generator.improved_prompt_generator import ImprovedPromptGenerator
from data.advanced_prompt_generator.advanced_prompt_generator import AdvancedPromptGenerator

import asyncio

def reward_response(prompt, completion, answer, state) -> int:
    response = completion[-1]['content']
    try:
        if len(PromptGenerator.parse_response(response)) != 12:
            return 0.0
    except Exception as e:
        return 0.0

    return 1.0

def load_dataset(gen: BasePromptGenerator, num_examples: int, random_seed: int = -1) -> Dataset:
    """
    loads the dataset used to prompt the synthetic data generator
    Args:
        - gen: an instance of prompt generator class
        - num_examples: the number of prompts to be generated

    Returns:
        - dataset of size num_examples
    """
    dict = gen.load_prompts(num_examples, random_seed)
    return Dataset.from_dict(dict)

async def main(base_url: str, api_key: str, model: str, gen: BasePromptGenerator, num_examples: int = -1, random_seed: int = 11111111, rollouts_per_example: int = 1):
    # load prompts
    assert hasattr(gen, "system_prompt"), "error: gen does not have system_prompt attribute"
    #assert rollouts_per_example == 1, "rollouts_per_example must be 1 until prompt generators are implemented to handle multiple."

    system_prompt : str = gen.system_prompt
    dataset = load_dataset(gen, num_examples, random_seed)
    
    # generate data
    env = vf.SingleTurnEnv(
        dataset=dataset,
        rubric=vf.Rubric(funcs=[reward_response]),
        system_prompt=system_prompt,
    )
    
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    
    results = await env.evaluate(
        client = client,
        model = model,
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
        sampling_args={"temperature": 0.7},
        max_concurrent=1
    )

    #env.make_dataset(results, push_to_hub=False, hub_name="abusch472/glaze-rl-data")
    save_results(
        results = results,
        push_to_hf_hub=False,
        hf_hub_dataset_name="abusch472/ai-psychosis-eval-data"
    )

    # gen.save_responses_to_json(
    #     filename=f"{results.metadata.path_to_save}/psychosis-eval-formatted.json",
    #     responses = make_dataset(results)
    # ) TODO: remove

if __name__ == "__main__":
    base_url = "https://openrouter.ai/api/v1"
    model = "openai/gpt-5-mini"
    api_key_loc = "OPENAI_API_KEY"

    # base_url = "https://api.anthropic.com/v1/"
    # model = "claude-haiku-4-5-20251001"
    # api_key_loc = "ANTHROPIC_API_KEY"

    # run parameters
    gen = AdvancedPromptGenerator()
    num_examples = 2
    random_seed = -1 # no shuffling
    rollouts_per_example = 2

    api_key = os.environ.get(api_key_loc)
    if api_key is None:
        raise ValueError(f"{api_key_loc} must be provided")

    asyncio.run(main(base_url, api_key, model, gen, num_examples, random_seed, rollouts_per_example))