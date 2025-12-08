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

from data.prompt_generator import PromptGenerator

import asyncio

def reward_response(prompt, completion, answer, state) -> int:
    response = completion[-1]['content']
    try:
        if len(PromptGenerator.parse_response(response)) != 12:
            return 0.0
    except Exception as e:
        return 0.0

    return 1.0

def load_dataset(gen: BasePromptGenerator, num_examples: int) -> Dataset:
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

async def main(base_url: str, api_key: str, model: str):
    # run parameters
    num_examples = 5
    rollouts_per_example = 1

    # load prompts
    gen = PromptGenerator()
    system_prompt = gen.system_prompt
    dataset = load_dataset(gen, num_examples)
    
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
        sampling_args={"temperature": 0.7}
    )

    #env.make_dataset(results, push_to_hub=False, hub_name="abusch472/glaze-rl-data")
    save_results(
        results = results,
        push_to_hf_hub=False,
        hf_hub_dataset_name="abusch472/ai-psychosis-eval-data"
    )

    gen.save_responses_to_json(
        filename=f"{results.metadata.path_to_save}/psychosis-eval-formatted.json",
        responses = make_dataset(results)
    )

if __name__ == "__main__":
    # base_url = "https://openrouter.ai/api/v1"

    base_url = "https://api.anthropic.com/v1/"
    model = "claude-haiku-4-5-20251001"

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key is None:
        raise ValueError("ANTHROPIC_API_KEY must be provided")

    asyncio.run(main(base_url, api_key, model))