from pathlib import Path
from datasets import Dataset
from data.base_prompt_generator import BasePromptGenerator
from data.advanced_prompt_generator.advanced_prompt_generator import AdvancedPromptGenerator

def main(
        results_path: Path,
        input_file: Path,
        random_categories : dict[str]
    ):
    gen = AdvancedPromptGenerator(input_file, random_categories)
    responses = Dataset.from_json(str(results_path))
    gen.save_responses_to_json(results_path.parent / 'batches', responses, batch_size=5)

if __name__ == '__main__':
    results_path = Path('outputs/evals/--openai--gpt-5-mini/e6563017/results.jsonl')
    input_file = Path('data/advanced_prompt_generator/harms_subset_prompt_config.yaml')
    random_categories={"style"}

    main(
        results_path,
        input_file,
        random_categories
    )