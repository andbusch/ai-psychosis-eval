# AI-Psychosis-Eval

AI-Psychosis-Eval is a repository containing a synthetic data pipeline for generating additional test cases to be used with psychosis-bench.

## Synthetic Data Pipeline

The synthetic data pipeline is still in development. Please reach out if you would like to contribute.

To run the pipeline, use the command

```bash
uv run --env-file .env python -m data.generate_synthetic_data
```

### Prompt Generator

In progress.

### Advanced Prompt Generator

In progress.

## Results Analzyer

To run the results analyzer, use the command

```bash
uv run -m results-analyzer.results_analyzer
```

## Disclaimer

The results and analyses published from this project must be interpreted with the following limitations in mind:

* **Benchmark-Specific Performance:** The findings reflect a model's performance **only on this specific, adversarial benchmark**. They are not a measure of a model's general utility, overall safety, or its behavior under normal operating conditions.
* **Not a Clinical Assessment:** This evaluation is a technical benchmark, not a clinical or psychological diagnosis. A model's tendency to generate responses that align with the traits in our dataset is an observation of its behavior on our test; it is **not a claim that the model induces psychosis in real-world users.**

