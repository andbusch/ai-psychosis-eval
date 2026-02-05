import json
import numpy as np
from pandas import DataFrame as df
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

def save_aggregate_box_plot(dataframe: df, save_path: Path, groupby: str):
    """
    saves a box plot of the groupby statistic as the x, the metric as the hue.
    """

    df_long = dataframe.melt(
        id_vars=[groupby], 
        value_vars=["avg_dcs", "avg_hes", "avg_sis"], 
        var_name='Metric', 
        value_name='Avg Score'
    )

    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")

    sns.boxplot(
        data=df_long, 
        x=groupby, 
        y='Avg Score', 
        hue='Metric'
    )

    plt.title(f'Score Distribution by {groupby.capitalize()}')
    plt.ylabel('Score Value (0.0 - 2.0)')
    plt.xlabel(groupby.capitalize())
    plt.ylim(0, 2.1)

    plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_per_model_box_plot(dataframe: df, save_path: Path, groupby: str, metric: str):
    """
    saves a box plot to a given path
    uses model as x axis, 
    """

    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")

    plot = sns.boxplot(
        data=dataframe, 
        x='model', 
        y=metric, 
        hue=groupby
    )

    # Clean up the labels
    plt.title(f'Distribution of {metric} by Model and {groupby.capitalize()}')
    plt.ylabel(metric.replace('_', ' ').upper())
    plt.xlabel('Model')
    
    # Adjust legend so it doesn't cover the data
    plt.legend(title=groupby.capitalize(), bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def process_single_file(data_dict : dict, file: Path):
    """
    Adds summary data to data_dict
    """
    with open(file, 'r') as f:
        file_json = json.load(f)

    for result in file_json["results"]:
        # add metadata
        for (test_k, test_v) in result["test_case"].items():
            if test_k != "prompts": 
                # add everything except for prompts
                data_dict.setdefault(test_k, []).append(test_v)
        data_dict["model"].append(result["model"]) # add model

        # calculate summary statistics
        avg_dcs = np.mean([t["scores"]["dcs"] for t in result["turns"] if (t["scores"].get("dcs") is not None)])
        avg_hes = np.mean([t["scores"]["hes"] for t in result["turns"] if (t["scores"].get("hes") is not None)])
        avg_sis = np.mean([t["scores"]["sis"] for t in result["turns"] if (t["scores"].get("sis") is not None)])

        data_dict.setdefault("avg_dcs", []).append(avg_dcs)
        data_dict.setdefault("avg_hes", []).append(avg_hes)
        data_dict.setdefault("avg_sis", []).append(avg_sis)


def initialize_dataframe_from_dir(dir_path: Path) -> df:
    """
    Initializes a dataframe from a directory path
    Iterates through every file in dir_path, adds results to a dataframe
        Args:
            dir_path: directory path
    """
    assert not dir_path.is_file(), f"Error: {dir_path.name} must be a directory."
    # data dictionary for which each row represents a single test run
    # also will add any additional element provided
    data_dict = {
        "model": [],
        "id": [],
        "name" : [],
        "theme" : [],
        "condition" : [],
        "harm_type": []
    }
    for file in dir_path.iterdir():
        if file.is_file() and file.suffix == ".json":
            process_single_file(data_dict, file)
        
    assert len(set((len(v) for v in data_dict.values()))) == 1, "Error: missing values for some parameter."
    return df(data_dict)
        

def main():
    results_dir = Path("outputs/experiment-subset/results")
    dataframe : df = initialize_dataframe_from_dir(results_dir)
    # dataframe.to_csv(results_dir / "results.csv", index=False)

    # save per model box plot
    save_per_model_box_plot(dataframe, results_dir / "model_results_jailbreak_dcs.png", "jailbreak", "avg_dcs")
    save_per_model_box_plot(dataframe, results_dir / "model_results_jailbreak_hes.png", "jailbreak", "avg_hes")
    save_per_model_box_plot(dataframe, results_dir / "model_results_jailbreak_sis.png", "jailbreak", "avg_sis")

    # save aggregate box plot
    save_aggregate_box_plot(dataframe, results_dir / "aggregate_scores_jailbreak.png", "jailbreak")

if __name__ == '__main__':
    main()