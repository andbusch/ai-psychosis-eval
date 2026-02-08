import json
import numpy as np
from pandas import DataFrame as df
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
import scikit_posthocs as sp # You may need to pip install scikit-posthocs

def check_confounding(dataframe, target_var, confounder_var):
    """
    Checks if the effect of target_var (e.g., 'jailbreak') is consistent 
    across different levels of confounder_var (e.g., 'model').
    """
    metrics = ["avg_dcs", "avg_hes", "avg_sis"]
    
    print(f"\n{'='*60}")
    print(f"CONFOUNDING ANALYSIS: '{target_var}' efficacy within each '{confounder_var}'")
    print(f"{'='*60}")

    # Get unique levels of the confounder (e.g., each individual model)
    confounder_levels = dataframe[confounder_var].unique()

    for level in confounder_levels:
        print(f"\n>>> Sub-analysis for {confounder_var.upper()}: {level}")
        subset = dataframe[dataframe[confounder_var] == level]

        for metric in metrics:
            # Group target_var within this specific confounder level
            groups = [
                group[metric].dropna().values 
                for name, group in subset.groupby(by=target_var)
            ]

            if len(groups) < 2:
                print(f"   Metric: {metric:8} | Not enough {target_var} variety in this {confounder_var}.")
                continue

            h_stat, p_val = stats.kruskal(*groups)
            sig_status = "!!!" if p_val < 0.05 else "---"
            
            print(f"   {sig_status} Metric: {metric:8} | p-value: {p_val:.4e} | ({target_var} effect)")

            # If significant within this model, show which jailbreak did it
            if p_val < 0.05:
                posthoc = sp.posthoc_dunn(subset, val_col=metric, group_col=target_var, p_adjust='holm')
                with pd.option_context('display.max_columns', None, 'display.width', 1000, 'display.precision', 4):
                    print(f"\n      [Dunn's Post-Hoc for {level}]")
                    print(posthoc.iloc[: , :5]) # Limiting columns for readability if very wide

def kruskal_wallis(dataframe: df, variable: str, dunn: bool = True):
    """
    Performs Kruskal-Wallis H-test for avg_dcs, avg_hes, and avg_sis
    across the groups defined by 'variable'.
    """
    metrics = ["avg_dcs", "avg_hes", "avg_sis"]
    
    print(f"\n--- Kruskal-Wallis Analysis: Grouped by {variable.upper()} ---")
    
    for metric in metrics:
        # Group the data: creates a list of arrays (one for each group level)
        groups = [
            group[metric].dropna().values 
            for name, group in dataframe.groupby(by=variable)
        ]
        
        # Check if we have at least 2 groups to compare
        if len(groups) < 2:
            print(f"Skipping {metric}: Not enough groups in {variable}.")
            continue

        h_stat, p_val = stats.kruskal(*groups)
        
        print(f"Metric: {metric:8} | H-statistic: {h_stat:7.2f} | p-value: {p_val:.4e}")

        # Significance Check (Standard alpha = 0.05)
        if p_val < 0.05:
            print(f"   [!] SIGNIFICANT: The {variable} likely impacts the {metric}.")
            
            # Dunn's test
            if dunn:
                print("   Performing Dunn's Post-Hoc test...")
                posthoc = sp.posthoc_dunn(dataframe, val_col=metric, group_col=variable, p_adjust='holm')
                with pd.option_context('display.max_columns', None, 
                           'display.expand_frame_repr', False, 
                           'display.precision', 6):
                    print(posthoc)
        else:
            print(f"   [-] NOT SIGNIFICANT: No evidence that {variable} impacts {metric}.")

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

    # # save per model box plot
    # save_per_model_box_plot(dataframe, results_dir / "model_results_jailbreak_dcs.png", "jailbreak", "avg_dcs")
    # save_per_model_box_plot(dataframe, results_dir / "model_results_jailbreak_hes.png", "jailbreak", "avg_hes")
    # save_per_model_box_plot(dataframe, results_dir / "model_results_jailbreak_sis.png", "jailbreak", "avg_sis")

    # # save aggregate box plot
    # save_aggregate_box_plot(dataframe, results_dir / "aggregate_scores_jailbreak.png", "jailbreak")
    # save_aggregate_box_plot(dataframe, results_dir / "aggregate_scores_model.png", "model")

    # ---KRUSKAL WALLIS---
    kruskal_wallis(dataframe, "jailbreak")
    # kruskal_wallis(dataframe, "model")
    # kruskal_wallis(dataframe, "harm_type")
    # kruskal_wallis(dataframe, "theme")
    # kruskal_wallis(dataframe, "style")

    # check_confounding(dataframe, "jailbreak", "model")


if __name__ == '__main__':
    main()