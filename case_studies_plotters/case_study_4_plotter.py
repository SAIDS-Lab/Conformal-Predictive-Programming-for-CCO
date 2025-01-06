"""
In this file, we plot the results for case study 4.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import MaxNLocator

def main():
    num_bins = 12
    font_size = 22
    label_size = 17
    legend_size = 18
    title_size = 24
    title_position = -0.2

    # Load the results.
    with open("case_studies_results/results_case_study_4/results_step_1.json", "r") as file:
        results_step_1 = json.load(file)
    with open("case_studies_results/results_case_study_4/results_step_2.json", "r") as file:
        results_step_2 = json.load(file)
    
    # Report time statistics.
    print("Num timeout union:", results_step_1["union"]["num_timeout"])
    print("Num infeasible union:", results_step_1["union"]["num_infeasible"])
    print("Num timeout max:", results_step_1["max"]["num_timeout"])
    print("Num infeasible max:", results_step_1["max"]["num_infeasible"])
    print("SOLVER TIME:")
    print(f"Average solving time for union (excluding auxiliary a posteriori problem): {np.mean(results_step_1['union']['solver_times'])}")
    print(f"Average solving time for max: {np.mean(results_step_1['max']['solver_times'])}")

    # Report EC.
    EC_union = results_step_2["union"]["EC"]
    EC_max = results_step_2["max"]["EC"]
    print("EC union:", EC_union)
    print("EC max:", EC_max)

    # Set up the 2 plots.
    fig, ax = plt.subplots(1, 2, figsize=(22, 4))
    ax = ax.flatten()
    # Plot Cs.
    union_Cs = results_step_2["union"]["Cs_m"]
    max_Cs = results_step_2["max"]["Cs_m"]
    min_value = min(min(union_Cs), min(max_Cs))
    max_value = max(max(union_Cs), max(max_Cs))
    y_1, x_1 = np.histogram(union_Cs, bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins, (max_value - min_value) / num_bins))
    y_2, x_2 = np.histogram(max_Cs, bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins, (max_value - min_value) / num_bins))
    sns.lineplot(x=x_1[:-1], y=y_1, ax=ax[0])
    ax[0].fill_between(x=x_1[:-1], y1=y_1, y2=0, alpha=0.3, label="Union Method")
    sns.lineplot(x=x_2[:-1], y=y_2, ax=ax[0])
    ax[0].fill_between(x=x_2[:-1], y1=y_2, y2=0, alpha=0.3, label="Pointwise-Maximum Method")
    ax[0].legend(fontsize=legend_size, loc="upper right")
    ax[0].tick_params("x", labelsize=label_size)
    ax[0].tick_params("y", labelsize=label_size)
    ax[0].set_ylim(0, 60)
    ax[0].set_ylabel("Frequency", fontsize=font_size)
    ax[0].set_title("$(b) \\bar{C}(x_l^*)$", fontsize=title_size, y=title_position)

    # Plot J.
    union_J = results_step_1["union"]["optimal_values"]
    max_J = results_step_1["max"]["optimal_values"]
    min_value = min(min(union_J), min(max_J))
    max_value = max(max(union_J), max(max_J))
    y_1, x_1 = np.histogram(union_J, bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins, (max_value - min_value) / num_bins))
    y_2, x_2 = np.histogram(max_J, bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins, (max_value - min_value) / num_bins))
    sns.lineplot(x=x_1[:-1], y=y_1, ax=ax[1])
    ax[1].fill_between(x=x_1[:-1], y1=y_1, y2=0, alpha=0.3, label="Union Method")
    sns.lineplot(x=x_2[:-1], y=y_2, ax=ax[1])
    ax[1].fill_between(x=x_2[:-1], y1=y_2, y2=0, alpha=0.3, label="Pointwise-Maximum Method")
    ax[1].legend(fontsize=legend_size, loc="upper right")
    ax[1].tick_params("x", labelsize=label_size)
    ax[1].tick_params("y", labelsize=label_size)
    ax[1].set_ylim(0, 60)
    ax[1].set_title("$(b) J(x_l^*)$", fontsize=title_size, y=title_position)

    fig.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig("case_studies_plots/case_study_4_figure.pdf")


if __name__ == "__main__":
    main()