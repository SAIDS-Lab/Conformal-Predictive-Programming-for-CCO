"""
In this file, we plot the results for case study 3.
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
    with open("case_studies_results/results_case_study_3/results_step_1_robust.json", "r") as file:
        results_step_1_robust = json.load(file)
    with open("case_studies_results/results_case_study_3/results_step_2_robust.json", "r") as file:
        results_step_2_robust = json.load(file)
    with open("case_studies_results/results_case_study_3/results_baseline_step_1.json", "r") as file:
        results_baseline_step_1 = json.load(file)
    with open("case_studies_results/results_case_study_3/results_baseline_step_2.json", "r") as file:
        results_baseline_step_2 = json.load(file)

    with open("case_studies_results/results_case_study_3/results_step_1_mondrian.json", "r") as file:
        results_step_1_mondrian = json.load(file)
    with open("case_studies_results/results_case_study_3/results_step_2_mondrian.json", "r") as file:
        results_step_2_mondrian = json.load(file)

    # Report time statistics.
    print("Robust: Num timeout:", results_step_1_robust["CPP-MIP"]["num_timeout"])
    print("Robust: Num infeasible:", results_step_1_robust["CPP-MIP"]["num_infeasible"])
    print("Baseline: Num timeout:", results_baseline_step_1["CPP-MIP"]["num_timeout"])
    print("Baseline: Num infeasible:", results_baseline_step_1["CPP-MIP"]["num_infeasible"])
    print("Mondrian: Num timeout:", results_step_1_mondrian["CPP-MIP"]["num_timeout"])
    print("Mondrian: Num infeasible:", results_step_1_mondrian["CPP-MIP"]["num_infeasible"])
    print("SOLVER TIME:")
    print(f"Robust: Average solving time for CPP-MIP: {np.mean(results_step_1_robust['CPP-MIP']['solver_times'])}")
    print(f"Baseline: Average solving time for CPP-MIP: {np.mean(results_baseline_step_1['CPP-MIP']['solver_times'])}")
    print(f"Mondrian: Average solving time for CPP-MIP: {np.mean(results_step_1_mondrian['CPP-MIP']['solver_times'])}")

    # Report EC.
    robust_EC = results_step_2_robust["CPP-MIP"]["EC"]
    baseline_EC = results_baseline_step_2["CPP-MIP"]["EC"]
    print("Robust: EC:", robust_EC)
    print("Baseline: EC:", baseline_EC)

    # Report MEC.
    MEC_vanilla = results_step_2_mondrian["CPP-MIP"]["MEC_vanilla"]
    MEC_mondrian = results_step_2_mondrian["CPP-MIP"]["MEC_mondrian"]
    print("Vanilla MEC:", MEC_vanilla)
    print("Mondrian MEC:", MEC_mondrian)

    # Set up the 3 plots.
    fig, ax = plt.subplots(1, 3, figsize=(22, 4))
    ax = ax.flatten()
    # Plot Cs.
    robust_Cs = results_step_2_robust["CPP-MIP"]["Cs"]
    baseline_Cs = results_baseline_step_2["CPP-MIP"]["Cs"]
    min_value = min(min(robust_Cs), min(baseline_Cs))
    max_value = max(max(robust_Cs), max(baseline_Cs))
    y_1, x_1 = np.histogram(robust_Cs, bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins, (max_value - min_value) / num_bins))
    y_2, x_2 = np.histogram(baseline_Cs, bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins, (max_value - min_value) / num_bins))
    sns.lineplot(x=x_1[:-1], y=y_1, ax=ax[0])
    ax[0].fill_between(x=x_1[:-1], y1=y_1, y2=0, alpha=0.3, label="$\\tilde{C}(x^*)$")
    sns.lineplot(x=x_2[:-1], y=y_2, ax=ax[0])
    ax[0].fill_between(x=x_2[:-1], y1=y_2, y2=0, alpha=0.3, label="$C_m(x^*)$")
    ax[0].legend(fontsize=legend_size, loc="upper right")
    ax[0].tick_params("x", labelsize=label_size)
    ax[0].tick_params("y", labelsize=label_size)
    ax[0].set_ylim(0, 100)
    ax[0].set_ylabel("Frequency", fontsize=font_size)
    ax[0].set_title("$\\tilde{C}(x^*)$ for RCPP and $C_m(x^*)$ for CPP", fontsize=title_size, y=title_position)

    # Plot J.
    robust_J = results_step_2_robust["CPP-MIP"]["J"]
    baseline_J = results_baseline_step_2["CPP-MIP"]["J"]
    min_value = min(min(robust_J), min(baseline_J))
    max_value = max(max(robust_J), max(baseline_J))
    y_1, x_1 = np.histogram(robust_J, bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins, (max_value - min_value) / num_bins))
    y_2, x_2 = np.histogram(baseline_J, bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins, (max_value - min_value) / num_bins))
    sns.lineplot(x=x_1[:-1], y=y_1, ax=ax[1])
    ax[1].fill_between(x=x_1[:-1], y1=y_1, y2=0, alpha=0.3, label="Robust CPP")
    sns.lineplot(x=x_2[:-1], y=y_2, ax=ax[1])
    ax[1].fill_between(x=x_2[:-1], y1=y_2, y2=0, alpha=0.3, label="CPP")
    ax[1].legend(fontsize=legend_size, loc="upper right")
    ax[1].tick_params("x", labelsize=label_size)
    ax[1].tick_params("y", labelsize=label_size)
    ax[1].set_ylim(0, 100)
    ax[1].set_title("$J(x^*)$", fontsize=title_size, y=title_position)

    # Plot Cs (for mondrian).
    vanilla_Cs = results_step_2_mondrian["CPP-MIP"]["Cs_m_vanilla"]
    mondrian_Cs = results_step_2_mondrian["CPP-MIP"]["Cs_m_mondrian"]
    min_value = min(min(vanilla_Cs), min(mondrian_Cs))
    max_value = max(max(vanilla_Cs), max(mondrian_Cs))
    y_1, x_1 = np.histogram(vanilla_Cs, bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins, (max_value - min_value) / num_bins))
    y_2, x_2 = np.histogram(mondrian_Cs, bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins, (max_value - min_value) / num_bins))
    sns.lineplot(x=x_1[:-1], y=y_1, ax=ax[2])
    ax[2].fill_between(x=x_1[:-1], y1=y_1, y2=0, alpha=0.3, label="$C_m(x^*)$")
    sns.lineplot(x=x_2[:-1], y=y_2, ax=ax[2])
    ax[2].fill_between(x=x_2[:-1], y1=y_2, y2=0, alpha=0.3, label="$C_{G_2}$")
    ax[2].legend(fontsize=legend_size, loc="upper right")
    ax[2].tick_params("x", labelsize=label_size)
    ax[2].tick_params("y", labelsize=label_size)
    ax[2].set_ylim(0, 100)
    ax[2].set_title("$C_m(x^*)$ and $C_{G_2}$", fontsize=title_size, y=title_position)

    fig.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig("case_studies_plots/case_study_3_figure.pdf")


if __name__ == "__main__":
    main()