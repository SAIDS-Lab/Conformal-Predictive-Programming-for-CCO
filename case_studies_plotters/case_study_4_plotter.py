"""
In this file, we plot the results for case study 4.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def main():
    num_bins = 15
    font_size = 22
    label_size = 17
    legend_size = 18
    title_size = 24
    title_position = -0.2

    with open("case_studies_results/results_case_study_4/results_step_1.json", "r") as file:
        results_step_1 = json.load(file)
        results_step_1_union = results_step_1["CPP-MIP Union"]
        results_step_1_max = results_step_1["CPP-MIP Max"]
    
    with open("case_studies_results/results_case_study_4/results_step_2.json", "r") as file:
        results_step_2 = json.load(file)
        results_step_2_union = results_step_2["CPP-MIP Union"]
        results_step_2_max = results_step_2["CPP-MIP Max"]

    results_plot_mip_union_1 = [results_step_2_union["40"]["Cs"], results_step_1_union["40"]["optimal_values"], results_step_1_union["40"]["empirical_coverages"], results_step_2_union["40"]["posterior_coverages"]]
    results_plot_mip_max_1 = [results_step_2_max["40"]["Cs"], results_step_1_max["40"]["optimal_values"], results_step_1_max["40"]["empirical_coverages"], results_step_2_max["40"]["posterior_coverages"]]
    results_plot_mip_union_2 = [results_step_2_union["80"]["Cs"], results_step_1_union["80"]["optimal_values"], results_step_1_union["80"]["empirical_coverages"], results_step_2_union["80"]["posterior_coverages"]]
    results_plot_mip_max_2 = [results_step_2_max["80"]["Cs"], results_step_1_max["80"]["optimal_values"], results_step_1_max["80"]["empirical_coverages"], results_step_2_max["80"]["posterior_coverages"]]
    results_plot_mip_union_3 = [results_step_2_union["120"]["Cs"], results_step_1_union["120"]["optimal_values"], results_step_1_union["120"]["empirical_coverages"], results_step_2_union["120"]["posterior_coverages"]]
    results_plot_mip_max_3 = [results_step_2_max["120"]["Cs"], results_step_1_max["120"]["optimal_values"], results_step_1_max["120"]["empirical_coverages"], results_step_2_max["120"]["posterior_coverages"]]

    #####################################################
    # The followings are the plot for our CPP framework
    fig, ax = plt.subplots(2, 4, figsize=(22, 10))
    ax = ax.flatten()
    for i in range(4):
        min_value = min(min(results_plot_mip_union_1[i]), min(results_plot_mip_union_2[i]), min(results_plot_mip_union_3[i]))
        max_value = max(max(results_plot_mip_union_1[i]), max(results_plot_mip_union_2[i]), max(results_plot_mip_union_3[i]))
        y_1, x_1 = np.histogram(results_plot_mip_union_1[i], bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins, (max_value - min_value) / num_bins))
        y_2, x_2 = np.histogram(results_plot_mip_union_2[i], bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins, (max_value - min_value) / num_bins))
        y_3, x_3 = np.histogram(results_plot_mip_union_3[i], bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins, (max_value - min_value) / num_bins))

        if i == 3:
            sns.lineplot(x=x_1[:-1], y=y_1, ax = ax[i])
            ax[i].fill_between(x=x_1[:-1], y1=y_1, y2=0, alpha=0.3, label = "K = 40")
            sns.lineplot(x=x_2[:-1], y=y_2, ax = ax[i])
            ax[i].fill_between(x=x_2[:-1], y1=y_2, y2=0, alpha=0.3, label = "K = 80")
            sns.lineplot(x=x_3[:-1], y=y_3, ax = ax[i])
            ax[i].fill_between(x=x_3[:-1], y1=y_3, y2=0, alpha=0.3, label = "K = 120")
            ax[i].legend(fontsize = legend_size)
        else:
            sns.lineplot(x=x_1[:-1], y=y_1, ax = ax[i])
            ax[i].fill_between(x=x_1[:-1], y1=y_1, y2=0, alpha=0.3)
            sns.lineplot(x=x_2[:-1], y=y_2, ax = ax[i])
            ax[i].fill_between(x=x_2[:-1], y1=y_2, y2=0, alpha=0.3)
            sns.lineplot(x=x_3[:-1], y=y_3, ax = ax[i])
            ax[i].fill_between(x=x_3[:-1], y1=y_3, y2=0, alpha=0.3)
        ax[i].tick_params("x", labelsize=label_size)
        ax[i].tick_params("y", labelsize=label_size)
        ax[i].set_ylim(0,35)
        # ax[i].set_ylabel("Frequency", fontsize = font_size)
    
    for i in range(4):
        min_value = min(min(results_plot_mip_max_1[i]), min(results_plot_mip_max_2[i]), min(results_plot_mip_max_3[i]))
        max_value = max(max(results_plot_mip_max_1[i]), max(results_plot_mip_max_2[i]), max(results_plot_mip_max_3[i]))
        y_1, x_1 = np.histogram(results_plot_mip_max_1[i], bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins, (max_value - min_value) / num_bins))
        y_2, x_2 = np.histogram(results_plot_mip_max_2[i], bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins, (max_value - min_value) / num_bins))
        y_3, x_3 = np.histogram(results_plot_mip_max_3[i], bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins, (max_value - min_value) / num_bins))
        if i == 3:
            sns.lineplot(x=x_1[:-1], y=y_1, ax = ax[i+4])
            ax[i+4].fill_between(x=x_1[:-1], y1=y_1, y2=0, alpha=0.3, label = "K = 40")
            sns.lineplot(x=x_2[:-1], y=y_2, ax = ax[i+4])
            ax[i+4].fill_between(x=x_2[:-1], y1=y_2, y2=0, alpha=0.3, label = "K = 80")
            sns.lineplot(x=x_3[:-1], y=y_3, ax = ax[i+4])
            ax[i+4].fill_between(x=x_3[:-1], y1=y_3, y2=0, alpha=0.3, label = "K = 120")
            # ax[i+4].legend(fontsize = legend_size)
        else:
            sns.lineplot(x=x_1[:-1], y=y_1, ax = ax[i+4])
            ax[i+4].fill_between(x=x_1[:-1], y1=y_1, y2=0, alpha=0.3)
            sns.lineplot(x=x_2[:-1], y=y_2, ax = ax[i+4])
            ax[i+4].fill_between(x=x_2[:-1], y1=y_2, y2=0, alpha=0.3)
            sns.lineplot(x=x_3[:-1], y=y_3, ax = ax[i+4])
            ax[i+4].fill_between(x=x_3[:-1], y1=y_3, y2=0, alpha=0.3)
        ax[i+4].tick_params("x", labelsize=label_size)
        ax[i+4].tick_params("y", labelsize=label_size)
        ax[i+4].set_ylim(0,35)
        # ax[i].set_ylabel("Frequency", fontsize = font_size)

    ax[0].set_ylabel("Frequency", fontsize = font_size)
    ax[4].set_ylabel("Frequency", fontsize = font_size)

    ax[0].set_title("(a) MIP-Union $C(x^*)$", fontsize = title_size, y=title_position)
    ax[1].set_title("(b) MIP-Union Optimal Value", fontsize = title_size, y=title_position)
    ax[2].set_title("(c) MIP-Union $EC_0$", fontsize = title_size, y=title_position)
    ax[3].set_title("(d) MIP-Union $EC_c$", fontsize = title_size, y=title_position)
    ax[4].set_title("(e) MIP-Max $C(x^*)$", fontsize = title_size, y=title_position)
    ax[5].set_title("(f) MIP-Max Optimal Value", fontsize = title_size, y=title_position)
    ax[6].set_title("(g) MIP-Max $EC_0$", fontsize = title_size, y=title_position)
    ax[7].set_title("(h) MIP-Max $EC_c$", fontsize = title_size, y=title_position)


    fig.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig("case_studies_plots/case_study_4_figure.pdf")

    print(f"Average solving time for mip_union K=40: {np.mean(results_step_1_union['40']['solver_times'])}")
    print(f"Average solving time for mip_union K=80: {np.mean(results_step_1_union['80']['solver_times'])}")
    print(f"Average solving time for mip_union K=120: {np.mean(results_step_1_union['120']['solver_times'])}")

    print(f"Average solving time for mip_max K=40: {np.mean(results_step_1_max['40']['solver_times'])}")
    print(f"Average solving time for mip_max K=80: {np.mean(results_step_1_max['80']['solver_times'])}")
    print(f"Average solving time for mip_max K=120: {np.mean(results_step_1_max['120']['solver_times'])}")


if __name__ == '__main__':
    main()

