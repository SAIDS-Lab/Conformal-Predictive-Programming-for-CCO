"""
In this file, we plot the results for case study 2.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    num_bins = 12
    font_size = 22
    label_size = 17
    legend_size = 18
    title_size = 24
    title_position = -0.2

    with open("case_studies_results/results_case_study_2/results_step_1.json", "r") as file:
        results_step_1 = json.load(file)
    
    with open("case_studies_results/results_case_study_2/results_step_2.json", "r") as file:
        results_step_2 = json.load(file)

    # Report number of timeouts.
    print("Num timeout in kkt:", results_step_1["CPP-KKT"]["num_timeout"])
    print("Num timeout in mip:", results_step_1["CPP-MIP"]["num_timeout"])
    print("Num infeasible in kkt:", results_step_1["CPP-KKT"]["num_infeasible"])
    print("Num infeasible in mip:", results_step_1["CPP-MIP"]["num_infeasible"])

    results_plot_bilevel = [results_step_2["CPP-KKT"]["Cs"], results_step_1["CPP-KKT"]["optimal_values"], results_step_1["CPP-KKT"]["empirical_coverages"], results_step_2["CPP-KKT"]["posterior_coverages"]]
    results_plot_mip = [results_step_2["CPP-MIP"]["Cs"], results_step_1["CPP-MIP"]["optimal_values"], results_step_1["CPP-MIP"]["empirical_coverages"], results_step_2["CPP-MIP"]["posterior_coverages"]]

    #####################################################
    # The followings are the plot for our CPP framework
    fig, ax = plt.subplots(1, 4, figsize=(22, 5))
    ax = ax.flatten()
    for i in range(4):
        min_value = min(min(results_plot_bilevel[i]), min(results_plot_mip[i]))
        max_value = max(max(results_plot_bilevel[i]), max(results_plot_mip[i]))
        y_1, x_1 = np.histogram(results_plot_bilevel[i], bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                            (max_value - min_value) / num_bins))
        y_2, x_2 = np.histogram(results_plot_mip[i], bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                            (max_value - min_value) / num_bins))
        

        if i == 3:
            sns.lineplot(x=x_1[:-1], y=y_1, ax = ax[i])
            ax[i].fill_between(x=x_1[:-1], y1=y_1, y2=0, alpha=0.3, label = "CPP-KKT")
            sns.lineplot(x=x_2[:-1], y=y_2, ax = ax[i])
            ax[i].fill_between(x=x_2[:-1], y1=y_2, y2=0, alpha=0.3, label = "CPP-MIP")
            ax[i].legend(fontsize = legend_size)
        else:
            sns.lineplot(x=x_1[:-1], y=y_1, ax = ax[i])
            ax[i].fill_between(x=x_1[:-1], y1=y_1, y2=0, alpha=0.3)
            sns.lineplot(x=x_2[:-1], y=y_2, ax = ax[i])
            ax[i].fill_between(x=x_2[:-1], y1=y_2, y2=0, alpha=0.3)
        ax[i].tick_params("x", labelsize=label_size)
        ax[i].tick_params("y", labelsize=label_size)
        ax[i].set_ylim(0, 30)
        # ax[i].set_ylabel("Frequency", fontsize = font_size)

    ax[0].set_ylabel("Frequency", fontsize = font_size)
    # ax[4].set_ylabel("Frequency", fontsize = font_size)

    ax[0].set_title("(a) $C(u^*)$", fontsize = title_size, y=title_position)
    ax[1].set_title("(b) $J(u^*)$", fontsize = title_size, y=title_position)
    ax[2].set_title("(c) $EC_{0}$", fontsize = title_size, y=title_position)
    ax[3].set_title("(d) $EC_{C}$", fontsize = title_size, y=title_position)
    # ax[4].set_title("(e) 1 - $\delta_{0}'$ for CPP-KKT", fontsize = title_size, y=title_position)
    # ax[5].set_title("(f) 1 - $\delta_{0}'$ for CPP-MIP", fontsize = title_size, y=title_position)
    # ax[6].set_title("(g) 1 - $\delta_{c}'$ for CPP-KKT", fontsize = title_size, y=title_position)
    # ax[7].set_title("(h) 1 - $\delta_{c}'$ for CPP-MIP", fontsize = title_size, y=title_position)


    fig.tight_layout(rect=[0, 0, 1, 1])
    # plt.subplots_adjust(hspace=0.25)
    plt.savefig("case_studies_plots/case_study_2_figure.pdf")

    print(f"Average solving time for CPP-KKT: {np.mean(results_step_1['CPP-KKT']['solver_times'])}")
    print(f"Average solving time for CPP-MIP: {np.mean(results_step_1['CPP-MIP']['solver_times'])}")

if __name__ == '__main__':
    main()