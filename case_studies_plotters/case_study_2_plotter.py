"""
In this file, we plot the results for case study 2.
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

    with open("case_studies_results/results_case_study_2/results_step_1.json", "r") as file:
        results_step_1 = json.load(file)
    with open("case_studies_results/results_case_study_2/results_step_2.json", "r") as file:
        results_step_2 = json.load(file)

     # Report number of timeouts.
    print("(marginal) Num timeout in kkt:", results_step_1["CPP-KKT_m"]["num_timeout"])
    print("(marginal) Num timeout in mip:", results_step_1["CPP-MIP_m"]["num_timeout"])
    print("(marginal) Num infeasible in kkt:", results_step_1["CPP-KKT_m"]["num_infeasible"])
    print("(marginal) Num infeasible in mip:", results_step_1["CPP-MIP_m"]["num_infeasible"])
    print("(conditional) Num timeout in kkt:", results_step_1["CPP-KKT_c"]["num_timeout"])
    print("(conditional) Num timeout in mip:", results_step_1["CPP-MIP_c"]["num_timeout"])
    print("(conditional) Num infeasible in kkt:", results_step_1["CPP-KKT_c"]["num_infeasible"])
    print("(conditional) Num infeasible in mip:", results_step_1["CPP-MIP_c"]["num_infeasible"])
    print("SOLVER TIME:")
    print(f"Average solving time for CPP-KKT (conditional): {np.mean(results_step_1['CPP-KKT_c']['solver_times'])}")
    print(f"Average solving time for CPP-MIP (conditional): {np.mean(results_step_1['CPP-MIP_c']['solver_times'])}")
    print(f"Average solving time for nonconvex SA: {np.mean(results_step_1['SA']['solver_times'])}")
    print("VALIDATION TIME:")
    print(f"Average delta computation time for CPP-KKT: {np.mean(results_step_2['CPP-KKT']['delta_comp_time'])}")
    print(f"Average delta computation time for CPP-MIP: {np.mean(results_step_2['CPP-MIP']['delta_comp_time'])}")
    print(f"Average support set computation time for nonconvex SA: {np.mean(results_step_2['SA']['sup_comp_time'])}")
    print(f"Average delta1 computation time for nonconvex SA: {np.mean(results_step_2['SA']['delta1_comp_time'])}")
    print(f"Average delta2 computation time for nonconvex SA: {np.mean(results_step_2['SA']['delta2_comp_time'])}")

    
    results_plot_bilevel = [results_step_2["CPP-KKT"]["Cs_m"], results_step_1["CPP-KKT_m"]["optimal_values"], results_step_2["CPP-KKT"]["CEC_c"], results_step_2["CPP-KKT"]["Cs_c"], results_step_1["CPP-KKT_c"]["optimal_values"], results_step_2["CPP-KKT"]["CEC_0_l_z"]]
    results_plot_mip = [results_step_2["CPP-MIP"]["Cs_m"], results_step_1["CPP-MIP_m"]["optimal_values"], results_step_2["CPP-MIP"]["CEC_c"], results_step_2["CPP-MIP"]["Cs_c"], results_step_1["CPP-MIP_c"]["optimal_values"], results_step_2["CPP-MIP"]["CEC_0_l_z"]]

    #####################################################
    # The followings are the plot for our CPP framework
    fig, ax = plt.subplots(2, 3, figsize=(22, 12))
    ax = ax.flatten()
    for i in range(6):
        min_value = min(min(results_plot_bilevel[i]), min(results_plot_mip[i]))
        max_value = max(max(results_plot_bilevel[i]), max(results_plot_mip[i]))
        y_1, x_1 = np.histogram(results_plot_bilevel[i], bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                            (max_value - min_value) / num_bins))
        y_2, x_2 = np.histogram(results_plot_mip[i], bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                            (max_value - min_value) / num_bins))
        
        if i == 0:
            sns.lineplot(x=x_1[:-1], y=y_1, ax = ax[i])
            ax[i].fill_between(x=x_1[:-1], y1=y_1, y2=0, alpha=0.3, label = "CPP-KKT")
            sns.lineplot(x=x_2[:-1], y=y_2, ax = ax[i])
            ax[i].fill_between(x=x_2[:-1], y1=y_2, y2=0, alpha=0.3, label = "CPP-MIP")
            ax[i].legend(fontsize = legend_size, loc = "upper right")
        else:
            sns.lineplot(x=x_1[:-1], y=y_1, ax = ax[i])
            ax[i].fill_between(x=x_1[:-1], y1=y_1, y2=0, alpha=0.3)
            sns.lineplot(x=x_2[:-1], y=y_2, ax = ax[i])
            ax[i].fill_between(x=x_2[:-1], y1=y_2, y2=0, alpha=0.3)
        ax[i].tick_params("x", labelsize=label_size)
        ax[i].tick_params("y", labelsize=label_size)
        ax[i].set_ylim(0, 100)


    ax[1].xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax[4].xaxis.set_major_locator(MaxNLocator(nbins=5))

    ax[0].set_ylabel("Frequency", fontsize = font_size)
    ax[3].set_ylabel("Frequency", fontsize = font_size)

    ax[0].set_title("(a) $C_m(x^*_m)$", fontsize = title_size, y=title_position)
    ax[1].set_title("(b) $J(x^*_m)$", fontsize = title_size, y=title_position)
    ax[2].set_title("(c) $CEC_{c, l}$", fontsize = title_size, y=title_position)
    ax[3].set_title("(d) $C_c(x^*_c)$", fontsize = title_size, y=title_position)
    ax[4].set_title("(e) $J(x^*_c)$", fontsize = title_size, y=title_position)
    ax[5].set_title("(f) $CEC_{0, l', z}$", fontsize = title_size, y=title_position)

    fig.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig("case_studies_plots/case_study_2_figure.pdf")


    ########## delta_star histogram ##########
    num_bins = 40
    delta_star_kkt = results_step_2["CPP-KKT"]["delta_star"]
    delta_star_mip = results_step_2["CPP-MIP"]["delta_star"]
    delta_star_tac = results_step_2["SA"]["delta_star_1"]  
    delta_star_mp = results_step_2["SA"]["delta_star_2"]  

    fig = plt.figure(figsize=(7, 6))
    min_value = min(min(delta_star_kkt), min(delta_star_mip), min(delta_star_tac), min(delta_star_mp))
    max_value = max(max(delta_star_kkt), max(delta_star_mip), max(delta_star_tac), max(delta_star_mp))
    y_1, x_1 = np.histogram(delta_star_kkt, bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                        (max_value - min_value) / num_bins))
    y_2, x_2 = np.histogram(delta_star_mip, bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                        (max_value - min_value) / num_bins))
    y_3, x_3 = np.histogram(delta_star_tac, bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                        (max_value - min_value) / num_bins))
    y_4, x_4 = np.histogram(delta_star_mp, bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                        (max_value - min_value) / num_bins))
    
    sns.lineplot(x=x_1[:-1], y=y_1)
    plt.fill_between(x=x_1[:-1], y1=y_1, y2=0, alpha=0.3, label = "CPP-KKT")
    sns.lineplot(x=x_2[:-1], y=y_2)
    plt.fill_between(x=x_2[:-1], y1=y_2, y2=0, alpha=0.3, label = "CPP-MIP")
    sns.lineplot(x=x_3[:-1], y=y_3)
    plt.fill_between(x=x_3[:-1], y1=y_3, y2=0, alpha=0.3, label = "[18]")
    sns.lineplot(x=x_4[:-1], y=y_4)
    plt.fill_between(x=x_4[:-1], y1=y_4, y2=0, alpha=0.3, label = "[33]")
    plt.legend(fontsize = legend_size, loc = "upper right")
    plt.tick_params("x", labelsize=label_size)
    plt.tick_params("y", labelsize=label_size)

    # Here, I set the xlim since there is one outlier with delta_star being 0.961, which is caused by the numerical issue. 
    plt.xlim(0, 0.25)
    plt.ylabel("Frequency", fontsize = font_size)
    plt.title("$\delta^*$", fontsize = font_size, y=title_position)
    fig.tight_layout()
    plt.savefig("case_studies_plots/case_study_2_figure_delta.pdf")


if __name__ == "__main__":
    main()