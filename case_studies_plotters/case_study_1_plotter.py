"""
In this file, we plot the results for case study 1.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches

def main():
    num_bins = 20
    font_size = 22
    label_size = 17
    legend_size = 18
    title_size = 24
    title_position = -0.2
    height = 140

    plot_1_K = "500"
    plot_2_Ks = ["300", "50"]
    plot_1_Ls = ["50", "200", "1000"]
    plot_2_Ls = ["50", "200", "1000"]
    # Read the data from the json files.
    with open("case_studies_results/results_case_study_1/results_step_1_case_1_K=50.json", "r") as file:
        results_K_50 = json.load(file)
    with open("case_studies_results/results_case_study_1/results_step_1_case_1_K=100.json", "r") as file:
        results_K_100 = json.load(file)
    with open("case_studies_results/results_case_study_1/results_step_1_case_1_K=200.json", "r") as file:
        results_K_200 = json.load(file)
    with open("case_studies_results/results_case_study_1/results_step_1_case_1_K=300.json", "r") as file:
        results_K_300 = json.load(file)
    with open("case_studies_results/results_case_study_1/results_step_1_case_1_K=500.json", "r") as file:
        results_K_500 = json.load(file)
    with open("case_studies_results/results_case_study_1/results_step_2_case_1.json", "r") as file:
        a_posteriori_results = json.load(file)

    fig, ax = plt.subplots(2, 4, figsize=(22, 10))
    ax = ax.flatten()
    new_min = -float("inf")
    new_max = float("inf")
    y_max = 0
    for category in ["CPP-KKT", "CPP-MIP"]:
        min_value = min(min(results_K_50[category]["optimal_values"]),
                        min(results_K_300[category]["optimal_values"]),
                        min(results_K_500[category]["optimal_values"]))
        max_value = max(max(results_K_50[category]["optimal_values"]),
                        max(results_K_300[category]["optimal_values"]),
                        max(results_K_500[category]["optimal_values"]))
        y_1, x_1 = np.histogram(results_K_50[category]["optimal_values"],
                                bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                               (max_value - min_value) / num_bins))
        y_2, x_2 = np.histogram(results_K_300[category]["optimal_values"],
                                bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                               (max_value - min_value) / num_bins))
        y_3, x_3 = np.histogram(results_K_500[category]["optimal_values"],
                                bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                               (max_value - min_value) / num_bins))

        y_max = max(y_max, max(y_1), max(y_2), max(y_3))
        index = int(category == "CPP-MIP")
        sns.lineplot(x=x_1[:-1], y=y_1, ax=ax[index])
        ax[index].fill_between(x=x_1[:-1], y1=y_1, y2=0, alpha=0.3, label="K=50")
        sns.lineplot(x=x_2[:-1], y=y_2, ax=ax[index])
        ax[index].fill_between(x=x_2[:-1], y1=y_2, y2=0, alpha=0.3, label="K=300")
        sns.lineplot(x=x_3[:-1], y=y_3, ax=ax[index])
        ax[index].fill_between(x=x_3[:-1], y1=y_3, y2=0, alpha=0.3, label="K=500")
        ax[index].legend(fontsize=legend_size, loc='upper right')
        ax[index].tick_params("x", labelsize=label_size)
        ax[index].tick_params("y", labelsize=label_size)
        if category == "CPP-KKT":
            ax[index].set_title("(a) $J(x^*)$ for CPP-KKT", fontsize=title_size, y=title_position)
        else:
            ax[index].set_title("(b) $J(x^*)$ for CPP-MIP", fontsize=title_size, y=title_position)
        new_min = min(new_min, min_value)
        new_max = max(new_max, max_value)
    ax[0].set_xlim(min_value, max_value)
    ax[1].set_xlim(min_value, max_value)
    ax[0].set_ylim(0, height)
    ax[1].set_ylim(0, height)

    new_min = float("inf")
    new_max = -float("inf")
    y_max = 0
    for category in ["CPP-KKT", "CPP-MIP"]:
        min_value = min(min(results_K_50[category]["empirical_coverages"]), min(results_K_300[category]["empirical_coverages"]), min(results_K_500[category]["empirical_coverages"]))
        max_value = max(max(results_K_50[category]["empirical_coverages"]), max(results_K_300[category]["empirical_coverages"]), max(results_K_500[category]["empirical_coverages"]))
        y_1, x_1 = np.histogram(results_K_50[category]["empirical_coverages"], bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                                (max_value - min_value) / num_bins))
        y_2, x_2 = np.histogram(results_K_300[category]["empirical_coverages"], bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                                (max_value - min_value) / num_bins))
        y_3, x_3 = np.histogram(results_K_500[category]["empirical_coverages"], bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                                (max_value - min_value) / num_bins))

        y_max = max(y_max, max(y_1), max(y_2), max(y_3))
        index = int(category == "CPP-MIP") + 2
        sns.lineplot(x=x_1[:-1], y=y_1, ax = ax[index])
        ax[index].fill_between(x=x_1[:-1], y1=y_1, y2=0, alpha=0.3, label="K=50")
        sns.lineplot(x=x_2[:-1], y=y_2, ax = ax[index])
        ax[index].fill_between(x=x_2[:-1], y1=y_2, y2=0, alpha=0.3, label="K=300")
        sns.lineplot(x=x_3[:-1], y=y_3, ax = ax[index])
        ax[index].fill_between(x=x_3[:-1], y1=y_3, y2=0, alpha=0.3, label="K=500")
        ax[index].legend(fontsize=legend_size, loc='upper right')
        ax[index].tick_params("x", labelsize=label_size)
        ax[index].tick_params("y", labelsize=label_size)
        if category == "CPP-KKT":
            ax[index].set_title("(c) $EC_{0}$ for CPP-KKT", fontsize=title_size, y=title_position)
        else:
            ax[index].set_title("(d) $EC_{0}$ for CPP-MIP", fontsize=title_size, y=title_position)
        new_min = min(new_min, min_value)
        new_max = max(new_max, max_value)
    ax[2].set_xlim(new_min, new_max)
    ax[3].set_xlim(new_min, new_max)
    ax[2].set_ylim(0, height)
    ax[3].set_ylim(0, height)

    for type in ["Cs", "posterior_coverages"]:
        new_min = float("inf")
        new_max = -float("inf")
        y_max = 0
        for category in ["CPP-KKT", "CPP-MIP"]:
            data_1 = a_posteriori_results[plot_1_K][plot_1_Ls[0]][category][type]
            data_2 = a_posteriori_results[plot_1_K][plot_1_Ls[1]][category][type]
            data_3 = a_posteriori_results[plot_1_K][plot_1_Ls[2]][category][type]
            min_value = min(min(data_1), min(data_2), min(data_3))
            max_value = max(max(data_1), max(data_2), max(data_3))
            new_min = min(new_min, min_value)
            new_max = max(new_max, max_value)
            y_1, x_1 = np.histogram(data_1, bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                                    (max_value - min_value) / num_bins))
            y_2, x_2 = np.histogram(data_2, bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                                    (max_value - min_value) / num_bins))
            y_3, x_3 = np.histogram(data_3, bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                                    (max_value - min_value) / num_bins))

            y_max = max(y_max, max(y_1), max(y_2), max(y_3))
            if type == "Cs":
                index = int(category == "CPP-MIP") + 4
            else:
                index = int(category == "CPP-MIP") + 6
            sns.lineplot(x=x_1[:-1], y=y_1, ax = ax[index])
            ax[index].fill_between(x=x_1[:-1], y1=y_1, y2=0, alpha=0.3, label=f"K = {plot_1_K}, L={plot_1_Ls[0]}")
            sns.lineplot(x=x_2[:-1], y=y_2, ax = ax[index])
            ax[index].fill_between(x=x_2[:-1], y1=y_2, y2=0, alpha=0.3, label=f"K = {plot_1_K}, L={plot_1_Ls[1]}")
            sns.lineplot(x=x_3[:-1], y=y_3, ax = ax[index])
            ax[index].fill_between(x=x_3[:-1], y1=y_3, y2=0, alpha=0.3, label=f"K = {plot_1_K}, L={plot_1_Ls[2]}")
            ax[index].legend(fontsize=legend_size, loc='upper right')
            ax[index].tick_params("x", labelsize=label_size)
            ax[index].tick_params("y", labelsize=label_size)
            if type == "Cs":
                if category == "CPP-KKT":
                    ax[index].set_title("(e) $C(x^*)$ for CPP-KKT", fontsize=title_size, y=title_position)
                else:
                    ax[index].set_title("(f) $C(x^*)$ for CPP-MIP", fontsize=title_size, y=title_position)
            else:
                if category == "CPP-KKT":
                    ax[index].set_title("(g) $EC_{C}$ for CPP-KKT", fontsize=title_size, y=title_position)
                else:
                    ax[index].set_title("(h) $EC_{C}$ for CPP-MIP", fontsize=title_size, y=title_position)
        if type == "Cs":
            ax[4].set_xlim(new_min, new_max)
            ax[5].set_xlim(new_min, new_max)
            ax[4].set_ylim(0, height)
            ax[5].set_ylim(0, height)
        else:
            ax[6].set_xlim(new_min, new_max)
            ax[7].set_xlim(new_min, new_max)
            ax[6].set_ylim(0, height)
            ax[7].set_ylim(0, height)

    ax[0].set_ylabel("Frequency", fontsize=font_size)
    ax[4].set_ylabel("Frequency", fontsize=font_size)
    fig.tight_layout(rect=[0, 0, 1, 1])
    plt.subplots_adjust(hspace=0.25)
    plt.savefig("case_studies_plots/case_study_1_figure_1.pdf")
    plt.show()

    # Generate the second plot.
    fig, ax = plt.subplots(2, 4, figsize=(22, 10))
    ax = ax.flatten()
    for i in range(len(plot_2_Ks)):
        k = plot_2_Ks[i]
        for type in ["Cs", "posterior_coverages"]:
            new_min = float("inf")
            new_max = -float("inf")
            y_max = 0
            for category in ["CPP-KKT", "CPP-MIP"]:
                data_1 = a_posteriori_results[k][plot_2_Ls[0]][category][type]
                data_2 = a_posteriori_results[k][plot_2_Ls[1]][category][type]
                data_3 = a_posteriori_results[k][plot_2_Ls[2]][category][type]
                min_value = min(min(data_1), min(data_2), min(data_3))
                max_value = max(max(data_1), max(data_2), max(data_3))
                new_min = min(new_min, min_value)
                new_max = max(new_max, max_value)
                y_1, x_1 = np.histogram(data_1,
                                        bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                                       (max_value - min_value) / num_bins))
                y_2, x_2 = np.histogram(data_2,
                                        bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                                       (max_value - min_value) / num_bins))
                y_3, x_3 = np.histogram(data_3,
                                        bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                                       (max_value - min_value) / num_bins))
                
                y_max = max(y_max, max(y_1), max(y_2), max(y_3))
                if i == 0:
                    if type == "Cs":
                        index = int(category == "CPP-MIP")
                    else:
                        index = int(category == "CPP-MIP") + 2
                else:
                    if type == "Cs":
                        index = int(category == "CPP-MIP") + 4
                    else:
                        index = int(category == "CPP-MIP") + 6
                sns.lineplot(x=x_1[:-1], y=y_1, ax=ax[index])
                ax[index].fill_between(x=x_1[:-1], y1=y_1, y2=0, alpha=0.3, label=f"K = {k}, L={plot_2_Ls[0]}")
                sns.lineplot(x=x_2[:-1], y=y_2, ax=ax[index])
                ax[index].fill_between(x=x_2[:-1], y1=y_2, y2=0, alpha=0.3, label=f"K = {k}, L={plot_2_Ls[1]}")
                sns.lineplot(x=x_3[:-1], y=y_3, ax=ax[index])
                ax[index].fill_between(x=x_3[:-1], y1=y_3, y2=0, alpha=0.3, label=f"K = {k}, L={plot_2_Ls[2]}")
                ax[index].legend(fontsize=legend_size, loc='upper right')
                ax[index].tick_params("x", labelsize=label_size)
                ax[index].tick_params("y", labelsize=label_size)
                if i == 0:
                    if type == "Cs":
                        if category == "CPP-KKT":
                            ax[index].set_title("(a) $C(x^*)$ for CPP-KKT", fontsize=title_size, y=title_position)
                        else:
                            ax[index].set_title("(b) $C(x^*)$ for CPP-MIP", fontsize=title_size, y=title_position)
                    else:
                        if category == "CPP-KKT":
                            ax[index].set_title("(c) $EC_{C}$ for CPP-KKT", fontsize=title_size, y=title_position)
                        else:
                            ax[index].set_title("(d) $EC_{C}$ for CPP-MIP", fontsize=title_size, y=title_position)
                else:
                    if type == "Cs":
                        if category == "CPP-KKT":
                            ax[index].set_title("(e) $C(x^*)$ for CPP-KKT", fontsize=title_size, y=title_position)
                        else:
                            ax[index].set_title("(f) $C(x^*)$ for CPP-MIP", fontsize=title_size, y=title_position)
                    else:
                        if category == "CPP-KKT":
                            ax[index].set_title("(g) $EC_{C}$ for CPP-KKT", fontsize=title_size, y=title_position)
                        else:
                            ax[index].set_title("(h) $EC_{C}$ for CPP-MIP", fontsize=title_size, y=title_position)
            ax[index].tick_params("x", labelsize=label_size)
            ax[index].tick_params("y", labelsize=label_size)
            ax[index - 1].tick_params("x", labelsize=label_size)
            ax[index - 1].tick_params("y", labelsize=label_size)
            ax[index].set_xlim(new_min, new_max)
            ax[index - 1].set_xlim(new_min, new_max)
            ax[index].set_ylim(0, height)
            ax[index - 1].set_ylim(0, height)
    
    # these two are used to make the same xlim (please write it in a better way in the final version)
    ax[0].set_xlim(-4, 10)
    ax[1].set_xlim(-4, 10)
    ax[4].set_xlim(-4, 10)
    ax[5].set_xlim(-4, 10)

    ax[0].set_ylabel("Frequency", fontsize=font_size)
    ax[4].set_ylabel("Frequency", fontsize=font_size)
    fig.tight_layout(rect=[0, 0, 1, 1])
    plt.subplots_adjust(hspace=0.25)
    plt.savefig("case_studies_plots/case_study_1_figure_2.pdf")
    plt.show()

    # Generate the table.
    print("\\begin{table}[h] ")
    print("\\renewcommand{\\arraystretch}{1.5}")
    print("\scriptsize")
    print("\\begin{center}")
    print("\caption{Average Statistics with $N = 300$ and $V = 1000$ with Varying $K$ and $L$ for CPP-KKT and CPP-MIP}")
    print("\\label{table:average_statistics}")
    print("\\begin{tabular}{ccccccccc}")
    print("\\toprule")
    print(" & &C(x^*) & J(x^*) & 1 - \delta'_{0} & 1 - \delta'_{c} & Solver Time (s) & No. Timeouts & No. Infeasible Solutions\\\\")
    print("\\midrule")
    for k in ["50", "100", "200", "300", "500"]:
        for l in ["50", "200", "500", "750", "1000"]:
            with open(f"case_studies_results/results_case_study_1/results_step_1_case_1_K={k}.json", "r") as file:
                a_prior_results = json.load(file)

            # Statistics for KKT.
            average_Cs_kkt = round(np.average(a_posteriori_results[k][l]["CPP-KKT"]["Cs"]), 3)
            average_Js_kkt = round(np.average(a_prior_results["CPP-KKT"]["optimal_values"]), 3)
            average_empirical_coverages_kkt = round(np.average(a_prior_results["CPP-KKT"]["empirical_coverages"]), 3)
            average_posteriori_coverages_kkt = round(np.average(a_posteriori_results[k][l]["CPP-KKT"]["posterior_coverages"]), 3)
            average_solver_times_kkt = round(np.average(a_prior_results["CPP-KKT"]["solver_times"]), 3)
            num_timeouts_kkt = a_prior_results["CPP-KKT"]["num_timeout"]
            num_infeasible_kkt = a_prior_results["CPP-KKT"]["num_infeasible"]

            # Statistics for MIP.
            average_Cs_mip = round(np.average(a_posteriori_results[k][l]["CPP-MIP"]["Cs"]), 3)
            average_Js_mip = round(np.average(a_prior_results["CPP-MIP"]["optimal_values"]), 3)
            average_empirical_coverages_mip = round(np.average(a_prior_results["CPP-MIP"]["empirical_coverages"]), 3)
            average_posteriori_coverages_mip = round(np.average(a_posteriori_results[k][l]["CPP-MIP"]["posterior_coverages"]), 3)
            average_solver_times_mip = round(np.average(a_prior_results["CPP-MIP"]["solver_times"]), 3)
            num_timeouts_mip = a_prior_results["CPP-MIP"]["num_timeout"]
            num_infeasible_mip = a_prior_results["CPP-MIP"]["num_infeasible"]

            print("\\multirow{{2}}{{*}}{{\\textbf{{K = {}, L = {}}}}} & CPP-KKT & {} & {} & {} & {} & {} & {} & {}\\\\".format(k, l, average_Cs_kkt, average_Js_kkt, average_empirical_coverages_kkt, average_posteriori_coverages_kkt, average_solver_times_kkt, num_timeouts_kkt, num_infeasible_kkt))
            print("& CPP-MIP & {} & {} & {} & {} & {} & {} & {}\\\\ \hline".format(average_Cs_mip, average_Js_mip, average_empirical_coverages_mip, average_posteriori_coverages_mip, average_solver_times_mip, num_timeouts_mip, num_infeasible_mip))
    print("\\bottomrule")
    print("\end{tabular}")
    print("\end{center}")
    print("\end{table}")

    # Generate comparison plots.
    results = dict()
    results["50"] = results_K_50
    results["100"] = results_K_100
    results["300"] = results_K_300
    results["500"] = results_K_500

    fig, ax = plt.subplots(2, 4, figsize=(22, 10))
    ax = ax.flatten()
    # First, show the costs of different methods in the same plot.
    categories = ["CPP-KKT", "CPP-MIP", "SAA_1", "SAA_2"]
    figure_lable_catagory_name = [f"CPP-KKT", f"CPP-MIP", f"SAA, $\omega$ = 0.01", f"SAA, $\omega$ = 0.03"]
    index = 0
    x_title_list = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h"}
    new_min = float("inf")
    new_max = -float("inf")
    y_max = 0
    for k in ["50", "100", "300", "500"]:
        for category in categories:
            min_value = min(results[k][category]["optimal_values"])
            max_value = max(results[k][category]["optimal_values"])
            new_min = min(new_min, min_value)
            new_max = max(new_max, max_value)
            y, x = np.histogram(results[k][category]["optimal_values"], bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                            (max_value - min_value) / num_bins))

            y_max = max(y_max, max(y))
            sns.lineplot(x=x[:-1], y=y, ax = ax[index])
            ax[index].fill_between(x=x[:-1], y1=y, y2=0, alpha=0.3)
        # ax[index].legend(fontsize=legend_size, loc='upper right')
        ax[index].tick_params("x", labelsize=label_size)
        ax[index].tick_params("y", labelsize=label_size)
        ax[index].set_ylabel("Frequency", fontsize=font_size)
        ax[index].set_xlabel(f"({x_title_list[index]}) $J(x^*), K = {k}$", fontsize=title_size, y=title_position)
        index += 1
    
    for i in range(4):
        ax[i].set_ylim(0, y_max + 6)
        ax[i].set_xlim(new_min, new_max)

    # Now, do the same thing for the empirical coverages.
    new_min = float("inf")
    new_max = -float("inf")
    y_max = 0
    for k in ["50", "100", "300", "500"]:
        for category in categories:
            min_value = min(results[k][category]["empirical_coverages"])
            max_value = max(results[k][category]["empirical_coverages"])
            new_min = min(new_min, min_value)
            new_max = max(new_max, max_value)
            y, x = np.histogram(results[k][category]["empirical_coverages"], bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                            (max_value - min_value) / num_bins))
        
            y_max = max(y_max, max(y))
            sns.lineplot(x=x[:-1], y=y, ax = ax[index])
            if index == 7:
                ax[index].fill_between(x=x[:-1], y1=y, y2=0, alpha=0.3, label=figure_lable_catagory_name[categories.index(category)])
            else:
                ax[index].fill_between(x=x[:-1], y1=y, y2=0, alpha=0.3)
        # ax[index].legend(fontsize=legend_size, loc='upper left')
        ax[index].tick_params("x", labelsize=label_size)
        ax[index].tick_params("y", labelsize=label_size)
        ax[index].set_ylabel("Frequency", fontsize=font_size)
        ax[index].set_xlabel(f"({x_title_list[index]}) $EC_{0}, K = {k}$", fontsize=title_size, y=title_position)
        index += 1
    for i in range(4, 8):
        ax[i].set_ylim(0, y_max + 6)
        ax[i].set_xlim(new_min, new_max)
    fig.tight_layout(rect=[0, 0, 1, 1])
    plt.subplots_adjust(hspace=0.25)
    ax[7].legend(fontsize=legend_size, loc='upper left')
    plt.savefig("case_studies_plots/case_study_1_figure_3.pdf")
    plt.show()

    # Do the same thing for SA.
    fig, ax = plt.subplots(2, 4, figsize=(22, 10))
    ax = ax.flatten()
    # First, show the costs of different methods in the same plot.
    categories = ["CPP-KKT", "CPP-MIP", "SA"]
    figure_lable_catagory_name = [f"CPP-KKT", f"CPP-MIP", f"SA"]
    index = 0
    new_min = float("inf")
    new_max = -float("inf")
    y_max = 0
    for k in ["50", "100", "300", "500"]:
        for category in categories:
            min_value = min(results[k][category]["optimal_values"])
            max_value = max(results[k][category]["optimal_values"])
            new_min = min(new_min, min_value)
            new_max = max(new_max, max_value)
            y, x = np.histogram(results[k][category]["optimal_values"],
                                bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                               (max_value - min_value) / num_bins))
            y_max = max(y_max, max(y))
            sns.lineplot(x=x[:-1], y=y, ax=ax[index])
            ax[index].fill_between(x=x[:-1], y1=y, y2=0, alpha=0.3)
        # ax[index].legend(fontsize=legend_size, loc='upper right')
        ax[index].tick_params("x", labelsize=label_size)
        ax[index].tick_params("y", labelsize=label_size)
        ax[index].set_ylabel("Frequency", fontsize=font_size)
        ax[index].set_xlabel(f"({x_title_list[index]}) $J(x^*), K = {k}$", fontsize=title_size, y=title_position)
        index += 1
    for i in range(4):
        ax[i].set_ylim(0, y_max + 6)
        ax[i].set_xlim(new_min, new_max)

    # Now, do the same thing for the empirical coverages.
    new_min = float("inf")
    new_max = -float("inf")
    y_max = 0
    for k in ["50", "100", "300", "500"]:
        for category in categories:
            min_value = min(results[k][category]["empirical_coverages"])
            max_value = max(results[k][category]["empirical_coverages"])
            new_min = min(new_min, min_value)
            new_max = max(new_max, max_value)
            y, x = np.histogram(results[k][category]["empirical_coverages"],
                                bins=np.arange(min_value, max_value + (max_value - min_value) / num_bins,
                                               (max_value - min_value) / num_bins))
            y_max = max(y_max, max(y))
            sns.lineplot(x=x[:-1], y=y, ax=ax[index])
            if index == 7:
                ax[index].fill_between(x=x[:-1], y1=y, y2=0, alpha=0.3, label=figure_lable_catagory_name[categories.index(category)])
            else:
                ax[index].fill_between(x=x[:-1], y1=y, y2=0, alpha=0.3)
        # ax[index].legend(fontsize=legend_size, loc='upper left')
        ax[index].tick_params("x", labelsize=label_size)
        ax[index].tick_params("y", labelsize=label_size)
        ax[index].set_ylabel("Frequency", fontsize=font_size)
        ax[index].set_xlabel(f"({x_title_list[index]}) $EC_{0}, K = {k}$", fontsize=title_size, y=title_position)
        index += 1
    ax[7].legend(fontsize=legend_size, loc='upper left')
    for i in range(4, 8):
        ax[i].set_ylim(0, y_max + 6)
        ax[i].set_xlim(new_min, new_max)
    fig.tight_layout(rect=[0, 0, 1, 1])
    plt.subplots_adjust(hspace=0.25)
    plt.savefig("case_studies_plots/case_study_1_figure_4.pdf")
    plt.show()


if __name__ == "__main__":
    main()