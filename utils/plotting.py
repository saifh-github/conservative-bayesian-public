import matplotlib.pyplot as plt
import torch as t
import numpy as np


def plot_deaths_and_reward_vs_alpha(
    results, plot_error_bars=True, save_path=None, save_format="pdf", return_fig=False, include_custom_metric=False
):
    n_plots = len(results["posterior"])
    guardrails_excluding_non_iid = ["iid", "posterior", "cheating"]
    colors = {
        "iid": "green",
        "posterior": "blue",
        "cheating": "purple",
        "non-iid": "orange",
        "new-non-iid": "magenta",
    }
    error_bar_positions = {
        "iid": 0.3,
        "posterior": 0.6,
        "cheating": 0.9,
    }

    num_col = 3 if include_custom_metric else 2

    fig, axes = plt.subplots(n_plots, num_col, figsize=(21, 5 * n_plots), squeeze=False)

    def format_to_1sf(x, pos):
        return f"{x:.1g}"

    metrics_list = ["reward", "deaths"]
    if include_custom_metric:
        metrics_list.append("custom_metric")

    for i, guardrail_threshold in enumerate([res[0] for res in results["posterior"]]):
        for j, metric in enumerate(metrics_list):
            ax = axes[i, j]

            # Plot non-iid data
            for guardrail_name in ["non-iid", "new-non-iid"]:
                alphas = sorted(results[guardrail_name].keys())
                non_iid_data = [
                    next(
                        res
                        for res in results[guardrail_name][alpha]
                        if res[0] == guardrail_threshold
                    )
                    for alpha in alphas
                ]

                if metric == "reward":
                    y = [res[1] for res in non_iid_data]
                    y_err = [res[2] for res in non_iid_data]
                elif metric == "deaths":
                    y = [res[3] for res in non_iid_data]
                    y_err = [res[4] for res in non_iid_data]
                elif metric == "custom_metric":
                    y = [res[6] for res in non_iid_data]
                    y_err = None

                # Use range(len(alphas)) for x-axis to space points evenly
                ax.errorbar(
                    range(len(alphas)),
                    y,
                    yerr=y_err,
                    fmt="-o",
                    color=colors[guardrail_name],
                    label="Prop 4.6" + (" (new)" if guardrail_name == "new-non-iid" else ""),
                    capsize=5,
                )

            # Plot other guardrails as dashed lines
            labels = ["Prop 3.4", "Posterior", "Cheating"]
            for k, guardrail in enumerate(guardrails_excluding_non_iid):
                data = next(res for res in results[guardrail] if res[0] == guardrail_threshold)

                if metric == "reward":
                    value = data[1]
                    error = data[2]
                elif metric == "deaths":
                    value = data[3]
                    error = data[4]
                else:
                    value = data[6]
                    error = None

                ax.axhline(
                    y=value, color=colors[guardrail], linestyle="--", label=labels[k]
                )
                if plot_error_bars:
                    ax.errorbar(
                        error_bar_positions[guardrail],
                        value,
                        yerr=error,
                        fmt="none",
                        ecolor=colors[guardrail],
                        capsize=5,
                    )

            ax.set_xlabel("Alpha")
            if metric == "reward":
                ax.set_ylabel("Reward")
            elif metric == "deaths":
                ax.set_ylabel("Deaths")
            elif metric == "custom_metric":
                ax.set_ylabel("Custom Metric")
            ax.set_title(f"{metric.capitalize()} vs Alpha (C = {guardrail_threshold})")
            ax.legend()

            # Remove gridlines
            ax.grid(False)

            # Set x-ticks to match non-iid alpha values and format to 1 significant figure
            ax.set_xticks(range(len(alphas)))
            ax.set_xticklabels([format_to_1sf(alpha, None) for alpha in alphas])
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    if return_fig:
        return fig
    if save_path:
        plt.savefig(save_path, format=save_format, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def fig_deaths_reward_custom_metric_vs_alpha_at_threshold(
    results, guardrail_threshold, plot_error_bars=True
):
    n_plots = len(results["posterior"])
    guardrails_excluding_non_iid = ["iid", "posterior", "cheating"]
    colors = {
        "iid": "green",
        "posterior": "blue",
        "cheating": "purple",
        "non-iid": "orange",
        "new-non-iid": "magenta",
    }
    error_bar_positions = {
        "iid": 0.3,
        "posterior": 0.6,
        "cheating": 0.9,
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5 * n_plots), squeeze=False)

    def format_to_1sf(x, pos):
        return f"{x:.1g}"

    for j, metric in enumerate(["reward", "deaths", "custom_metric"]):
        ax = axes[0, j]

        # Plot non-iid data
        for guardrail_name in ["non-iid", "new-non-iid"]:
            alphas = sorted(results[guardrail_name].keys())
            non_iid_data = [
                next(
                    res
                    for res in results[guardrail_name][alpha]
                    if res[0] == guardrail_threshold
                )
                for alpha in alphas
            ]

            if metric == "reward":
                y = [res[1] for res in non_iid_data]
                y_err = [res[2] for res in non_iid_data]
            elif metric == "deaths":
                y = [res[3] for res in non_iid_data]
                y_err = [res[4] for res in non_iid_data]
            elif metric == "custom_metric":
                y = [res[6] for res in non_iid_data]
                y_err = None

            # Use range(len(alphas)) for x-axis to space points evenly
            ax.errorbar(
                range(len(alphas)),
                y,
                yerr=y_err,
                fmt="-o",
                color=colors[guardrail_name],
                label="Prop 4.6" + (" (new)" if guardrail_name == "new-non-iid" else ""),
                capsize=5,
            )

        # Plot other guardrails as dashed lines
        labels = ["Prop 3.4", "Posterior", "Cheating"]
        for k, guardrail in enumerate(guardrails_excluding_non_iid):
            data = next(res for res in results[guardrail] if res[0] == guardrail_threshold)
            if metric == "reward":
                value = data[1]
                error = data[2]
            elif metric == "deaths":
                value = data[3]
                error = data[4]
            else:
                value = data[6]
                error = None

            ax.axhline(
                y=value, color=colors[guardrail], linestyle="--", label=labels[k]
            )
            if plot_error_bars:
                ax.errorbar(
                    error_bar_positions[guardrail],
                    value,
                    yerr=error,
                    fmt="none",
                    ecolor=colors[guardrail],
                    capsize=5,
                )

        ax.set_xlabel("Alpha")
        if metric == "reward":
            ax.set_ylabel("Reward")
        elif metric == "deaths":
            ax.set_ylabel("Deaths")
        elif metric == "custom_metric":
            ax.set_ylabel("Custom Metric")
        ax.set_title(f"{metric.capitalize()} vs Alpha (C = {guardrail_threshold})")
        ax.legend()

        # Remove gridlines
        ax.grid(False)

        # Set x-ticks to match non-iid alpha values and format to 1 significant figure
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([format_to_1sf(alpha, None) for alpha in alphas])
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    return fig

def plot_overestimation(
    results, plot_error_bars=True, save_path=None, save_format="pdf"
):
    args = results["args"]
    alphas = args.alphas
    overestimates = results["overestimates"]
    overestimate_error = results["overestimate error"]
    p = 1 / (args.k**args.d_arm)
    theoretical_lower_bound = [1 - (alpha / p) for alpha in alphas]
    theoretical_lower_bound = [max(0, i) for i in theoretical_lower_bound]

    fig, ax = plt.subplots(figsize=(10, 6))

    def format_to_1sf(x, pos):
        return f"{x:.1g}"

    # Plot overestimation data
    y = overestimates
    y_err = overestimate_error if plot_error_bars else None

    ax.errorbar(
        range(len(alphas)),
        y,
        yerr=y_err,
        fmt="-o",
        color="blue",
        capsize=5,
    )
    ax.plot(
        range(len(alphas)),
        theoretical_lower_bound,
        "r--",
        label="Theoretical lower bound",
    )

    ax.set_xlabel("Alpha")
    ax.set_ylabel("Overestimation Frequency")

    # Remove gridlines
    ax.grid(False)

    # Set x-ticks to match alpha values and format to 1 significant figure
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([format_to_1sf(alpha, None) for alpha in alphas])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Use scientific notation for y-axis if values are very small
    ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format=save_format, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def violin_plot(results, save_path=None, save_format="pdf"):
    args = results["args"]
    alphas = args.alphas
    harm_estimates = results["harm estimates"]
    print(type(harm_estimates[0]))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.violinplot(harm_estimates, showmeans=True)
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Harm Estimate")
    ax.axhline(y=0.5, color="red", linestyle="--", label="ground truth")

    # Set x-ticks to match alpha values
    ax.set_xticks(range(1, len(alphas) + 1))
    ax.set_xticklabels([f"{alpha:.1g}" for alpha in alphas])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format=save_format, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def box_plot(results, save_path=None, save_format="pdf"):
    args = results["args"]
    alphas = args.alphas
    harm_estimates = results["harm estimates"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create box plot without outliers
    bp = ax.boxplot(harm_estimates, patch_artist=True, showfliers=False, whis=[0, 100])

    # Customize box plot
    for element in ["boxes", "whiskers", "means", "medians", "caps"]:
        plt.setp(bp[element], color="black")

    for patch in bp["boxes"]:
        patch.set(facecolor="lightblue", alpha=0.7)

    ax.set_xlabel("Alpha")
    ax.set_ylabel("Harm Estimate")
    ax.axhline(y=0.5, color="red", linestyle="--", label="ground truth")

    # Set x-ticks to match alpha values
    ax.set_xticks(range(1, len(alphas) + 1))
    ax.set_xticklabels([f"{alpha:.1g}" for alpha in alphas])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format=save_format, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
