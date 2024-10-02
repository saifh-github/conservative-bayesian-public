import gzip
import datetime
import pickle
import os
import time
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import wandb
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath("../"))
sys.path.insert(0, os.path.abspath("./"))

import torch as t

from utils import utils, plotting
import agents.agents as agents


def custom_metric(reward_mean, deaths_mean):
    # Prioritize minimizing deaths over maximizing reward
    return reward_mean / (1 + deaths_mean**2)


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):
    # Calculate alphas based on d_arm
    if cfg.experiment.alphas == []:
        P_i_star = 1 / (2**cfg.environment.d_arm)
        delta = 0.1  # 1-delta = 90% probability for Prop. 4.6
        max_alpha = P_i_star * delta  # α ≤ δ * P(i*)
        cfg.experiment.alphas = [
            max_alpha * (0.1**i) for i in range(9)
        ]  # log-spaced values
    cfg.save_path = f"results/non_iid/{cfg.experiment.n_episodes}/results.pkl.gz"

    wandb_config = dict(cfg)
    wandb.init(project="conservative-bayesian-agent", config=wandb_config)

    t.set_default_device(t.device(cfg.device))
    t.set_grad_enabled(False)

    if cfg.device == "cuda":
        assert (
            t.cuda.is_available()
        ), "Cuda not available. Use --device='cpu', or get cuda working."
    else:
        print("Running on CPU. Use --device='cuda' for GPU.")

    results = {"cfg": cfg}
    results["args"] = {
        "n_episodes": cfg.experiment.n_episodes,
        "episode_length": cfg.experiment.episode_length,
        "guardrail_thresholds": cfg.experiment.guardrail_thresholds,
        "guardrail_baselines": cfg.experiment.guardrail_baselines,
        "alphas": cfg.experiment.alphas,
        "exploding_frac": cfg.environment.exploding_frac,
        "n_arm": cfg.environment.n_arm,
        "d_arm": cfg.environment.d_arm,
        "k": cfg.environment.k,
        "sigma_r": cfg.environment.sigma_r,
        "beta": cfg.environment.beta,
        "device": cfg.device,
        "print": cfg.print,
        "save_path": cfg.save_path,
    }
    for guardrail in cfg.experiment.guardrail_baselines:
        results[guardrail] = []
    results["non-iid"] = {}
    results["new-non-iid"] = {}
    for alpha in cfg.experiment.alphas:
        results["non-iid"][alpha] = []
        results["new-non-iid"][alpha] = []

    start_time = time.time()


    def update_live_wandb_table(threshold, results, baseline_data, processed_alphas):
        # Start with the cached baseline data
        data = baseline_data

        # Collect data for processed alphas
        for guardrail_name in ["non-iid", "new-non-iid"]:
            if guardrail_name in results:
                for alpha in processed_alphas:
                    if alpha in results[guardrail_name]:
                        for record in results[guardrail_name][alpha]:
                            threshold_value, reward_mean, reward_error, deaths_mean, deaths_error, extras, custom_score = record
                            if threshold_value != threshold:
                                continue
                            data.append({
                                'Alpha': float(alpha),
                                'Reward': reward_mean,
                                'Deaths': deaths_mean,
                                'Reward_Error': reward_error,
                                'Deaths_Error': deaths_error,
                                'Custom_Score': custom_score,
                                'Guardrail': guardrail_name,
                                'Threshold': threshold,
                                'Is_Baseline': False
                            })

        df = pd.DataFrame(data)
        table = wandb.Table(dataframe=df)

        # Log the table so it can be used in the W&B UI
        wandb.log({f"data_threshold_{threshold}": table})

        # For Reward vs Alpha
        line_plot_reward = wandb.plot.line(
            table,
            x='Alpha',
            y='Reward',
            title=f"Reward vs Alpha (Threshold: {threshold})",
            stroke='Guardrail'
        )

        wandb.log({
            f"Reward_vs_Alpha_Threshold_{threshold}": line_plot_reward
        })

        # For Deaths vs Alpha
        line_plot_deaths = wandb.plot.line(
            table,
            x='Alpha',
            y='Deaths',
            title=f"Deaths vs Alpha (Threshold: {threshold})",
            stroke='Guardrail'
        )

        wandb.log({
            f"Deaths_vs_Alpha_Threshold_{threshold}": line_plot_deaths
        })

    for threshold in tqdm(cfg.experiment.guardrail_thresholds, desc="guardrail threshold"):
        env_variable = utils.make_env(cfg.environment)
        if not cfg.device == "cuda":
            env_variable.reset()
            env_variable.render()
        if cfg.print:
            print(f"\nGuardrail threshold = {threshold}")

        # Process baselines once per threshold
        baseline_data = []
        for guardrail in tqdm(cfg.experiment.guardrail_baselines, desc="guardrail"):
            agent = agents.Boltzmann(
                env=env_variable,
                beta=cfg.environment.beta,
                guardrail=guardrail,
                threshold=threshold,
            )
            guardrail_results = utils.run_episodes(agent, cfg)
            reward_mean, reward_error, deaths_mean, deaths_error, extras = guardrail_results
            custom_score = custom_metric(reward_mean, deaths_mean)
            results[guardrail].append(
                (
                    threshold,
                    reward_mean,
                    reward_error,
                    deaths_mean,
                    deaths_error,
                    extras,
                    custom_score,
                )
            )

            baseline_data.append({
                'Alpha': 0.0,
                'Reward': reward_mean,
                'Deaths': deaths_mean,
                'Reward_Error': reward_error,
                'Deaths_Error': deaths_error,
                'Custom_Score': custom_score,
                'Guardrail': guardrail,
                'Threshold': threshold,
                'Is_Baseline': True
            })

            wandb.log(
                {
                    f"{guardrail}_reward_mean_threshold_{threshold}": reward_mean,
                    f"{guardrail}_reward_error_threshold_{threshold}": reward_error,
                    f"{guardrail}_deaths_mean_threshold_{threshold}": deaths_mean,
                    f"{guardrail}_deaths_error_threshold_{threshold}": deaths_error,
                    f"{guardrail}_custom_score_threshold_{threshold}": custom_score,
                }
            )

        processed_alphas = set()
        new_non_iid_custom_scores = []
        for alpha in tqdm(cfg.experiment.alphas, desc="alpha"):
            for guardrail_name in ["non-iid", "new-non-iid"]:
                agent = agents.Boltzmann(
                    env=env_variable,
                    beta=cfg.environment.beta,
                    alpha=alpha,
                    guardrail=guardrail_name,
                    threshold=threshold,
                    guardrail_params=cfg.guardrail_new_non_iid,
                )
                guardrail_results = utils.run_episodes(agent, cfg)
                reward_mean, reward_error, deaths_mean, deaths_error, extras = guardrail_results
                custom_score = custom_metric(reward_mean, deaths_mean)

                if alpha not in results[guardrail_name]:
                    results[guardrail_name][alpha] = []

                results[guardrail_name][alpha].append(
                    (
                        threshold,
                        reward_mean,
                        reward_error,
                        deaths_mean,
                        deaths_error,
                        extras,
                        custom_score,
                    )
                )

                wandb_log_dict = {
                    "alpha": alpha,
                    f"{guardrail_name}_reward_mean_threshold_{threshold}": reward_mean,
                    f"{guardrail_name}_reward_error_threshold_{threshold}": reward_error,
                    f"{guardrail_name}_deaths_mean_threshold_{threshold}": deaths_mean,
                    f"{guardrail_name}_deaths_error_threshold_{threshold}": deaths_error,
                    f"{guardrail_name}_custom_score_threshold_{threshold}": custom_score,
                }

                # add all the baseline guardrails to wandb_log_dict
                for baseline in cfg.experiment.guardrail_baselines:
                    if baseline in results:
                        if results[baseline]:
                            baseline_data = results[baseline][0]
                            wandb_log_dict[f"{baseline}_reward_mean_threshold_{threshold}"] = baseline_data[1]
                            wandb_log_dict[f"{baseline}_reward_error_threshold_{threshold}"] = baseline_data[2]
                            wandb_log_dict[f"{baseline}_deaths_mean_threshold_{threshold}"] = baseline_data[3]
                            wandb_log_dict[f"{baseline}_deaths_error_threshold_{threshold}"] = baseline_data[4]
                            wandb_log_dict[f"{baseline}_custom_score_threshold_{threshold}"] = baseline_data[6]
                wandb.log(wandb_log_dict)
                if guardrail_name == "new-non-iid":
                    new_non_iid_custom_scores.append(custom_score)

            processed_alphas.add(alpha)
            update_live_wandb_table(threshold, results, baseline_data, processed_alphas)
        plt_fig = plotting.fig_deaths_reward_custom_metric_vs_alpha_at_threshold(results, threshold)
        wandb.log({f"plot_deaths_reward_custom_metric_vs_alpha_at_threshold_{threshold}": wandb.Image(plt_fig)})

    end_time = time.time()
    # Average custom metric for new-non-iid: the metric we want to maximize
    average_custom_metric = sum(new_non_iid_custom_scores) / len(
        new_non_iid_custom_scores
    )
    wandb.log({"average_custom_metric": average_custom_metric})

    if cfg.print:
        print(f"Average custom metric for new-non-iid: {average_custom_metric}")
        utils.print_results_table(results)
    execution_time = end_time - start_time
    results["execution_time"] = execution_time
    execution_time_hours_minutes_seconds = time.strftime(
        "%H:%M:%S", time.gmtime(execution_time)
    )
    print(
        f"execution time was {round(execution_time)} seconds, or {round(execution_time/60)} minutes, or {round(execution_time/3600, ndigits=3)} hours for {results['args'].n_episodes} episodes"
    )
    results["execution_time_readable"] = execution_time_hours_minutes_seconds
    wandb.log({"execution_time": execution_time_hours_minutes_seconds})

    # Save results
    save_path = cfg.save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with gzip.open(save_path, "wb") as f:
        pickle.dump(results, f)

    # Create a new artifact for the results on WandB
    results_artifact = wandb.Artifact("experiment_results", type="results")
    results_artifact.add_file(save_path)
    wandb.log_artifact(results_artifact)

    # Generate and log matplotlib plots on WandB
    print(
        f"Results saved locally to {save_path} 📁 and uploaded to W&B as an artifact. ✅"
    )

    plot_definitions = [
        {
            "name": "plot_deaths_and_reward_vs_alpha",
            "function": plotting.plot_deaths_and_reward_vs_alpha,
            "kwargs": {
                "plot_error_bars": True,
                "return_fig": True,
                "include_custom_metric": True,
            },
        },
        {
            "name": "plot_deaths_and_reward_vs_alpha_error",
            "function": plotting.plot_deaths_and_reward_vs_alpha,
            "kwargs": {
                "plot_error_bars": False,
                "return_fig": True,
                "include_custom_metric": True,
            },
        },
    ]

    for plot in plot_definitions:
        fig = plot["function"](results, **plot["kwargs"])
        if fig:
            wandb.log({f"{plot['name']}": wandb.Image(fig)})
            print(f"Uploaded {plot['name']} to WandB. 🎉")

if __name__ == "__main__":
    main()