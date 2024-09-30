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
        P_i_star = 1 / (2 ** cfg.environment.d_arm)
        delta = 0.1  # 1-delta = 90% probability for Prop. 4.6
        max_alpha = P_i_star * delta  # α ≤ δ * P(i*)
        cfg.experiment.alphas = [max_alpha * (0.1 ** i) for i in range(9)]  # log-spaced values
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

    # Initialize results dictionary
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

    # Use hyperparameters from cfg.guardrail
    for threshold in tqdm(cfg.experiment.guardrail_thresholds, desc="guardrail threshold"):
        env_variable = utils.make_env(cfg.environment)
        if not cfg.device == "cuda":
            env_variable.reset()
            env_variable.render()
        if cfg.print:
            print(f"Guardrail threshold = {threshold}")

        for guardrail in tqdm(
            cfg.experiment.guardrail_baselines, desc="guardrail"
        ):
            agent = agents.Boltzmann(
                env=env_variable,
                beta=cfg.environment.beta,
                guardrail=guardrail,
                threshold=threshold,
            )
            guardrail_results = utils.run_episodes(agent, cfg)
            reward_mean, reward_error, deaths_mean, deaths_error, extras = (
                guardrail_results
            )
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
            wandb.log({
                'guardrail': guardrail,
                'threshold': threshold,
                'reward_mean': reward_mean,
                'reward_error': reward_error,
                'deaths_mean': deaths_mean,
                'deaths_error': deaths_error,
                'extras': extras,
                'custom_score': custom_score,
                f'reward_mean_threshold_{threshold}': reward_mean,
                f'deaths_mean_threshold_{threshold}': deaths_mean,
            })

        new_non_iid_custom_scores = []
        for alpha in tqdm(cfg.experiment.alphas, desc="alpha"):
            for guardrail_name in ["non-iid", "new-non-iid"]:
                # Initialize agent with new-non-iid guardrail
                agent = agents.Boltzmann(
                    env=env_variable,
                    beta=cfg.environment.beta,
                    alpha=alpha,
                    guardrail=guardrail_name,
                    threshold=threshold,
                    guardrail_params=cfg.guardrail_new_non_iid
                )
                guardrail_results = utils.run_episodes(agent, cfg)
                reward_mean, reward_error, deaths_mean, deaths_error, extras = guardrail_results
                # Calculate custom metric
                custom_score = custom_metric(reward_mean, deaths_mean)

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

                # Log metrics to WandB
                wandb.log({
                    'guardrail': guardrail_name,
                    'threshold': threshold,
                    'alpha': alpha,
                    'reward_mean': reward_mean,
                    'reward_error': reward_error,
                    'deaths_mean': deaths_mean,
                    'deaths_error': deaths_error,
                    'extras': extras,
                    'custom_score': custom_score,
                })
                if guardrail_name == "new-non-iid":
                    new_non_iid_custom_scores.append(custom_score)

    end_time = time.time()
    # Calculate and log the average custom metric for new-non-iid
    average_custom_metric = sum(new_non_iid_custom_scores) / len(new_non_iid_custom_scores)
    wandb.log({'average_custom_metric': average_custom_metric})
    
    if cfg.print:
        print(f"Average custom metric for new-non-iid: {average_custom_metric}")
        utils.print_results_table(results)
    execution_time = end_time - start_time
    results["execution_time"] = execution_time
    execution_time_hours_minutes_seconds = time.strftime("%H:%M:%S", time.gmtime(execution_time))
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
    results_artifact = wandb.Artifact('experiment_results', type='results')
    results_artifact.add_file(save_path)
    wandb.log_artifact(results_artifact)

    # Generate and log plots as artifacts
    artifact = wandb.Artifact('plots', type='experiment_plots')

    print(f"Results saved locally to {save_path} 📁 and uploaded to W&B as an artifact. ✅")

    plot_definitions = [
        {
            "filename": "deaths_and_rewards_vs_alpha.png",
            "function": plotting.plot_deaths_and_reward_vs_alpha,
            "kwargs": {
                "plot_error_bars": False,
                "save_path": os.path.join(os.path.dirname(save_path), "deaths_and_rewards_vs_alpha.png"),
                "save_format": "png"
            }
        },
        {
            "filename": "deaths_and_rewards_vs_alpha_error.png",
            "function": plotting.plot_deaths_and_reward_vs_alpha,
            "kwargs": {
                "plot_error_bars": True,
                "save_path": os.path.join(os.path.dirname(save_path), "deaths_and_rewards_vs_alpha_error.png"),
                "save_format": "png"
            }
        }
    ]

    for plot in plot_definitions:
        try:
            plot["function"](results, **plot["kwargs"])
            if os.path.exists(plot["kwargs"]["save_path"]):
                artifact.add_file(plot["kwargs"]["save_path"])
                print(f"Plot saved to {plot['kwargs']['save_path']} and added to artifact.")
            else:
                print(f"Warning: Plot file not found at {plot['kwargs']['save_path']}")
        except Exception as e:
            print(f"Error generating plot {plot['filename']}: {str(e)}")

    # Log the artifact to WandB
    wandb.log_artifact(artifact)
    print("All plots have been uploaded to WandB as artifacts. 🎉")

    # Create custom charts with error bars
    for threshold in cfg.experiment.guardrail_thresholds:
        reward_data = {
            'non-iid': {'x': [], 'y': [], 'error': []},
            'new-non-iid': {'x': [], 'y': [], 'error': []}
        }
        deaths_data = {
            'non-iid': {'x': [], 'y': [], 'error': []},
            'new-non-iid': {'x': [], 'y': [], 'error': []}
        }

        for alpha in cfg.experiment.alphas:
            for guardrail_name in ['non-iid', 'new-non-iid']:
                result = next((r for r in results[guardrail_name][alpha] if r[0] == threshold), None)
                if result:
                    reward_data[guardrail_name]['x'].append(float(alpha))
                    reward_data[guardrail_name]['y'].append(result[1])  # reward_mean
                    reward_data[guardrail_name]['error'].append(result[2])  # reward_error
                    deaths_data[guardrail_name]['x'].append(float(alpha))
                    deaths_data[guardrail_name]['y'].append(result[3])  # deaths_mean
                    deaths_data[guardrail_name]['error'].append(result[4])  # deaths_error

        wandb.log({
            f"reward_chart_threshold_{threshold}": wandb.plot.line_series(
                xs=[reward_data['non-iid']['x'], reward_data['new-non-iid']['x']],
                ys=[reward_data['non-iid']['y'], reward_data['new-non-iid']['y']],
                keys=["non-iid", "new-non-iid"],
                title=f"Reward vs Alpha (Threshold: {threshold})",
                xname="Alpha",
                yname="Reward Mean",
                error_y=[reward_data['non-iid']['error'], reward_data['new-non-iid']['error']]
            ),
            f"deaths_chart_threshold_{threshold}": wandb.plot.line_series(
                xs=[deaths_data['non-iid']['x'], deaths_data['new-non-iid']['x']],
                ys=[deaths_data['non-iid']['y'], deaths_data['new-non-iid']['y']],
                keys=["non-iid", "new-non-iid"],
                title=f"Deaths vs Alpha (Threshold: {threshold})",
                xname="Alpha",
                yname="Deaths Mean",
                error_y=[deaths_data['non-iid']['error'], deaths_data['new-non-iid']['error']]
            )
        })

if __name__ == "__main__":
    main()
