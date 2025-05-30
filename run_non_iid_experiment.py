import argparse
import gzip
import datetime
import pickle
import os
import time
from tqdm import tqdm
from termcolor import colored

import torch as t
import gymnasium as gym

from utils import utils
import agents.agents as agents
from envs.exploding_bandit import ExplodingBandit

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--save_path", default=f"results/non_iid/{timestamp}/results.pkl.gz", type=str
)
parser.add_argument("--device", default="auto", type=str, help="Device to use: 'cpu', 'cuda', 'mps', or 'auto'")
parser.add_argument("--print", default=True, type=bool)

# hidden, fixed, hyperparameters
parser.add_argument("--beta", default=0.5, type=float)
parser.add_argument("--exploding_frac", default=0.1, type=float)
parser.add_argument("--n_arm", default=10, type=int)
parser.add_argument("--episode_length", default=25, type=int)
parser.add_argument("--n_episodes", default=10000, type=int)
parser.add_argument("--sigma_r", default=1.0, type=float)
parser.add_argument("--k", default=2, type=int)
parser.add_argument("--d_arm", default=10, type=int)


# hyperparameters we vary in the experiment
parser.add_argument(
    "--alphas",
    default=[1e-5, 3e-5, 1e-4, 3e-4, 1e-2, 3e-2, 1e-1, 3e-1, 1.0],
    type=list,
)

parser.add_argument("--guardrail_thresholds", default=[1e-3, 1e-2, 1e-1], type=list)
parser.add_argument("--include_new_non_iid", action="store_true", help="Include new-non-iid guardrail in experiments")


def main(args):
    # Calculate alphas based on d_arm
    if args.alphas == []:
        P_i_star = 1 / (2 ** args.d_arm)
        delta = 0.1  # 1-delta = 90% probability for Prop. 4.6
        max_alpha = P_i_star * delta  # α ≤ δ * P(i*)
        args.alphas = [max_alpha * (0.1 ** i) for i in range(9)]  # log-spaced values
    start_time = time.time()
    args.save_path = f"results/non_iid/{args.n_episodes}/results.pkl.gz"

    # Set device
    device = utils.get_device(args.device)
    print(colored(f"🖥️  Using device: {device}", "cyan"))
    t.set_default_device(t.device(device))
    
    if device == "cuda":
        assert t.cuda.is_available(), "CUDA not available. Use another device option."
    elif device == "mps":
        assert hasattr(t.backends, "mps") and t.backends.mps.is_available(), "MPS not available. Use another device option."
    else:
        print(colored(f"🖥️  Running on {device}. Use --device='cuda' for GPU.", "cyan"))

    results = {}
    results["args"] = args
    for guardrail in ["none", "cheating", "posterior", "iid"]:
        results[guardrail] = []
    results["non-iid"] = {}
    if args.include_new_non_iid:
        results["new-non-iid"] = {}
    for alpha in args.alphas:
        results["non-iid"][alpha] = []
        if args.include_new_non_iid:
            results["new-non-iid"][alpha] = []

    for threshold in tqdm(args.guardrail_thresholds, desc="guardrail threshold"):
        env_variable = utils.make_env(args, d_arm=args.d_arm)
        # if not device == "cuda":
        #     env_variable.reset()
        #     env_variable.render()
        if args.print:
            print(colored(f"🎯 Guardrail threshold = {threshold}", "yellow"))

        for guardrail in tqdm(
            ["none", "cheating", "posterior", "iid"], desc="guardrail"
        ):
            agent = agents.Boltzmann(
                env=env_variable,
                beta=args.beta,
                guardrail=guardrail,
                threshold=threshold,
                device=device,
            )
            guardrail_results = utils.run_episodes(agent, args)
            reward_mean, reward_error, deaths_mean, deaths_error, extras = (
                guardrail_results
            )
            results[guardrail].append(
                (
                    threshold,
                    reward_mean,
                    reward_error,
                    deaths_mean,
                    deaths_error,
                    extras,
                )
            )

        for alpha in tqdm(args.alphas, desc="alpha"):
            guardrails = ["non-iid"]
            if args.include_new_non_iid:
                guardrails.append("new-non-iid")
                
            for guardrail in guardrails:
                agent = agents.Boltzmann(
                    env=env_variable,
                    beta=args.beta,
                    alpha=alpha,
                    guardrail=guardrail,
                    threshold=threshold,
                    device=device,
                )
                assert agent.guardrail.alpha == alpha
                guardrail_results = utils.run_episodes(agent, args)
                reward_mean, reward_error, deaths_mean, deaths_error, extras = (
                    guardrail_results
                )
                results[guardrail][alpha].append(
                    (
                        threshold,
                        reward_mean,
                        reward_error,
                        deaths_mean,
                        deaths_error,
                        extras,
                    )
                )

        utils.print_results_table(results)

        end_time = time.time()
        execution_time = end_time - start_time
        print(colored(f"⏱️  Execution time: {execution_time:.2f} seconds", "yellow"))
        results["execution_time"] = execution_time

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with gzip.open(args.save_path, "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
