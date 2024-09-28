import argparse
import gzip
import datetime
import pickle
import os
import time
from tqdm import tqdm

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
parser.add_argument("--device", default="cpu", type=str, help="cpu or cuda")
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


def main(args):
    start_time = time.time()
    args.save_path = f"results/non_iid/{args.n_episodes}/results.pkl.gz"

    t.set_default_device(t.device(args.device))
    t.set_grad_enabled(False)
    if args.device == "cuda":
        assert (
            t.cuda.is_available()
        ), "Cuda not available. Use --device='cpu', or get cuda working."
    else:
        print("Running on CPU. Use --device='cuda' for GPU.")

    results = {}
    results["args"] = args
    for guardrail in ["none", "cheating", "posterior", "iid"]:
        results[guardrail] = []
    results["non-iid"] = {}
    results["new-non-iid"] = {}
    for alpha in args.alphas:
        results["non-iid"][alpha] = []
        results["new-non-iid"][alpha] = []

    for threshold in tqdm(args.guardrail_thresholds, desc="guardrail threshold"):
        # env_fixed = utils.make_env(
        #     args, d_arm=args.d_arm, fixed_explosion_threshold=10
        # )
        env_variable = utils.make_env(args, d_arm=args.d_arm)
        if not args.device == "cuda":
            env_variable.reset()
            env_variable.render()
        if args.print:
            print(f"guardrail threshold = {threshold}")

        for guardrail in tqdm(
            ["none", "cheating", "posterior", "iid"], desc="guardrail"
        ):
            agent = agents.Boltzmann(
                env=env_variable,
                beta=args.beta,
                guardrail=guardrail,
                threshold=threshold,
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
            for guardrail in ["non-iid", "new-non-iid"]:
                agent = agents.Boltzmann(
                    env=env_variable,
                    beta=args.beta,
                    alpha=alpha,
                    guardrail=guardrail,
                    threshold=threshold,
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
        print(f"Execution time: {execution_time:.2f} seconds")
        results["execution_time"] = execution_time

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with gzip.open(args.save_path, "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
