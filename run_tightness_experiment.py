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
    "--save_path", default=f"results/tightness/{timestamp}/results.pkl.gz", type=str
)
parser.add_argument("--device", default="cpu", type=str, help="cpu or cuda")
parser.add_argument("--print", default=True, type=bool)

# hidden, fixed, hyperparameters
parser.add_argument("--exploding_frac", default=0.1, type=float)
parser.add_argument("--n_arm", default=10, type=int)
parser.add_argument("--episode_length", default=25, type=int)
parser.add_argument("--n_episodes", default=10000, type=int)
parser.add_argument("--sigma_r", default=1.0, type=float)
parser.add_argument("--k", default=2, type=int)
parser.add_argument("--d_arm", default=10, type=int)


# hyperparameters we vary in the experiment.
parser.add_argument(
    "--alphas",
    default=None,
    type=float,
    nargs='+',
    help="List of alpha values. If not provided, will be calculated based on d_arm."
)

def main(args):
    # Calculate alphas based on d_arm
    if args.alphas is None:
        P_i_star = 1 / (2 ** args.d_arm)
        delta = 0.1  # 1-delta = 90% probability for Prop. 4.6
        max_alpha = P_i_star * delta  # α ≤ δ * P(i*)
        args.alphas = [max_alpha * (0.1 ** i) for i in range(11)]  # 11 log-spaced values
    start_time = time.time()
    args.save_path = f"results/tightness/{args.n_episodes}/results.pkl.gz"

    t.set_default_device(t.device(args.device))
    if args.device == "cuda":
        assert (
            t.cuda.is_available()
        ), "Cuda not available. Use --device='cpu', or get cuda working."
    else:
        print("Running on CPU. Use --device='cuda' for GPU.")

    results = {}
    results["args"] = args
    results["overestimates"] = []
    results["overestimate error"] = []
    results["harm estimates"] = []

    env = utils.make_env(args, d_arm=args.d_arm, exploding=False)
    for alpha in tqdm(args.alphas, desc="alphas"):

        agent = agents.Uniform(env=env, alpha=alpha)

        overestimate_mean, overestimate_error, harm_estimates = (
            utils.run_tightness_episodes(agent, args)
        )

        results["overestimates"].append(overestimate_mean)
        results["overestimate error"].append(overestimate_error)
        results["harm estimates"].append(harm_estimates)

    assert len(results["overestimates"]) == len(args.alphas)
    assert len(results["overestimate error"]) == len(args.alphas)

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
