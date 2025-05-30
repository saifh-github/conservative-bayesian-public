import argparse
import gzip
import datetime
import pickle
import os
import time
from tqdm import tqdm
from termcolor import colored
import numpy as np

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
parser.add_argument("--device", default="auto", type=str, help="Device to use: 'cpu', 'cuda', 'mps', or 'auto'")
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
    default=[1e-5, 3e-5, 1e-4, 3e-4, 1e-2, 3e-2, 1e-1, 3e-1, 1.0],
    type=list,
)

def main(args):
    # Calculate alphas based on d_arm
    if args.alphas == []:
        P_i_star = 1 / (2 ** args.d_arm)
        delta = 0.1  # 1-delta = 90% probability for Prop. 4.6
        max_alpha = P_i_star * delta  # α ≤ δ * P(i*)
        args.alphas = [max_alpha * (0.1 ** i) for i in range(11)]  # 11 log-spaced values
    
    start_time = time.time()
    args.save_path = f"results/tightness/{args.n_episodes}/results.pkl.gz"

    # Set device
    device = utils.get_device(args.device)
    print(colored(f"🖥️  Using device: {device}", "cyan"))
    t.set_default_device(t.device(device))

    results = {}
    results["args"] = args
    results["overestimates"] = []
    results["overestimate error"] = []
    results["harm estimates"] = []
    results["failed_alphas"] = []  # Track which alphas failed
    results["harm_stats"] = {}  # Add detailed statistics for each alpha

    env = utils.make_env(args, d_arm=args.d_arm, exploding=False)
    for alpha in tqdm(args.alphas, desc="alphas"):
        agent = agents.Uniform(env=env, alpha=alpha, guardrail="non-iid")
        agent.device = device
        
        try:
            overestimate_mean, overestimate_error, harm_estimates = utils.run_tightness_episodes(agent, args)
            
            # Add validation checks
            if harm_estimates is None or len(harm_estimates) == 0:
                raise ValueError(f"Empty harm estimates for alpha={alpha}")
            
            # Calculate detailed statistics
            harm_array = np.array(harm_estimates)
            stats = {
                "mean": float(np.mean(harm_array)),
                "std": float(np.std(harm_array)),
                "min": float(np.min(harm_array)),
                "max": float(np.max(harm_array)),
                "count": len(harm_array),
                "unique_values": len(np.unique(harm_array)),
                "percentiles": {
                    "25": float(np.percentile(harm_array, 25)),
                    "50": float(np.percentile(harm_array, 50)),
                    "75": float(np.percentile(harm_array, 75))
                }
            }
            results["harm_stats"][alpha] = stats
            
            print(f"\nStatistics for alpha={alpha}:")
            print(f"Mean: {stats['mean']:.6f}")
            print(f"Std Dev: {stats['std']:.6f}")
            print(f"Min: {stats['min']:.6f}")
            print(f"Max: {stats['max']:.6f}")
            print(f"Sample Count: {stats['count']}")
            print(f"Unique Values: {stats['unique_values']}")
            print(f"Quartiles: {stats['percentiles']}")
            
            # Move results to CPU before storing
            if isinstance(overestimate_mean, t.Tensor):
                overestimate_mean = overestimate_mean.cpu().numpy()
            if isinstance(overestimate_error, t.Tensor):
                overestimate_error = overestimate_error.cpu().numpy()
            if isinstance(harm_estimates, t.Tensor):
                harm_estimates = harm_estimates.cpu().numpy()
            
            results["overestimates"].append(overestimate_mean)
            results["overestimate error"].append(overestimate_error)
            results["harm estimates"].append(harm_estimates)
        except Exception as e:
            print(colored(f"❌ Detailed error for alpha={alpha}:", "red"))
            print(colored(f"Error type: {type(e).__name__}", "red"))
            print(colored(f"Error message: {str(e)}", "red"))
            results["failed_alphas"].append(alpha)
            if device == "mps":
                t.mps.empty_cache()  # Clear MPS cache on error
            continue

    end_time = time.time()
    execution_time = end_time - start_time
    print(colored(f"⏱️  Execution time: {execution_time:.2f} seconds", "yellow"))
    results["execution_time"] = execution_time
    
    assert len(results["overestimates"]) == len(args.alphas), "Missing results for some alphas"
    assert len(results["overestimate error"]) == len(args.alphas), "Missing results for some alphas"
    assert len(results["harm estimates"]) == len(args.alphas), "Missing results for some alphas"

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with gzip.open(args.save_path, "wb") as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
