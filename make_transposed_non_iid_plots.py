import os
import glob
import gzip
import pickle
import argparse
from utils import plotting

parser = argparse.ArgumentParser()
parser.add_argument("--timestamp", default="latest")
parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")


def get_latest(directory="results/non_iid"):
    pattern = os.path.join(directory, "*")
    files = glob.glob(pattern)
    latest = max(files, key=os.path.getmtime)
    return latest


def main(args):
    plot_config = None
    if args.config and os.path.exists(args.config):
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        plot_config = plotting.PlotConfig.from_config(cfg)
    
    if args.timestamp == "latest":
        results_dir = get_latest()
    else:
        results_dir = os.path.join("results/non_iid", args.timestamp)

    path = os.path.join(results_dir, "results.pkl.gz")

    with gzip.open(path, "rb") as f:
        results = pickle.load(f)
    print(f"Loaded {path}")
    ex_time = results["execution_time"]
    print(
        f"execution time was {round(ex_time)} seconds, or {round(ex_time/60)} minutes, or {round(ex_time/3600, ndigits=3)} hours for {results['args'].n_episodes} episodes"
    )

    save_path = os.path.join(results_dir, "deaths_and_rewards_vs_alpha.pdf")
    plotting.plot_deaths_and_reward_vs_alpha_2x3(
        results, plot_error_bars=False, save_path=save_path, save_format="pdf", plot_config=plot_config
    )
    save_path = os.path.join(results_dir, "deaths_and_rewards_vs_alpha_error.pdf")
    plotting.plot_deaths_and_reward_vs_alpha_2x3(
        results, plot_error_bars=True, save_path=save_path, save_format="pdf", plot_config=plot_config
    )
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
