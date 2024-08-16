import os
import glob
import gzip
import pickle
import argparse
from utils import plotting

parser = argparse.ArgumentParser()
parser.add_argument("--timestamp", default="latest")


def get_latest(directory="results/tightness"):
    pattern = os.path.join(directory, "*")
    files = glob.glob(pattern)
    latest = max(files, key=os.path.getmtime)
    return latest


def main(args):
    if args.timestamp == "latest":
        results_dir = get_latest()
    else:
        results_dir = os.path.join("results/tightness", args.timestamp)

    path = os.path.join(results_dir, "results.pkl.gz")

    with gzip.open(path, "rb") as f:
        results = pickle.load(f)
    print(f"Loaded {path}")
    ex_time = results["execution_time"]
    print(
        f"execution time was {round(ex_time)} seconds, or {round(ex_time/60)} minutes, or {round(ex_time/3600, ndigits=3)} hours for {results['args'].n_episodes} episodes"
    )

    save_path = os.path.join(results_dir, "overestimation.pdf")
    plotting.plot_overestimation(
        results, plot_error_bars=False, save_path=save_path, save_format="pdf"
    )
    save_path = os.path.join(results_dir, "overestimation_error.pdf")
    plotting.plot_overestimation(
        results, plot_error_bars=True, save_path=save_path, save_format="pdf"
    )
    save_path = os.path.join(results_dir, "harm_estimates.pdf")
    plotting.box_plot(results, save_path=save_path, save_format="pdf")

    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
