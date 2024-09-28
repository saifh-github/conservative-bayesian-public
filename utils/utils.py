import numpy as np
import plotly.graph_objects as go
import torch as t
import gymnasium as gym


device = t.device("cuda" if t.cuda.is_available() else "cpu")


def make_env(args, d_arm, exploding=True, fixed_explosion_threshold=None):

    gym.envs.registration.register(
        id="ExplodingBandit",
        entry_point="__main__:ExplodingBandit",
        kwargs={
            "n_arm": args.n_arm,
            "exploding_frac": args.exploding_frac,
            "exploding": exploding,
            "sigma_r": args.sigma_r,
            "k": args.k,
            "d_arm": d_arm,
            "fixed_explosion_threshold": fixed_explosion_threshold,
        },
    )

    env = gym.make("ExplodingBandit")

    return env


def get_mean_and_error(data):
    mean = np.mean(data)
    sem = np.std(data, ddof=1) / np.sqrt(len(data))
    two_sigma_error = 2 * sem
    return mean, two_sigma_error


def run_episodes(agent, args):

    rewards, rejections, timesteps_survived, deaths = [], [], [], []

    for i in range(args.n_episodes):
        ep_rewards, ep_rejections, ep_timesteps_survived, ep_deaths = agent.run_episode(
            args.episode_length
        )
        rewards.append(ep_rewards)
        rejections.append(ep_rejections)
        timesteps_survived.append(ep_timesteps_survived)
        deaths.append(ep_deaths)

    reward_mean, reward_error = get_mean_and_error(rewards)
    deaths_mean, deaths_error = get_mean_and_error(deaths)
    timesteps_mean, timesteps_error = get_mean_and_error(timesteps_survived)
    rejections_mean, rejections_error = get_mean_and_error(rejections)

    extras = {
        "timesteps_mean": timesteps_mean,
        "timesteps_error": timesteps_error,
        "rejections_mean": rejections_mean,
        "rejections_error": rejections_error,
    }
    return reward_mean, reward_error, deaths_mean, deaths_error, extras


def run_tightness_episodes(agent, args):

    overestimates, harm_estimates = [], []

    for i in range(args.n_episodes):
        ep_overestimates, ep_harm_estimates = agent.run_episode(args.episode_length)
        overestimates.append(ep_overestimates)
        harm_estimates += ep_harm_estimates

    assert (
        len(harm_estimates) == args.n_episodes * args.episode_length
    ), f"should have {args.n_episodes*args.episode_length} harm estimates but have {len(harm_estimates)}"

    overestimate_mean, overestimate_error = get_mean_and_error(overestimates)
    return overestimate_mean, overestimate_error, harm_estimates


def print_results_table(results):
    """
    Prints the results table for the most recent phase of training
    """
    headers = ["Guardrail", "Reward", "Deaths", "Timesteps", "Rejections"]
    rows = []

    def format_mean_error(mean, error):
        return f"{mean:.2f} ± {error:.2f}"

    for guardrail in ["none", "cheating", "posterior", "iid"]:
        if guardrail in results:
            _, reward_mean, reward_error, deaths_mean, deaths_error, extras = results[
                guardrail
            ][-1]
            row = [
                guardrail,
                format_mean_error(reward_mean, reward_error),
                format_mean_error(deaths_mean, deaths_error),
                format_mean_error(extras["timesteps_mean"], extras["timesteps_error"]),
                format_mean_error(
                    extras["rejections_mean"], extras["rejections_error"]
                ),
            ]
            rows.append(row)

    if "non-iid" in results:
        for alpha, data in results["non-iid"].items():
            _, reward_mean, reward_error, deaths_mean, deaths_error, extras = data[-1]

            row = [
                f"non-iid, alpha={alpha}",
                format_mean_error(reward_mean, reward_error),
                format_mean_error(deaths_mean, deaths_error),
                format_mean_error(extras["timesteps_mean"], extras["timesteps_error"]),
                format_mean_error(
                    extras["rejections_mean"], extras["rejections_error"]
                ),
            ]
            rows.append(row)

    if "new-non-iid" in results:
        for alpha, data in results["new-non-iid"].items():
            _, reward_mean, reward_error, deaths_mean, deaths_error, extras = data[-1]

            row = [
                f"new-non-iid, alpha={alpha}",
                format_mean_error(reward_mean, reward_error),
                format_mean_error(deaths_mean, deaths_error),
                format_mean_error(
                    extras["timesteps_mean"], extras["timesteps_error"]
                ),
                format_mean_error(
                    extras["rejections_mean"], extras["rejections_error"]
                ),
            ]
            rows.append(row)
    col_widths = [
        max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))
    ]

    print_row = lambda row: print(
        "| "
        + " | ".join(str(cell).ljust(width) for cell, width in zip(row, col_widths))
        + " |"
    )

    print_row(headers)
    print("+".join("-" * (width + 2) for width in col_widths))

    for row in rows:
        print_row(row)
