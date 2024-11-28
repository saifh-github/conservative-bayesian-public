import numpy as np
import plotly.graph_objects as go
import torch as t
import gymnasium as gym
import agents.agents as agents
from envs.exploding_bandit import ExplodingBandit
from torch.multiprocessing import Pool, cpu_count
import functools
from gymnasium.vector import AsyncVectorEnv
import ray
from termcolor import colored

def get_device(device_arg):
    if device_arg == "auto":
        if t.cuda.is_available():
            return "cuda"
        elif hasattr(t.backends, "mps") and t.backends.mps.is_available():
            optimize_mps()
            return "mps"
        return "cpu"
    return device_arg

def optimize_mps():
    t.mps.set_per_process_memory_fraction(1.0)
    t.mps.empty_cache()
    
    if hasattr(t.backends.mps, 'enable_graph_mode'):
        t.backends.mps.enable_graph_mode()
    
    t.set_float32_matmul_precision('medium')

def make_env(cfg, d_arm=None, exploding=None, num_envs=4):
    def _init_env():
        # If d_arm is provided, we're using the old calling convention
        if d_arm is not None:
            env = ExplodingBandit(
                n_arm=cfg.n_arm,
                exploding_frac=cfg.exploding_frac,
                d_arm=d_arm,
                sigma_r=cfg.sigma_r,
                k=cfg.k,
                exploding=exploding if exploding is not None else True,
                fixed_explosion_threshold=getattr(cfg, 'fixed_explosion_threshold', None),
            )
        # Otherwise, we're using the new calling convention with a single config object
        else:
            env = ExplodingBandit(
                n_arm=cfg.n_arm,
                exploding_frac=cfg.exploding_frac,
                d_arm=cfg.d_arm,
                sigma_r=cfg.sigma_r,
                k=cfg.k,
                exploding=cfg.exploding,
                fixed_explosion_threshold=cfg.fixed_explosion_threshold,
            )
        return env

    # return AsyncVectorEnv([_init_env() for _ in range(num_envs)])
    return _init_env()
def get_mean_and_error(data):
    mean = np.mean(data)
    sem = np.std(data, ddof=1) / np.sqrt(len(data))
    two_sigma_error = 2 * sem
    return mean, two_sigma_error


def create_agent(agent_template, cfg):
    """Create a new agent instance based on a template agent"""
    # Filter environment parameters
    env_params = {
        'n_arm': cfg.n_arm,
        'exploding_frac': cfg.exploding_frac,
        'd_arm': cfg.d_arm,
        'sigma_r': cfg.sigma_r,
        'k': cfg.k,
        'exploding': getattr(cfg, 'exploding', True),
        'fixed_explosion_threshold': getattr(cfg, 'fixed_explosion_threshold', None)
    }
    
    # Get all agent parameters
    agent_params = {
        'env': type(agent_template.env)(**env_params),
        'beta': agent_template.beta if hasattr(agent_template, 'beta') else None,
        'guardrail': agent_template.guardrail_type if hasattr(agent_template, 'guardrail_type') else None,
        'threshold': agent_template.threshold if hasattr(agent_template, 'threshold') else None,
        'alpha': agent_template.alpha if hasattr(agent_template, 'alpha') else None,
        'device': agent_template.device if hasattr(agent_template, 'device') else None
    }
    
    # Filter out None values
    agent_params = {k: v for k, v in agent_params.items() if v is not None}
    
    # Create new agent instance
    agent = type(agent_template)(**agent_params)
    
    # Set guardrail alpha only if both guardrail and alpha exist
    if hasattr(agent_template, 'alpha') and hasattr(agent, 'guardrail') and agent.guardrail is not None:
        agent.guardrail.alpha = agent_template.alpha
        
    return agent

def run_single_episode(args):
    agent_template, cfg, episode_length, _ = args
    agent = create_agent(agent_template, cfg)
    ep_rewards, ep_rejections, ep_timesteps_survived, ep_deaths = agent.run_episode(episode_length)
    return ep_rewards, ep_rejections, ep_timesteps_survived, ep_deaths


def run_episodes(agent, cfg):
    if hasattr(cfg, 'experiment'):
        episode_length = cfg.experiment.episode_length
        n_episodes = cfg.experiment.n_episodes
    else:
        episode_length = cfg.episode_length
        n_episodes = cfg.n_episodes

    device = agent.device if hasattr(agent, 'device') else 'cpu'
    if device == 'mps':
        optimize_mps()
    
    if not ray.is_initialized():
        ray.init()
    
    @ray.remote
    class EpisodeActor:
        def __init__(self, agent_template, cfg):
            self.agent = create_agent(agent_template, cfg)
            if device == 'mps':
                optimize_mps()  # Optimize MPS for each worker
            if hasattr(agent_template, 'guardrail') and hasattr(agent_template.guardrail, 'alpha'):
                self.agent.guardrail.alpha = agent_template.guardrail.alpha
            elif hasattr(agent_template, 'alpha'):
                self.agent.guardrail.alpha = agent_template.alpha
            
        def run_episode(self, episode_length):
            return self.agent.run_episode(episode_length)
    
    # get actor count from config or use default calculation
    if hasattr(cfg, 'experiment') and cfg.experiment.n_actors is not None:
        n_actors = cfg.experiment.n_actors
    else:
        max_actor_fraction = getattr(cfg.experiment, 'max_actor_fraction', 0.85) if hasattr(cfg, 'experiment') else 0.85
        max_actors = int(cpu_count() * max_actor_fraction)
        n_actors = min(cpu_count(), max_actors)
    
    print(colored(f"🚀 Running {n_episodes} episodes using {n_actors} actors on {device}", "cyan"))
    
    actors = [EpisodeActor.remote(agent, cfg) for _ in range(n_actors)]
    
    futures = []
    for i in range(n_episodes):
        actor = actors[i % len(actors)]
        futures.append(actor.run_episode.remote(episode_length))
    
    results = ray.get(futures)
    
    rewards, rejections, timesteps_survived, deaths = zip(*results)

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

def run_tightness_episodes(agent, cfg):
    """Run tightness experiment episodes using Ray for parallelization."""
    if not ray.is_initialized():
        ray.init()

    device = agent.device if hasattr(agent, 'device') else 'cpu'
    if device == 'mps':
        optimize_mps()
    
    @ray.remote
    class EpisodeActor:
        def __init__(self, agent_template, cfg):
            self.agent = create_agent(agent_template, cfg)
            if device == 'mps':
                optimize_mps()  # Optimize MPS for each worker
            if hasattr(agent_template, 'guardrail') and hasattr(agent_template.guardrail, 'alpha'):
                self.agent.guardrail.alpha = agent_template.guardrail.alpha
            elif hasattr(agent_template, 'alpha'):
                self.agent.guardrail.alpha = agent_template.alpha
            
        def run_episode(self, episode_length):
            return self.agent.run_episode(episode_length)
    
    # get actor count from config or use default calculation
    if hasattr(cfg, 'experiment') and cfg.experiment.n_actors is not None:
        n_actors = cfg.experiment.n_actors
    else:
        max_actor_fraction = getattr(cfg.experiment, 'max_actor_fraction', 0.85) if hasattr(cfg, 'experiment') else 0.85
        max_actors = int(cpu_count() * max_actor_fraction)
        n_actors = min(cpu_count(), max_actors)
    
    print(colored(f"🚀 Running {cfg.n_episodes} episodes using {n_actors} actors on {device}", "cyan"))
    
    actors = [EpisodeActor.remote(agent, cfg) for _ in range(n_actors)]
    
    futures = []
    for i in range(cfg.n_episodes):
        actor = actors[i % len(actors)]
        futures.append(actor.run_episode.remote(cfg.episode_length))
    
    results = ray.get(futures)
    
    overestimates, harm_estimates = [], []
    for ep_overestimates, ep_harm_estimates in results:
        overestimates.append(ep_overestimates)
        harm_estimates.extend(ep_harm_estimates)
    
    overestimate_mean, overestimate_error = get_mean_and_error(overestimates)
    return overestimate_mean, overestimate_error, harm_estimates


def print_results_table(results):
    headers = ["Guardrail", "Reward", "Deaths", "Timesteps", "Rejections", "Custom Score"]
    rows = []

    def format_mean_error(mean, error):
        return f"{mean:.2f} ± {error:.2f}"

    for guardrail in ["none", "cheating", "posterior", "iid"]:
        if guardrail in results:
            data = results[guardrail][-1]
            if len(data) == 7:  # Check if custom_score is included
                _, reward_mean, reward_error, deaths_mean, deaths_error, extras, custom_score = data
            else:
                _, reward_mean, reward_error, deaths_mean, deaths_error, extras = data
                custom_score = "N/A"
            
            row = [
                guardrail,
                format_mean_error(reward_mean, reward_error),
                format_mean_error(deaths_mean, deaths_error),
                format_mean_error(extras["timesteps_mean"], extras["timesteps_error"]),
                format_mean_error(extras["rejections_mean"], extras["rejections_error"]),
                f"{custom_score:.2f}" if isinstance(custom_score, float) else custom_score,
            ]
            rows.append(row)

    if "non-iid" in results:
        for alpha, data in results["non-iid"].items():
            if len(data[-1]) == 7:  # Check if custom_score is included
                _, reward_mean, reward_error, deaths_mean, deaths_error, extras, custom_score = data[-1]
            else:
                _, reward_mean, reward_error, deaths_mean, deaths_error, extras = data[-1]
                custom_score = "N/A"

            row = [
                f"non-iid, alpha={alpha}",
                format_mean_error(reward_mean, reward_error),
                format_mean_error(deaths_mean, deaths_error),
                format_mean_error(extras["timesteps_mean"], extras["timesteps_error"]),
                format_mean_error(
                    extras["rejections_mean"], extras["rejections_error"]
                ),
                f"{custom_score:.2f}" if isinstance(custom_score, float) else custom_score,
            ]
            rows.append(row)

    if "new-non-iid" in results:
        for alpha, data in results["new-non-iid"].items():
            if len(data[-1]) == 7:  # Check if custom_score is included
                _, reward_mean, reward_error, deaths_mean, deaths_error, extras, custom_score = data[-1]
            else:
                _, reward_mean, reward_error, deaths_mean, deaths_error, extras = data[-1]
                custom_score = "N/A"

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
                f"{custom_score:.2f}" if isinstance(custom_score, float) else custom_score,
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
