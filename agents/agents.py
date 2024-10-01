import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch as t
from agents.guardrails import (
    CheatingGuardrail,
    PosteriorGuardrail,
    IidGuardrail,
    NonIidGuardrail,
)
import einops
import scipy


class Agent:
    def __init__(self, env_id=None, env=None):

        if env_id and not env:
            self.env = gym.make(env_id)
            self.env_name = env_id
        elif env and not env_id:
            self.env = env
        else:
            raise ValueError("Exactly one of env_id or env must be provided")

    def get_action(self):
        raise NotImplementedError


class Bayesian(Agent):

    def __init__(
        self,
        env_id=None,
        env=None,
        name="BayesianAgent",
        guardrail=None,
        threshold=0.05,
        alpha=None,
    ):
        super().__init__(env_id, env)
        self.name = name

        guardrails = {
            "cheating": CheatingGuardrail(self, threshold),
            "posterior": PosteriorGuardrail(self, threshold),
            "iid": IidGuardrail(self, threshold),
            "non-iid": NonIidGuardrail(self, threshold, alpha),
            "none": None,
        }

        self.guardrail = guardrails.get(guardrail)

        if self.guardrail is not None:
            self.name += f"+{guardrail}Guardrail"

        self.threshold = threshold
        self.d_arm = self.env.unwrapped.d_arm
        self.k = self.env.unwrapped.k
        self.episode_rejections = 0
        self.actions_rejected_this_timestep = []

        # The agent is given the correct prior over the reward weights, which is the uniform over {0, ..., k-1}^d.
        # There are k^d possible reward weight vectors, which we index from 0 to k^d - 1.
        prior = t.ones(self.k**self.d_arm) / (self.k**self.d_arm)
        assert t.isclose(prior.sum(), t.tensor(1.0))
        self.log_prior = t.log(prior)
        ranges = [t.arange(self.k) for _ in range(self.d_arm)]
        self.hypotheses = t.cartesian_prod(*ranges)
        self.log_posterior = self.log_prior.clone()

    def reset(self):
        self.log_posterior = self.log_prior

    def update_beliefs(self, action, reward):
        features = self.env.unwrapped.arm_features[action]
        hypothesised_reward_means = einops.einsum(
            self.hypotheses.float(),
            features.float(),
            "n_hypotheses d_arm, d_arm -> n_hypotheses",
        )

        log_prior = self.log_posterior
        log_likelihoods = t.distributions.Normal(
            loc=hypothesised_reward_means, scale=self.env.unwrapped.sigma_r
        ).log_prob(t.tensor(reward))

        unnormalised_log_posterior = log_prior + log_likelihoods
        self.log_posterior = unnormalised_log_posterior - t.logsumexp(
            unnormalised_log_posterior, dim=0
        )  # normalise in logspace

    def run_episode(
        self,
        episode_length,
        use_guardrail=True,
    ):
        self.reset()
        self.env.reset()
        terminated = False
        self.episode_rewards = []
        timestep = 0
        self.episode_rejections = 0
        self.overestimates = 0

        for i in range(episode_length):

            self.actions_rejected_this_timestep = []
            action = self.get_action()
            if use_guardrail:
                while (
                    self.guardrail is not None and self.guardrail.check(action) == False
                ):

                    if (
                        len(self.actions_rejected_this_timestep)
                        == self.env.unwrapped.n_arm
                    ):  # If all actions have been rejected...

                        self.episode_rejections += self.env.unwrapped.n_arm * (
                            episode_length - timestep - 1
                        )  # ...skip to end of episode.
                        return (
                            np.sum(self.episode_rewards),
                            self.episode_rejections,
                            episode_length,
                            0,
                        )

                    action = self.get_action()
                    while action in self.actions_rejected_this_timestep:
                        action = self.get_action()

            _, reward, _, terminated, _ = self.env.step(action)

            self.episode_rewards.append(reward)

            self.update_beliefs(action, reward)

            timestep += 1

            if terminated:
                break

        if timestep < episode_length:
            self.episode_deaths = 1
        else:
            self.episode_deaths = 0

        rewards = np.sum(self.episode_rewards)
        self.episode_timesteps_survived = timestep

        return (
            rewards,
            self.episode_rejections,
            self.episode_timesteps_survived,
            self.episode_deaths,
        )


class RandomAgent(Bayesian):
    def __init__(
        self, env_id=None, env=None, guardrail=None, alpha=None, name="Random"
    ):
        super().__init__(
            env_id=env_id, env=env, name=name, guardrail=guardrail, alpha=alpha
        )
        self.name = name

    def get_action(self):
        return self.env.action_space.sample()


class Boltzmann(Bayesian):
    """
    Chooses its actions by softmaxing the expected rewards according to its beliefs. Beta is the inverse temperature parameter.
    If some actions have already been rejected this timestep, we zero out their probabilities. This doesn't change the distribution of accepted actions,
    but saves repeatedly checking and rejecting the same ones.
    """

    def __init__(
        self,
        env_id=None,
        env=None,
        name="Boltzmann",
        guardrail=None,
        threshold=0.05,
        beta=0.5,
        alpha=None,
    ):
        super().__init__(
            env_id=env_id,
            env=env,
            name=name,
            guardrail=guardrail,
            threshold=threshold,
            alpha=alpha,
        )
        self.beta = beta

    def get_action(self):

        features = self.env.unwrapped.arm_features
        posterior = t.exp(self.log_posterior)
        posterior_mean = einops.einsum(
            posterior,
            self.hypotheses.to(t.float32),
            "n_hypotheses, n_hypotheses d_arm  -> d_arm",
        )

        estimated_reward_means = einops.einsum(
            features.to(t.float32), posterior_mean, "n_arm d_arm, d_arm -> n_arm"
        )
        probs = t.softmax(self.beta * estimated_reward_means, dim=0)
        probs[self.actions_rejected_this_timestep] = (
            0  # so we don't choose an action that's already been rejected
        )
        probs = probs / t.sum(probs)  # renormalise
        action = t.multinomial(probs, 1).item()
        return action


class Uniform(Bayesian):
    """
    Chooses its actions uniformly. This is for the experiments where we look at the tightness of the bounds, so its run_episode function records different stuff, and it has both the iid and the cheating guardrail in order to make comparisons between them.

    We record overestimate frequency, and also record the value of the harm estimate of the action with 0.5 harm_probability.
    """

    def __init__(
        self,
        env_id=None,
        env=None,
        name="Uniform",
        alpha=None,
    ):
        super().__init__(
            env_id=env_id, env=env, name=name, guardrail=None, threshold=None
        )

        self.cheating_guardrail = CheatingGuardrail(self, None)
        self.non_iid_guardrail = NonIidGuardrail(self, None, alpha)

    def get_action(self):
        return np.random.randint(self.env.unwrapped.n_arm)

    def run_episode(
        self,
        episode_length,
    ):

        actions = t.tensor(t.arange(self.env.unwrapped.n_arm))
        self.reset()
        assert t.all(self.log_posterior == self.log_prior)
        self.env.reset()
        fifty_fifty_mask = (
            self.env.unwrapped.reward_means == self.env.unwrapped.explosion_threshold
        )
        estimates = 0
        for fifty_fifty in actions[fifty_fifty_mask]:
            assert (
                self.cheating_guardrail.harm_estimate(fifty_fifty) == 0.5
            ), "fifty fifties should be actions with p(harm)=0.5"
        overestimates = 0
        fifty_fifties_harm_estimates = []

        for i in range(episode_length):

            action = self.get_action()
            non_iid_estimate = self.non_iid_guardrail.harm_estimate(action)
            ground_truth = self.cheating_guardrail.harm_estimate(action)
            estimates += 1
            if non_iid_estimate >= ground_truth:
                overestimates += 1

            for fifty_fifty in actions[fifty_fifty_mask]:
                fifty_fifty_harm_estimate = float(
                    self.non_iid_guardrail.harm_estimate(fifty_fifty)
                )
                fifty_fifties_harm_estimates.append(fifty_fifty_harm_estimate)

            _, reward, _, _, _ = self.env.step(action)
            self.update_beliefs(action, reward)

        overestimate_frequency = overestimates / estimates
        return overestimate_frequency, fifty_fifties_harm_estimates
