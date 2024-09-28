import gymnasium as gym
import numpy as np
import torch as t
import matplotlib.pyplot as plt
import einops


class ExplodingBandit(gym.Env):

    def __init__(
        self,
        n_arm=10,
        exploding_frac=0.01,
        d_arm=10,
        k=2,
        sigma_r=0.5,
        exploding=True,
        fixed_explosion_threshold=None,
    ):

        super().__init__()
        self.observation_space = gym.spaces.Discrete(1)
        self.action_space = gym.spaces.Discrete(n_arm)
        self.n_arm = n_arm
        self.d_arm = d_arm
        self.k = k  # The reward weights and arm features are sampled uniformly from {0, ..., k=1}^d_arm.
        self.sigma_r = (
            sigma_r  # The reward for an action is sampled from N(mu_r, sigma_r).
        )
        self.exploding = exploding
        self.exploding_frac = exploding_frac
        self.fixed_explosion_threshold = fixed_explosion_threshold
        self.reset()

    def step(self, arm):

        assert self.action_space.contains(arm), f"{arm} is not a valid action"
        reward = t.normal(self.reward_means[arm].float(), self.sigma_r)
        obs = 0
        truncated = False
        terminated = reward > self.explosion_threshold and self.exploding
        info = {}
        self.total_reward += reward

        return (obs, float(reward), bool(truncated), bool(terminated), info)

    def reset(self, options=None, seed=None):
        super().reset()
        self.reward_weights = t.randint(0, self.k, size=(self.d_arm,))
        self.arm_features = t.randint(0, self.k, size=(self.n_arm, self.d_arm))
        self.reward_means = einops.einsum(
            self.reward_weights.float(),
            self.arm_features.float(),
            "d_arm, n_arm d_arm -> n_arm",
        )
        self.total_reward = 0
        if self.fixed_explosion_threshold is not None:
            self.explosion_threshold = self.fixed_explosion_threshold
        elif self.exploding_frac > 0.0:
            sorted_reward_means = t.sort(self.reward_means).values
            threshold_pos = int(np.ceil(self.n_arm * self.exploding_frac))
            self.explosion_threshold = sorted_reward_means[-threshold_pos]
        else:
            self.explosion_threshold = 9999
        return 0, {}

    def render(self):
        """
        Shows the reward distribution of each arm as a violin plot.
        """
        bandit_samples = []
        for arm in range(self.action_space.n):
            bandit_samples += [
                t.normal(self.reward_means[arm], self.sigma_r, size=(1000,))
            ]
        plt.violinplot(bandit_samples, showmeans=True)
        plt.xlabel("Bandit Arm")
        plt.ylabel("Reward Distribution")
        plt.axhline(
            y=self.explosion_threshold, color="red", linestyle="--", label="explosion"
        )
        plt.legend()
        plt.savefig(
            f"arm_distributions_d={self.d_arm}.png", dpi=300, bbox_inches="tight"
        )

        plt.show()
        plt.close()
