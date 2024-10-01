import gymnasium as gym
import numpy as np
import torch as t
import matplotlib.pyplot as plt
import einops


class ExplodingBandit(gym.Env):

    def __init__(
        self, n_arm=10, exploding_frac=0.01, d_arm=10, k=2, sigma_r=0.5, exploding=True
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
        # self.exploding_frac = exploding_frac
        self.explosion_threshold = self.set_explosion_threshold()
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

    def set_explosion_threshold(self):
        """
        We choose the explosion threshold to roughly approximate the expected largest reward mean in a batch of arms.
        We round it to the nearest integer so there are often arms whose mean is on the threshold (such arms are the object of one of the tightness experiments.)
        """

        n_batches = 100
        reward_weights = t.randint(0, self.k, size=(n_batches, self.d_arm)).float()
        arm_features = t.randint(
            0, self.k, size=(n_batches, self.n_arm, self.d_arm)
        ).float()
        reward_means = einops.einsum(
            reward_weights,
            arm_features,
            "n_batches d_arm, n_batches n_arm d_arm -> n_batches n_arm",
        )
        max_reward_means = reward_means.max(dim=-1)[0]
        assert max_reward_means.shape == (n_batches,)
        average_max_reward_mean = max_reward_means.mean()
        return t.round(average_max_reward_mean)

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

        # if self.exploding_frac > 0.0:
        #     sorted_reward_means = t.sort(self.reward_means).values
        #     threshold_pos = int(np.ceil(self.n_arm * self.exploding_frac))
        #     self.explosion_threshold = sorted_reward_means[-threshold_pos]
        # else:
        #     self.explosion_threshold = 9999
        # return 0, {}

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
