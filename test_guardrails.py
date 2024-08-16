from utils import *
import gymnasium as gym
import pytest
from agents import agents as agents
from envs.exploding_bandit import ExplodingBandit
import torch as t
import einops
import scipy
import matplotlib.pyplot as plt


@pytest.fixture
def env():
    gym.envs.registration.register(
        id="ExplodingBandit",
        entry_point="envs.exploding_bandit:ExplodingBandit",
        kwargs={
            "n_arm": 10,
            "exploding_frac": 0.1,
            "d_arm": 10,
            "k": 2,
            "exploding": False,
            "sigma_r": 0.5,
        },
    )
    return gym.make("ExplodingBandit")


def test_run_episode(env):
    agent = agents.RandomAgent("ExplodingBandit")
    n_steps = 100
    agent.run_episode(n_steps)
    assert (
        len(agent.episode_rewards) == n_steps
    ), f"the agent ran for {len(agent.episode_rewards)} timesteps, but should have been {n_steps}"


def test_belief_update(env):

    agent = agents.RandomAgent("ExplodingBandit", guardrail=None)
    n_steps = 100
    agent.run_episode(n_steps)
    true_theory = agent.env.unwrapped.reward_weights
    agent_mode = agent.hypotheses[agent.log_posterior.argmax()]
    print("\nguardrail = {guardrail}")
    print(f"True theory = {true_theory}")
    print("final posterior:")
    for i in range(len(agent.hypotheses)):
        print(
            f"hypothesis {agent.hypotheses[i]} has probability {t.exp(agent.log_posterior[i])}"
        )
    assert t.allclose(
        true_theory, agent_mode
    ), f"Bayesian has not converged to the correct mode after {n_steps} timesteps. True theory = {true_theory}. Agent mode = {agent_mode}."


def test_setup_with_guardrail(env):
    for guardrail in ["none", "cheating", "posterior", "non_iid", "iid"]:
        agent = agents.Boltzmann(
            "ExplodingBandit", guardrail=guardrail, alpha=0.77, beta=0.23, threshold=0.8
        )
        assert agent.threshold == 0.8
        if agent.guardrail == "non_iid":
            assert agent.guardrail.alpha == 0.77
        assert agent.beta == 0.23


def test_posterior_guardrail(env):
    agent = agents.Boltzmann("ExplodingBandit", guardrail="posterior", threshold=0.05)
    for arm in range(env.unwrapped.n_arm):
        hypothesised_reward_means = einops.einsum(
            agent.hypotheses,
            agent.env.unwrapped.arm_features[arm],
            "n_hypotheses d_arm, d_arm -> n_hypotheses",
        )
        hypothesised_explosion_probs = 1 - t.distributions.Normal(
            loc=hypothesised_reward_means.float(), scale=agent.env.unwrapped.sigma_r
        ).cdf(agent.env.unwrapped.explosion_threshold)
        posterior_explosion_prob = einops.einsum(
            t.exp(agent.log_posterior),
            hypothesised_explosion_probs,
            "n_hypotheses, n_hypotheses -> ",
        )
        assert t.isclose(
            posterior_explosion_prob, agent.guardrail.harm_estimate(arm), atol=0.001
        ), f"our prob = {posterior_explosion_prob}, guardrail_prob = {agent.guardrail.harm_estimate(arm)}."
        assert t.all(
            posterior_explosion_prob <= agent.threshold
        ) == agent.guardrail.check(arm)

    agent.run_episode(100)

    for arm in range(env.unwrapped.n_arm):
        hypothesised_reward_means = einops.einsum(
            agent.hypotheses,
            agent.env.unwrapped.arm_features[arm],
            "n_hypotheses d_arm, d_arm -> n_hypotheses",
        )
        hypothesised_explosion_probs = 1 - t.distributions.Normal(
            loc=hypothesised_reward_means.float(), scale=agent.env.unwrapped.sigma_r
        ).cdf(agent.env.unwrapped.explosion_threshold)
        posterior_explosion_prob = einops.einsum(
            t.exp(agent.log_posterior),
            hypothesised_explosion_probs,
            "n_hypotheses, n_hypotheses -> ",
        )
        assert t.isclose(
            posterior_explosion_prob, agent.guardrail.harm_estimate(arm), atol=0.001
        ), f"our prob = {posterior_explosion_prob}, guardrail_prob = {agent.guardrail.harm_estimate(arm)}, explosion threshold = {agent.env.unwrapped.explosion_threshold}"
        assert t.all(
            posterior_explosion_prob <= agent.threshold
        ) == agent.guardrail.check(arm)


def test_cheating_guardrail(env):
    agent = agents.Boltzmann("ExplodingBandit", guardrail="cheating", threshold=0.05)
    for arm in range(env.unwrapped.n_arm):
        reward_mean = t.dot(
            agent.env.unwrapped.arm_features[arm], agent.env.unwrapped.reward_weights
        )
        assert (
            reward_mean == agent.env.unwrapped.reward_means[arm]
        ), "bandit's reward means disagree with our calculated ones"
        true_explosion_prob = t.tensor(
            1
            - scipy.stats.norm(reward_mean, agent.env.unwrapped.sigma_r).cdf(
                agent.env.unwrapped.explosion_threshold
            )
        ).float()
        assert t.isclose(
            true_explosion_prob, agent.guardrail.harm_estimate(arm), atol=0.001
        ), f"our prob = {true_explosion_prob}, guardrail_prob = {agent.guardrail.harm_estimate(arm)}.\n reward mean = {agent.env.unwrapped.reward_means[arm]}. reward sd = {agent.env.unwrapped.sigma_r}\n explosion threshold = {agent.env.unwrapped.explosion_threshold}"
        assert t.all(true_explosion_prob <= agent.threshold) == agent.guardrail.check(
            arm
        )

    agent.run_episode(100)

    for arm in range(env.unwrapped.n_arm):
        reward_mean = t.dot(
            agent.env.unwrapped.arm_features[arm], agent.env.unwrapped.reward_weights
        )
        assert (
            reward_mean == agent.env.unwrapped.reward_means[arm]
        ), "bandit's reward means disagree with our calculated ones"

        true_explosion_prob = t.tensor(
            1
            - scipy.stats.norm(reward_mean, agent.env.unwrapped.sigma_r).cdf(
                agent.env.unwrapped.explosion_threshold
            )
        ).float()

        assert t.isclose(
            true_explosion_prob, agent.guardrail.harm_estimate(arm), atol=0.001
        ), f"our prob = {true_explosion_prob}, guardrail_prob = {agent.guardrail.harm_estimate(arm)}.\n reward mean = {agent.env.unwrapped.reward_means[arm]}. reward sd = {agent.env.unwrapped.sigma_r}, \n explosion threshold = {agent.env.unwrapped.explosion_threshold}"
        assert t.all(true_explosion_prob <= agent.threshold) == agent.guardrail.check(
            arm
        )


def test_m_alpha(env):
    agent = agents.Boltzmann("ExplodingBandit", guardrail="non-iid", threshold=0.05)
    agent.log_posterior = t.tensor([0.1, 0.5, 0.2, 0.2]).log()
    alphas = [0.001, 0.01, 0.1, 0.2, 0.3, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
    for alpha in alphas:
        agent.guardrail.alpha = alpha
        m_alpha = agent.guardrail.m_alpha()
        print(f"For alpha = {alpha}, guardrail m_alpha = {m_alpha}")
        if alpha > 2 / 7:
            assert t.all(m_alpha == t.tensor([False, True, False, False]))
        if alpha < 2 / 7 and alpha > 2 / 9:
            assert t.all(m_alpha == t.tensor([False, True, True, False])) or t.all(
                m_alpha=t.tensor([False, True, False, True])
            )
        if alpha < 2 / 9 and alpha > 1 / 10:
            assert t.all(m_alpha == t.tensor([False, True, True, True]))
        if alpha < 1 / 10:
            assert t.all(m_alpha == t.tensor([True, True, True, True]))

    agent.log_posterior = t.tensor([0.4, 0.1, 0.2, 0.3]).log()
    for alpha in alphas:
        agent.guardrail.alpha = alpha
        m_alpha = agent.guardrail.m_alpha()
    if alpha > 3 / 7:
        assert t.all(m_alpha == t.tensor([True, False, False, False]))
    if alpha < 3 / 7 and alpha > 2 / 9:
        assert t.all(m_alpha == t.tensor([True, False, False, True]))
    if alpha < 2 / 7 and alpha > 1 / 10:
        assert t.all(m_alpha == t.tensor([True, False, True, True]))
    if alpha < 1 / 10:
        assert t.all(m_alpha == t.tensor([True, True, True, True]))

    agent.log_posterior = t.ones(32).log()
    for alpha in alphas:
        agent.guardrail.alpha = alpha
        m_alpha = agent.guardrail.m_alpha()
    for i in range(1, 32):
        if alpha < 1 / i and alpha > 1 / (i + 1):
            assert m_alpha.sum() == i


@pytest.mark.parametrize("alpha", [0, 1.0])
def test_non_iid_guardrail(env, alpha):

    xs = list(range(-10, 11))
    valid = 1 - scipy.stats.norm.cdf(xs)

    agent = agents.Boltzmann(
        "ExplodingBandit", guardrail="non-iid", threshold=0.05, alpha=alpha
    )
    assert agent.guardrail.alpha == alpha

    print(f"\n alpha = {alpha}. Before ep")

    for arm in range(agent.env.n_arm):
        p_harm_given_theory = agent.guardrail.p_harm_given_theory(arm)
        m_alpha = agent.guardrail.m_alpha()
        valid_p_harm_given_theory = [
            p_harm_given_theory[i]
            for i in range(len(p_harm_given_theory))
            if m_alpha[i]
        ]
        guardrail_estimate = agent.guardrail.harm_estimate(arm)
        assert len(valid_p_harm_given_theory) == t.sum(m_alpha)
        assert max(valid_p_harm_given_theory) == guardrail_estimate
        assert any(t.tensor(valid) - guardrail_estimate < 0.01)

    print(f"\n alpha = {alpha}. After ep")
    agent.run_episode(10)

    for arm in range(agent.env.n_arm):
        p_harm_given_theory = agent.guardrail.p_harm_given_theory(arm)
        m_alpha = agent.guardrail.m_alpha()
        valid_p_harm_given_theory = [
            p_harm_given_theory[i]
            for i in range(len(p_harm_given_theory))
            if m_alpha[i]
        ]
        assert len(valid_p_harm_given_theory) == t.sum(m_alpha)
        assert max(valid_p_harm_given_theory) == agent.guardrail.harm_estimate(arm)


def test_iid_guardrail(env):
    agent = agents.Boltzmann("ExplodingBandit", guardrail="iid", threshold=0.05)
    for arm in range(env.unwrapped.n_arm):
        hypothesised_reward_means = einops.einsum(
            agent.hypotheses,
            agent.env.unwrapped.arm_features[arm],
            "n_hypotheses d_arm, d_arm -> n_hypotheses",
        )
        hypothesised_explosion_probs = 1 - t.distributions.Normal(
            loc=hypothesised_reward_means.float(), scale=agent.env.unwrapped.sigma_r
        ).cdf(agent.env.unwrapped.explosion_threshold)
        plausible_harm = einops.einsum(
            t.exp(agent.log_posterior),
            hypothesised_explosion_probs,
            "n_hypotheses, n_hypotheses -> n_hypotheses",
        )
        argmax_plausible_harm = t.argmax(plausible_harm)
        harm_estimates = hypothesised_explosion_probs[argmax_plausible_harm]
        harm_estimate = t.max(harm_estimates)
        assert t.isclose(
            harm_estimate, agent.guardrail.harm_estimate(arm), atol=0.001
        ), f"our prob = {posterior_explosion_prob}, guardrail_prob = {agent.guardrail.harm_estimate(arm)}."
        assert t.all(harm_estimate <= agent.threshold) == agent.guardrail.check(arm)

    agent.run_episode(100)

    for arm in range(env.unwrapped.n_arm):
        hypothesised_reward_means = einops.einsum(
            agent.hypotheses,
            agent.env.unwrapped.arm_features[arm],
            "n_hypotheses d_arm, d_arm -> n_hypotheses",
        )
        hypothesised_explosion_probs = 1 - t.distributions.Normal(
            loc=hypothesised_reward_means.float(), scale=agent.env.unwrapped.sigma_r
        ).cdf(agent.env.unwrapped.explosion_threshold)
        plausible_harm = einops.einsum(
            t.exp(agent.log_posterior),
            hypothesised_explosion_probs,
            "n_hypotheses, n_hypotheses -> n_hypotheses",
        )
        argmax_plausible_harm = t.argmax(plausible_harm)
        harm_estimates = hypothesised_explosion_probs[argmax_plausible_harm]
        harm_estimate = t.max(harm_estimates)
        assert t.isclose(
            harm_estimate, agent.guardrail.harm_estimate(arm), atol=0.001
        ), f"our prob = {harm_estimate}, guardrail_prob = {agent.guardrail.harm_estimate(arm)}."
        assert t.all(harm_estimate <= agent.threshold) == agent.guardrail.check(arm)


def test_uniform(env):
    alphas = [1e-5, 3e-5, 1e-4, 3e-4, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
    violin_data = []
    for alpha in alphas:
        agent = agents.Uniform("ExplodingBandit", alpha=alpha)
        n_steps = 3
        overestimate_frequency, worst_action_harm_probs = agent.run_episode(n_steps)
        assert 0 <= overestimate_frequency <= 1
        assert len(worst_action_harm_probs) == n_steps
        violin_data.append(worst_action_harm_probs)
    plt.violinplot(violin_data)
    plt.xlabel("Alpha")
    plt.ylabel("Prop 4.6 harm estimate")
    plt.axhline(y=0.5, color="red", linestyle="--", label="ground truth")
    plt.legend()
    plt.savefig(f"violin_plot.png", dpi=300, bbox_inches="tight")
