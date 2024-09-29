import numpy as np
import torch as t
import einops
from agents import agents as agents


class Guardrail:

    def __init__(self, agent, threshold):

        self.agent = agent
        self.threshold = threshold

    def harm_estimate(self, action):
        raise NotImplementedError

    def check(self, action=None, harm_estimate=None):

        harm_estimate = self.harm_estimate(action)

        assert (
            np.isclose(harm_estimate.cpu(), 0)
            or np.isclose(harm_estimate.cpu(), 1)
            or (0 < harm_estimate < 1)
        ), f"Harm estimate must be approximately between 0 and 1, but got {harm_estimate}"

        if harm_estimate > self.threshold:
            self.agent.episode_rejections += 1
            self.agent.actions_rejected_this_timestep.append(action)

            return False

        return True

    def marginal_p_harm(self, action):

        p_theory = t.exp(self.agent.log_posterior)

        return einops.einsum(
            p_theory, self.p_harm_given_theory(action), "n_hypotheses, n_hypotheses -> "
        )

    def p_harm_given_theory(self, action):

        arm_features = self.agent.env.unwrapped.arm_features[action]
        reward_means_given_theory = einops.einsum(
            arm_features.float(),
            self.agent.hypotheses.float(),
            "d_arm, n_hypotheses d_arm -> n_hypotheses",
        )

        p_harm_given_theory = 1 - t.distributions.Normal(
            loc=reward_means_given_theory, scale=self.agent.env.unwrapped.sigma_r
        ).cdf(self.agent.env.unwrapped.explosion_threshold)

        return p_harm_given_theory

    def p_harm_given_single_theory(self, theory, action):
        arm_features = self.agent.env.unwrapped.arm_features[action].float()
        mu_r = t.dot(theory, arm_features)
        p_harm_given_theory = 1 - t.distributions.Normal(
            loc=mu_r, scale=self.agent.env.unwrapped.sigma_r
        ).cdf(self.agent.env.unwrapped.explosion_threshold)
        return p_harm_given_theory


class CheatingGuardrail(Guardrail):

    def __init__(self, agent, threshold):
        super().__init__(agent, threshold)

    def harm_estimate(self, action):

        true_theory = self.agent.env.unwrapped.reward_weights.float()
        return self.p_harm_given_single_theory(true_theory, action)


class PosteriorGuardrail(Guardrail):

    def __init__(self, agent, threshold):
        super().__init__(agent, threshold)

    def harm_estimate(self, action):

        return self.marginal_p_harm(action)


class IidGuardrail(Guardrail):

    def __init__(self, agent, threshold, tiebreak="max"):
        super().__init__(agent, threshold)
        self.tiebreak = tiebreak

    def harm_estimate(self, action):

        p_theory = t.exp(self.agent.log_posterior)
        p_harm_given_theory = self.p_harm_given_theory(action)
        plausible_harm = p_theory * p_harm_given_theory
        indices = t.argmax(plausible_harm)
        harms_of_argmax_theories = p_harm_given_theory[indices]
        harm_estimate = t.max(
            harms_of_argmax_theories
        )  # if there are multiple argmax plausible-harm theories with different harm estimates, we take the max harm estimate
        return harm_estimate


class NonIidGuardrail(Guardrail):

    def __init__(self, agent, threshold, alpha):
        super().__init__(agent, threshold)
        self.alpha = alpha

    def m_alpha(self):
        posterior = t.exp(self.agent.log_posterior)
        sorted_posterior, sorted_indices = t.sort(posterior, descending=True)
        cumulative_sorted_posterior = t.cumsum(sorted_posterior, dim=0)
        included = sorted_posterior >= self.alpha * cumulative_sorted_posterior
        m_alpha = t.empty_like(included)
        m_alpha[sorted_indices] = included
        return m_alpha

    def harm_estimate(self, action):
        m_alpha = self.m_alpha()
        p_harm_given_theory_m_alpha = self.p_harm_given_theory(action)[m_alpha]
        assert len(p_harm_given_theory_m_alpha) == m_alpha.sum()
        if self.alpha == 1.0:
            assert len(p_harm_given_theory_m_alpha) == 1
        if self.alpha == 0.0:
            assert len(p_harm_given_theory_m_alpha) == len(self.agent.log_posterior)
        harm_estimate = t.max(p_harm_given_theory_m_alpha)

        return harm_estimate

class NewNonIidGuardrail(Guardrail):

    def __init__(self, agent, threshold, alpha):
        super().__init__(agent, threshold)
        self.alpha = alpha

    def m_alpha(self):
        posterior = t.exp(self.agent.log_posterior)
        max_indices = t.argmax(posterior)
        m_alpha = t.zeros_like(posterior, dtype=t.bool)
        if max_indices.dim() == 0:
            m_alpha[max_indices.item()] = True
        else:
            m_alpha[max_indices[0]] = True
        m_alpha |= (posterior >= self.alpha / len(self.agent.log_posterior))
        return m_alpha

    def harm_estimate(self, action):
        m_alpha = self.m_alpha()
        p_harm_given_theory_m_alpha = self.p_harm_given_theory(action)[m_alpha]
        assert len(p_harm_given_theory_m_alpha) == m_alpha.sum()
        # if self.alpha == 1.0:
        #     assert len(p_harm_given_theory_m_alpha) == 1
        # if self.alpha == 0.0:
        #     assert len(p_harm_given_theory_m_alpha) == len(self.agent.log_posterior)
        harm_estimate = t.max(p_harm_given_theory_m_alpha)
        return harm_estimate