import numpy as np
import torch as t
import einops
from agents import agents as agents
from utils.utils_numerical import log_survival_function


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

        return t.dot(
            p_theory, self.p_harm_given_theory(action)
        )  # n_hypotheses, n_hypotheses ->

    def p_harm_given_theory(self, action):

        arm_features = self.agent.env.unwrapped.arm_features[action]
        reward_means_given_theory = t.mv(
            self.agent.hypotheses, arm_features
        )  # n_hypotheses d_arm, d_arm -> n_hypotheses

        p_harm_given_theory = 1 - t.distributions.Normal(
            loc=reward_means_given_theory, scale=self.agent.env.unwrapped.sigma_r
        ).cdf(self.agent.env.unwrapped.explosion_threshold)

        return p_harm_given_theory

    def log_p_harm_given_theory(self, action):
        arm_features = self.agent.env.unwrapped.arm_features[action]
        reward_means_given_theory = t.mv(
            self.agent.hypotheses, arm_features
        )  # n_hypotheses d_arm, d_arm -> n_hypotheses

        # # Calculate log(1 - CDF) directly for numerical stability
        # log_p_harm_given_theory = t.distributions.Normal(
        #     loc=reward_means_given_theory, scale=self.agent.env.unwrapped.sigma_r
        # ).log_survival_function(self.agent.env.unwrapped.explosion_threshold)
        explosion_threshold = self.agent.env.unwrapped.explosion_threshold
        sigma_r = self.agent.env.unwrapped.sigma_r

        log_p_harm_given_theory = log_survival_function(
            explosion_threshold, loc=reward_means_given_theory, scale=sigma_r
        )

        return log_p_harm_given_theory

    def p_harm_given_single_theory(self, theory, action):
        arm_features = self.agent.env.unwrapped.arm_features[action]
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
    def __init__(
        self,
        agent,
        threshold,
        alpha,
        mean_type="arithmetic",
        posterior_increases=False,
        softmax_temperature=None,
        power_mean_exponent=1.0,
        quantile=0.8,
        harm_estimates_weights=None,
    ):
        super().__init__(agent, threshold)
        self.alpha = alpha
        self.mean_type = mean_type
        self.posterior_increases = posterior_increases
        self.softmax_temperature = softmax_temperature
        self.power_mean_exponent = power_mean_exponent
        self.quantile = quantile
        if harm_estimates_weights is not None:
            total_weight = (
                harm_estimates_weights["max"]
                + harm_estimates_weights["mean"]
                + harm_estimates_weights["quantile"]
            )
            if total_weight == 0:
                raise ValueError("Sum of harm_estimates_weights cannot be zero.")
            self.harm_estimates_weights = {
                "max": harm_estimates_weights["max"] / total_weight,
                "mean": harm_estimates_weights["mean"] / total_weight,
                "quantile": harm_estimates_weights["quantile"] / total_weight,
            }
        else:
            self.harm_estimates_weights = {
                "max": 1.0,
                "mean": 0.0,
                "quantile": 0.0,
            }

    def m_alpha(self, posterior=None):
        if posterior is None:
            posterior = t.exp(self.agent.log_posterior)
        max_indices = t.argmax(posterior)
        m_alpha = t.zeros_like(posterior, dtype=t.bool)
        if max_indices.dim() == 0:
            m_alpha[max_indices.item()] = True
        else:
            m_alpha[max_indices[0]] = True
        m_alpha |= posterior >= self.alpha

        # # Sample additional hypotheses where posterior < alpha
        # low_posterior_mask = (posterior < self.alpha)
        # low_posterior_indices = low_posterior_mask.nonzero(as_tuple=True)[0]
        # if low_posterior_indices.numel() > 0:
        #     low_posterior_values = posterior[low_posterior_indices]
        #     low_posterior_probs = low_posterior_values / low_posterior_values.sum()
        #     num_samples = min(self.num_samples, low_posterior_indices.numel())
        #     sampled_indices = t.multinomial(low_posterior_probs, num_samples, replacement=False)
        #     sampled_indices = low_posterior_indices[sampled_indices]
        #     m_alpha[sampled_indices] = True

        return m_alpha

    def harm_estimate(self, action):
        posterior = t.exp(self.agent.log_posterior)
        m_alpha = self.m_alpha(posterior)
        harm_estimate = self.balance_harm_estimates(
            posterior,
            action,
            m_alpha,
            harm_estimates_weights=self.harm_estimates_weights,
            mean_type=self.mean_type,
            posterior_increases=self.posterior_increases,
            softmax_temperature=self.softmax_temperature,
            power_mean_exponent=self.power_mean_exponent,
        )
        return harm_estimate

    def balance_harm_estimates(
        self,
        posterior,
        action,
        m_alpha,
        harm_estimates_weights=None,
        mean_type="arithmetic",
        posterior_increases=False,
        softmax_temperature=None,
        power_mean_exponent=1.0,
    ):
        total_weight = (
            harm_estimates_weights["max"]
            + harm_estimates_weights["mean"]
            + harm_estimates_weights["quantile"]
        )
        if total_weight == 0:
            raise ValueError("Sum of harm_estimates_weights cannot be zero.")

        if harm_estimates_weights["max"] > 0:
            max_harm = self.max_harm_estimate(action, m_alpha)
        else:
            max_harm = 0.0

        if harm_estimates_weights["mean"] > 0:
            mean_harm = self.posterior_mean_harm_estimate(
                posterior,
                action,
                m_alpha,
                mean_type,
                posterior_increases,
                softmax_temperature,
                power_mean_exponent,
            )
        else:
            mean_harm = 0.0

        if harm_estimates_weights["quantile"] > 0:
            quantile_harm = self.quantile_harm_estimate(
                action, m_alpha, quantile=self.quantile
            )
        else:
            quantile_harm = 0.0

        harm_estimate = (
            harm_estimates_weights["max"] * max_harm
            + harm_estimates_weights["mean"] * mean_harm
            + harm_estimates_weights["quantile"] * quantile_harm
        )

        return harm_estimate

    def quantile_harm_estimate(self, action, m_alpha, quantile=0.8):
        p_harm_given_theory_m_alpha = self.p_harm_given_theory(action)[m_alpha]
        return t.quantile(p_harm_given_theory_m_alpha, quantile)

    def max_harm_estimate(self, action, m_alpha):
        p_harm_given_theory_m_alpha = self.p_harm_given_theory(action)[m_alpha]
        return t.max(p_harm_given_theory_m_alpha)

    def posterior_mean_harm_estimate(
        self,
        posterior,
        action,
        m_alpha,
        mean_type="arithmetic",
        posterior_increases=False,
        softmax_temperature=None,
        power_mean_exponent=1.0,
        epsilon=1e-12, # to avoid division by zero
    ):
        selected_posteriors = posterior[m_alpha]
        if posterior_increases:
            # Weighted by increases in posteriors
            selected_priors = t.exp(self.agent.log_prior)[m_alpha]
            weights = t.clamp(selected_posteriors - selected_priors, min=0)
        else:
            weights = selected_posteriors

        if t.allclose(weights, t.zeros_like(weights)):
            # If all weights are zero, use the harm estimate of the top posterior
            p_harm_given_theory_m_alpha = self.p_harm_given_theory(action)[m_alpha]
            harm_estimate = t.max(
                p_harm_given_theory_m_alpha[t.argmax(selected_posteriors)]
            )
            return harm_estimate

        if softmax_temperature is not None:
            weights = t.softmax(weights / softmax_temperature, dim=0)

        if mean_type == "arithmetic":
            p_harm_given_theory_m_alpha = self.p_harm_given_theory(action)[m_alpha]
            p_harm_given_theory_m_alpha = p_harm_given_theory_m_alpha ** power_mean_exponent
            harm_estimate = (
                t.dot(weights, p_harm_given_theory_m_alpha)
                / weights.sum()
            ) ** (1 / power_mean_exponent)
        elif mean_type == "geometric":
            log_p_harm_given_theory_m_alpha = power_mean_exponent * self.log_p_harm_given_theory(action)[
                m_alpha
            ]
            harm_estimate = t.exp(
                t.dot(weights, log_p_harm_given_theory_m_alpha)
                / (weights.sum() * power_mean_exponent)
            )
        elif mean_type == "harmonic":
            p_harm_given_theory_m_alpha = self.p_harm_given_theory(action)[m_alpha]
            p_harm_given_theory_m_alpha = p_harm_given_theory_m_alpha ** power_mean_exponent
            harm_estimate = weights.sum() / t.sum(
                weights / p_harm_given_theory_m_alpha
            )
            harm_estimate = harm_estimate ** (1 / power_mean_exponent)
        else:
            raise ValueError(f"Unknown mean_type: {mean_type}")
        return harm_estimate
