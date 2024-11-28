# Conservative Bayesian Agent

<p align="center">
  <a href="https://arxiv.org/abs/2408.05284"><img src="http://img.shields.io/badge/arXiv-b31b1b.svg?style=flat-square&logo=arXiv" alt="arXiv"></a>&nbsp;
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54" alt="Python"></a>&nbsp;
  <a href="https://wandb.ai/site"><img src="https://img.shields.io/badge/W%26B-000000?style=flat-square&logo=weightsandbiases&logoColor=white" alt="W&B"></a>&nbsp;
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square" alt="License: MIT"></a>
</p>


The experiments for the paper [Can a Bayesian Oracle Prevent Harm from an Agent? (Bengio et al., 2024)](https://arxiv.org/abs/2408.05284).

## Features

- Implementation of different safety guardrails for a multi-armed bandit agent:
  - Cheating guardrail (using ground truth)
  - Posterior predictive guardrail
  - IID guardrail (Prop 3.4)
  - Non-IID guardrail (Prop 4.6)
  - New Non-IID guardrail with configurable harm estimation aggregation methods

- Configurable exploding bandit environment
- Plotting utilities
- Experimentation framework with:
  - Hyperparameter sweeps using Weights & Biases
  - Custom metrics for reward/safety trade-offs

## Installation

### Using conda

```bash
conda env create -f environment.yaml
```

### Using venv

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
.\venv\Scripts\activate  # On Windows
```

2. Install PyTorch:
```bash
# For CUDA 11.8
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch==2.3.1 torchvision torchaudio

# For MacOS with MPS (Apple Silicon)
pip install torch==2.2.2 torchvision torchaudio
```

3. Install the remaining requirements:
```bash
pip install -r requirements.txt
```

The code supports CPU, CUDA, and Apple Silicon (MPS) devices. The device can be configured in `configs/config.yaml` or via arguments:
- `device: "cpu"` - CPU
- `device: "cuda"` - NVIDIA GPU
- `device: "mps"` - Apple Silicon GPU
- `device: "auto"` - Automatically select best available device

## Running Experiments

### Basic Usage

#### Results in Figure 1

Run non-IID experiments with default parameters:
```bash
# Generates the results in Figure 1
python run_non_iid_experiment.py
```

Create plots from the results:
```bash
python make_non_iid_plots.py
```

To also include the experimental new non-IID guardrail in the comparison (not included in the main paper's figures):
```bash
python run_non_iid_experiment.py --include_new_non_iid
```

#### Results in Figure 2

Create plots from the results:
```bash
# Generates the results in Figure 2
python run_tightness_experiment.py
```

Create plots from the results:
```bash
python make_tightness_plots.py
```

#### New Non-IID Guardrails

For the new non-IID guardrails (implemented in the `NewNonIidGuardrail` class in [agents/guardrails.py](agents/guardrails.py)):

```bash
python run_new_non_iid.py
```

This script uses parameters defined in `configs/config.yaml`.

**Key Parameters in `configs/config.yaml`:**

-   `guardrail_new_non_iid.mean_type`:  Specifies the type of mean used for aggregating harm estimates (options: "arithmetic", "geometric", "harmonic").
-   `guardrail_new_non_iid.posterior_increases`:  A boolean determining whether the aggregation (for the harm estimate) is over posterior probabilities increases. What we mean by "increases" is the pointwise absolute value of the difference between the posterior probabilities and the prior (i.e. from the previous step) probabilities.
-   `guardrail_new_non_iid.softmax_temperature`:  The temperature parameter for the softmax function, controlling the distribution's sharpness.
-   `guardrail_new_non_iid.power_mean_exponent`:  The exponent applied in the power mean calculation.
-   `guardrail_new_non_iid.quantile`:  The quantile used in quantile-based harm estimation (e.g., 0.8 for the 80th percentile).
-   `guardrail_new_non_iid.harm_estimates_weights`:  Weights assigned to different harm estimation methods (max, mean, quantile).

An example of how these parameters can be varied for hyperparameter tuning is shown in `configs/sweep.yaml` (see the hyperparameter sweep section below).


**Harm Estimation Methods:**

The guardrail estimates harm using a combination of the following methods:

-   **Maximum Harm Estimate**: Uses the highest harm probability from the posterior.
-   **Quantile Harm Estimate**: Uses the specified quantile (e.g., `0.8`) of the harm probabilities.
-   **Mean Harm Estimate**: Uses arithmetic, geometric, or harmonic means based on the `mean_type` parameter, with a given power mean exponent (`power_mean_exponent`, defaulting to `1.0`) and/or softmax temperature (`softmax_temperature`, defaulting to `1.0`).

The weights for these methods are defined in `harm_estimates_weights`. These weights can be based on the posteriors of the theories or on the posterior increases (depending on `posterior_increases`), favoring theories with rapidly increasing posteriors.

The overall harm estimate is a weighted combination of these individual estimates. The goal is to find a harm bound that mitigates risky behaviors (high reward/high death rate) without resulting in overly conservative actions (low reward/low death rate).

**Custom Metric for Evaluation:**

A custom metric can be used to roughly approximate the performance of different `NewNonIidGuardrail` configurations:

```python
custom_metric = reward_mean / exp(2 * death_rate)
```

The point of this metric is to (roughly) prioritize higher rewards at the expense of increased death rates (trying to strike a balanced reward/safety trade-off). It is normalized against the score of a "cheating" guardrail, which should, by definition, have the highest possible metric value. This metric is useful for guiding hyperparameter search and evaluating whether various `NewNonIidGuardrail` configurations achieve a good balance between safety and performance.


### Hyperparameter Sweeps

Hyperparameter sweeps use Weights & Biases (W&B) to explore a range of guardrail configurations. 
You can view an example sweep instance [here](https://wandb.ai/ox/conservative-bayesian-agent?nw=nwuseryoukad).

Launch a W&B sweep to explore different `NewNonIidGuardrail` configurations:

```bash
wandb sweep configs/sweep.yaml
wandb agent <sweep_id>
```

### Configuration

Key parameters can be configured in `configs/config.yaml`:
- Experiment settings (episodes, thresholds, etc.)
  - `experiment.n_episodes`: Number of episodes to run
  - `experiment.episode_length`: Length of each episode
  - `experiment.n_actors`: Number of parallel actors to use (if null, calculated automatically)
  - `experiment.max_actor_fraction`: Fraction of CPU cores to use as maximum actors (default: 0.85)
- Environment parameters (dimensions, reward variance, etc.) 
- Guardrail parameters (mean types, weights, etc.)
- Plotting parameters (sizes, fonts, styles, etc.)

The parallel execution can be configured in two ways:
1. Set `experiment.n_actors` to a specific number to use exactly that many parallel actors
2. Set `experiment.max_actor_fraction` to control what fraction of CPU cores to use (e.g., 0.5 for 50% of cores)
3. Leave both as default to automatically use 85% of available CPU cores

## Testing

Run the test suite:
```bash
pytest
```

## Project Structure

```
.
├── agents/               # Agent and guardrail implementations
├── configs/             # Configuration files
├── envs/                # Environment implementations
├── utils/               # Helper functions and plotting
├── run_*.py            # Main experiment scripts
├── make_*_plots.py     # Plotting scripts
└── test_*.py           # Test files
```

## Citation

```bibtex
@article{bengio2024bayesian,
  title={Can a Bayesian Oracle Prevent Harm from an Agent?},
  author={Bengio, Yoshua and Cohen, Michael K and Malkin, Nikolay and MacDermott, Matthew and Fornasiere, Damiano and Greiner, Pietro and Kaddar, Younesse},
  journal={arXiv preprint arXiv:2408.05284},
  year={2024}
}
```