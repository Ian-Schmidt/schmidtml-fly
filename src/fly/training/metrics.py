"""How the fly is measured: choice accuracy and what happens in the brain per neuron class."""

from typing import TypedDict

import torch
import torch.nn.functional as F
from torch import Tensor

from fly.model.brain import CLASSES, READ_LAST, FlyBrain
from fly.utils.config import load

ACTIVE_RATE: float = load("training")["active_rate"]  # the rate above which a neuron counts as active


class EvalMetrics(TypedDict):
    """Quality of the fly on a set of tasks.

    Attributes:
        acc: Share of tasks where the most attractive option is one of the correct ones. The headline number.
        loss: Cross-entropy with soft targets.
        p_correct: How much softmax probability goes to the correct options on average — a soft version
            of accuracy: it grows even while the argmax has not flipped yet.
    """

    acc: float
    loss: float
    p_correct: float


class ClassStats(TypedDict):
    """State of one neuron class.

    Attributes:
        rate: Mean firing rate in the readout window — where the signal reaches.
        active: Share of neurons with a rate above 0.05 — sparsity of the code.
        gain: Mean |log_gain| — how strongly training retuned the class.
        bias: Mean threshold — whether the neurons became more or less excitable.
    """

    rate: float
    active: float
    gain: float
    bias: float


def accuracy(scores: Tensor, t: Tensor) -> float:
    """Computes choice accuracy.

    Args:
        scores: Option scores `[B, K]`.
        t: Targets `[B, K]`, positive at the correct options.

    Returns:
        Share of tasks where the most attractive option is one of the correct ones.

    Example:
        >>> scores = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        >>> accuracy(scores, torch.tensor([[0.0, 1.0], [0.0, 1.0]]))
        0.5
    """
    return (t.gather(1, scores.argmax(1, keepdim=True)) > 0).float().mean().item()


def evaluate(brain: FlyBrain, q: Tensor, a: Tensor, t: Tensor, idx: Tensor, bs: int) -> EvalMetrics:
    """Runs the fly over tasks without training.

    Args:
        brain: The fly brain.
        q: Embeddings of all the level's questions `[n, D]`.
        a: Embeddings of all the options `[n, K, D]`.
        t: Targets `[n, K]`.
        idx: Indices of the tasks to evaluate on.
        bs: Batch size; every question is K whole-brain simulations, so watch the memory.

    Returns:
        Accuracy, loss and confidence on the selected tasks.
    """
    with torch.no_grad():
        s = torch.cat([brain.choose(q[b], a[b]) for b in idx.split(bs)])
    return {"acc": accuracy(s, t[idx]), "loss": F.cross_entropy(s, t[idx]).item(),
            "p_correct": (s.softmax(1) * (t[idx] > 0)).sum(1).mean().item()}


def brain_stats(brain: FlyBrain, q: Tensor, a: Tensor, cls: Tensor) -> dict[str, ClassStats]:
    """Takes a snapshot of the brain per neuron class.

    Args:
        brain: The fly brain.
        q: Embeddings of the probe questions `[B, D]`.
        a: Embeddings of their options `[B, K, D]`.
        cls: Class number of every neuron from `fly.classify`.

    Returns:
        Statistics keyed by the class key from `fly.CLASSES`. A class with no neurons gets NaN values.
    """
    with torch.no_grad():
        _, trace = brain.choose(q, a, record=True)
        rate = trace[-READ_LAST:].mean((0, 2))
        stats: dict[str, ClassStats] = {}
        for i, neuron_class in enumerate(CLASSES):
            m = cls == i
            stats[neuron_class.key] = {
                "rate": rate[m].mean().item(), "active": (rate[m] > ACTIVE_RATE).float().mean().item(),
                "gain": brain.log_gain[m].abs().mean().item(), "bias": brain.bias[m].mean().item()}
    return stats
