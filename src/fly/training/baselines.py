"""No-fly baselines: what the same embeddings are worth without the connectome — to see honestly what the brain adds."""

from collections.abc import Callable, Sequence
from typing import TypedDict

import torch
import torch.nn.functional as F
from torch import Tensor

from fly.model.brain import FlyBrain
from fly.training.metrics import accuracy
from fly.utils.config import load

_CFG = load("training")["baselines"]


class References(TypedDict):
    """No-fly baselines on the same split as the fly.

    Attributes:
        chance: Chance level: mean share of correct options on the exam.
        cosine: Picking the option most similar to the question by embedding cosine — no training.
        bilinear_train: Linear model on the full embedding, training accuracy.
        bilinear_test: Same on the exam — how much the same vectors yield without the fly.
        linear_on_input_train: Linear model on the fly's sense channels, training accuracy.
        linear_on_input_test: Same on the exam — the input ceiling, the fair bar for the brain.
    """

    chance: float
    cosine: float
    bilinear_train: float
    bilinear_test: float
    linear_on_input_train: float
    linear_on_input_test: float


def linear_on_input(
    brain: FlyBrain, q: Tensor, a: Tensor, t: Tensor, tr: Tensor, te: Tensor,
    steps: int = _CFG["input_steps"], decays: Sequence[float] = tuple(_CFG["input_decays"]),
) -> tuple[float, float]:
    """Computes the input ceiling: a linear scorer on exactly what the fly receives — its sense channels.

    This is the fair bar for the brain: whatever is above it was already lost by the "nose", whatever
    is below it is lost by the brain. Regularization is tuned on a held-out part of the training set;
    the exam is never seen during tuning.

    Args:
        brain: The fly brain; only its "nose" is used — the projection of text onto sense channels.
        q: Question embeddings `[n, D]`.
        a: Option embeddings `[n, K, D]`.
        t: Targets `[n, K]`.
        tr: Training indices.
        te: Exam indices.
        steps: Adam steps per scorer fit.
        decays: Weight-decay candidates.

    Returns:
        Training and exam accuracy. When question and option use different senses, the channel
        correspondence is learned by a C×C matrix with more parameters than data, so the estimate is
        biased low.
    """
    dev = brain.bias.device
    with torch.no_grad():
        gq = brain.code(q.to(dev), brain.q_sense)
        ga = brain.code(a.reshape(-1, a.shape[-1]).to(dev), brain.a_sense).view(*a.shape[:2], -1)
    y = t.to(dev)
    cut = int(len(tr) * (1 - _CFG["input_val_frac"]))
    fit, val = tr[:cut], tr[cut:]

    same = gq.shape[1] == ga.shape[2]   # одно чувство на оба текста → сходство каналов сравнивается напрямую

    def trained(idx: Tensor, wd: float) -> Callable[[Tensor], Tensor]:
        # При общем чувстве хватает 3·C весов (поэлементное произведение каналов). При разных чувствах
        # соответствие каналов приходится учить матрицей C×C: на широком входе её параметров больше, чем
        # данных, и такая оценка потолка занижена — в DOCS это оговорено.
        u = torch.zeros(ga.shape[2], device=dev, requires_grad=True)
        v = torch.zeros(gq.shape[1], device=dev, requires_grad=True)
        W = torch.zeros(gq.shape[1], ga.shape[2] if not same else 0, device=dev, requires_grad=True)
        w = torch.zeros(gq.shape[1] if same else 0, device=dev, requires_grad=True)
        score = (lambda i: (gq[i][:, None] * ga[i]) @ w + ga[i] @ u + (gq[i] @ v)[:, None]) if same else (
            lambda i: ((gq[i] @ W)[:, None] * ga[i]).sum(-1) + ga[i] @ u + (gq[i] @ v)[:, None])
        opt = torch.optim.Adam([W, w, u, v], _CFG["lr"], weight_decay=wd)
        for _ in range(steps):
            loss = F.cross_entropy(score(idx), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        return score

    def acc_of(score: Callable[[Tensor], Tensor], idx: Tensor) -> float:
        with torch.no_grad():
            return accuracy(score(idx), y[idx])

    best = max(decays, key=lambda wd: acc_of(trained(fit, wd), val))
    score = trained(tr, best)   # с выбранной регуляризацией — заново на всей обучающей части
    return acc_of(score, tr), acc_of(score, te)


def references(
    brain: FlyBrain, q: Tensor, a: Tensor, t: Tensor, tr: Tensor, te: Tensor, epochs: int = _CFG["bilinear_epochs"]
) -> References:
    """Computes what the answers are worth without the fly — to see what the brain adds and what the "nose" (e5) does.

    Args:
        brain: The fly brain; needed only for the input ceiling.
        q: Question embeddings `[n, D]`.
        a: Option embeddings `[n, K, D]`.
        t: Targets `[n, K]`.
        tr: Training indices.
        te: Exam indices.
        epochs: Adam steps for the linear model on the full embedding.

    Returns:
        All baselines; computed once before training and logged to MLflow as `reference/*`.
    """
    cosine = accuracy((q[te, None] * a[te]).sum(-1), t[te])
    w = torch.zeros(q.shape[1], requires_grad=True)
    u = torch.zeros(q.shape[1], requires_grad=True)
    score = lambda i: (q[i, None] * a[i]) @ w + a[i] @ u  # noqa: E731
    opt = torch.optim.Adam([w, u], _CFG["lr"])
    for _ in range(epochs):
        loss = F.cross_entropy(score(tr), t[tr])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        chance = (t[te] > 0).float().mean().item()
        bilinear_train, bilinear_test = accuracy(score(tr), t[tr]), accuracy(score(te), t[te])
    input_train, input_test = linear_on_input(brain, q, a, t, tr, te)
    return {"chance": chance, "cosine": cosine, "bilinear_train": bilinear_train, "bilinear_test": bilinear_test,
            "linear_on_input_train": input_train, "linear_on_input_test": input_test}
