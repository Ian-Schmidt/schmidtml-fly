import torch
import torch.nn.functional as F

import fly
from conftest import N
from fly.data import tasks
from fly.training import baselines


def similarity_task(n=60, k=4, d=16, seed=0):
    """A similarity task, like level 2: the correct option is a noisy copy of the question."""
    g = torch.Generator().manual_seed(seed)
    q = F.normalize(torch.randn(n, d, generator=g), dim=-1)
    a = F.normalize(torch.randn(n, k, d, generator=g), dim=-1)
    y = torch.randint(0, k, (n,), generator=g)
    a[torch.arange(n), y] = F.normalize(q + 0.1 * torch.randn(n, d, generator=g), dim=-1)
    return q, a, F.one_hot(y, k).float(), *tasks.split(n)


def test_references_on_similarity_task(edges, ann):
    senses = {"smell": (torch.arange(N), torch.arange(N))}  # wide input: every neuron has its own channel
    b = fly.FlyBrain(*fly.wiring(edges, N), senses, fly.descending(ann), emb_dim=16)
    q, a, t, tr, te = similarity_task()
    ref = baselines.references(b, q, a, t, tr, te, epochs=50)
    assert ref["chance"] == 0.25 and ref["cosine"] == 1.0, "сходство решает задачу без обучения"
    assert ref["linear_on_input_test"] > 0.8, "широкий вход сохраняет геометрию — потолок высокий"


def test_linear_on_input_with_two_senses(edges, ann):
    """Different senses for question and answer -> the branch with a C×C channel-matching matrix."""
    b = fly.FlyBrain(*fly.wiring(edges, N), {k: fly.sense(ann, k) for k in ("smell", "sight")},
                     fly.descending(ann), emb_dim=16)
    q, a, t, tr, te = similarity_task()
    acc_tr, acc_te = baselines.linear_on_input(b, q, a, t, tr, te, steps=30, decays=(0.01,))
    assert 0 <= acc_te <= 1 and acc_tr > 0.25
