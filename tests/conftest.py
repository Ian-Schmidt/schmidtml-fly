"""A toy brain of 300 neurons: the tests need neither the downloaded connectome nor any question data."""

import numpy as np
import pandas as pd
import pytest
import torch

import fly
from fly.data.tasks import Task

N = 300


def toy_edges(n=N, e=4000, seed=0):
    rng = np.random.default_rng(seed)
    pre, post = rng.integers(0, n, e), rng.integers(0, n, e)
    return pd.DataFrame({"Presynaptic_Index": pre, "Postsynaptic_Index": post,
                         "Connectivity": rng.integers(5, 30, e), "Excitatory": np.where(pre % 5 == 0, -1, 1)})


def toy_ann(n=N):
    """Annotations in the FlyWire format: 30 olfactory receptors of 10 types, 40 photoreceptors, 20 descending."""
    ann = pd.DataFrame({"pos_x": np.arange(n, dtype=float), "pos_y": 1.0, "pos_z": 2.0,
                        "super_class": "central", "cell_class": None, "cell_type": None})
    ann.loc[:29, ["super_class", "cell_class"]] = ["sensory", "olfactory"]
    ann.loc[:29, "cell_type"] = [f"ORN_{i % 10}" for i in range(30)]
    ann.loc[30:69, ["super_class", "cell_class"]] = ["sensory", "visual"]
    ann.loc[70:79, "cell_class"] = "Kenyon_Cell"
    ann.loc[80:84, "cell_type"] = "MBON01"
    ann.loc[100:119, "super_class"] = "optic"
    ann.loc[280:, "super_class"] = "descending"
    return ann


def task(answer, q="q", options=("a", "b", "c", "d"), **extra) -> Task:
    """A task with sensible default fields: a test shows only what it actually checks."""
    return Task(q=q, options=list(options), answer=answer, **extra)


@pytest.fixture
def edges():
    return toy_edges()


@pytest.fixture
def ann():
    return toy_ann()


@pytest.fixture
def brain(edges, ann):
    torch.manual_seed(0)
    return fly.FlyBrain(*fly.wiring(edges, N), {"smell": fly.sense(ann, "smell")}, fly.descending(ann), emb_dim=16)


@pytest.fixture
def quiz():
    """8 questions with 4 options each: (q [8, 16], a [8, 4, 16], soft targets [8, 4])."""
    g = torch.Generator().manual_seed(0)
    q = torch.nn.functional.normalize(torch.randn(8, 16, generator=g), dim=-1)
    a = torch.nn.functional.normalize(torch.randn(8, 4, 16, generator=g), dim=-1)
    t = torch.nn.functional.one_hot(torch.randint(0, 4, (8,), generator=g), 4).float()
    return q, a, t
