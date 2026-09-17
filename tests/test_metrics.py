import math

import pytest
import torch

import fly
from conftest import task
from fly.data import tasks
from fly.training import metrics


def test_accuracy_counts_any_correct_answer():
    t = tasks.targets([task([0, 2]), task([1])], 3)
    scores = torch.tensor([[0.1, 0.2, 0.9], [0.8, 0.1, 0.1]])
    assert metrics.accuracy(scores, t) == 0.5


def test_evaluate(brain, quiz):
    q, a, t = quiz
    m = metrics.evaluate(brain, q, a, t, torch.arange(8), bs=3)
    assert set(m) == {"acc", "loss", "p_correct"}
    assert 0 <= m["acc"] <= 1 and 0 < m["p_correct"] < 1
    assert m["loss"] == pytest.approx(math.log(4), abs=0.2), "необученная муха выбирает почти наугад"


def test_brain_stats_per_class(brain, quiz, ann):
    q, a, _ = quiz
    with torch.no_grad():
        brain.log_gain[:30] = 0.5  # «перенастроили» обонятельные рецепторы
    stats = metrics.brain_stats(brain, q[:2], a[:2], torch.tensor(fly.classify(ann)))
    assert stats["orn"]["gain"] == pytest.approx(0.5) and stats["central"]["gain"] == 0
    assert stats["orn"]["rate"] > 0 and 0 <= stats["dn"]["active"] <= 1
