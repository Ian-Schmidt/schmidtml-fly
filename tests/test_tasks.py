import json
import sys
import types

import pytest
import torch

from conftest import task
from fly.data import tasks


@pytest.fixture
def site(tmp_path, monkeypatch):
    monkeypatch.setattr(tasks, "LEVELS", tmp_path / "levels")
    monkeypatch.setattr(tasks, "CACHE", tmp_path / "cache")
    return tmp_path


def write_level(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, ensure_ascii=False))


def test_fingerprint_tracks_content():
    item = task([0], options=["a", "b"])
    same = tasks.fingerprint([task([0], options=["a", "b"], id="не влияет")])
    assert tasks.fingerprint([item]) == same and len(same) == 10
    assert tasks.fingerprint([task([1], options=["a", "b"])]) != same


def test_load_level_by_name(site):
    write_level(site / "levels" / "quiz.json", [task([0, 2], q="Что выведет?", topic="python")])
    items = tasks.load("quiz")
    assert items[0]["q"] == "Что выведет?" and items[0]["answer"] == [0, 2] and items[0]["topic"] == "python"
    with pytest.raises(FileNotFoundError):
        tasks.load("interview")


def test_load_level_by_path_and_reject_uneven_options(tmp_path):
    path = tmp_path / "level.json"
    write_level(path, [task([0], options=["a", "b"])])
    assert tasks.load(str(path))[0]["options"] == ["a", "b"]
    write_level(path, [task([0], options=["a", "b"]), task([0], options=["a"])])
    with pytest.raises(ValueError, match="одинаковое число вариантов"):
        tasks.load(str(path))


def test_stage_keys_are_unique():
    keys = [s.key for s in tasks.STAGES]
    assert len(keys) == len(set(keys)) == 7


def test_embed_encodes_once_then_reads_cache(site, monkeypatch):
    calls = []

    class FakeEncoder:
        def __init__(self, name):
            assert name == tasks.ENCODER

        def encode(self, texts, **_):
            calls.append(texts)
            g = torch.Generator().manual_seed(len(texts))
            return (torch.randn(len(texts), 8, generator=g) + 3).numpy()  # a shared offset — like e5's anisotropy

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeEncoder))
    items = [task([0], q=f"q{i}", options=[f"o{i}{k}" for k in range(4)]) for i in range(6)]

    q, a = tasks.embed(items, "quiz")
    assert q.shape == (6, 8) and a.shape == (6, 4, 8)
    assert calls[0][0] == "query: q0" and calls[1][0] == "passage: o00", "префиксы e5 для поиска"
    assert torch.allclose(q.norm(dim=-1), torch.ones(6)) and torch.allclose(a.norm(dim=-1), torch.ones(6, 4))
    cos = (q[:, None] * a).sum(-1).mean()
    assert abs(cos) < 0.3, f"после центрирования тексты не смотрят в одну сторону, косинус {cos:.2f}"

    q2, _ = tasks.embed(items, "quiz")
    assert len(calls) == 2 and torch.equal(q, q2), "второй вызов читает кеш"
    assert list((site / "cache").glob("quiz-*.pt"))


def test_split_is_disjoint_and_seeded():
    tr, te = tasks.split(100)
    assert len(tr) == 80 and len(te) == 20
    assert sorted(tr.tolist() + te.tolist()) == list(range(100))
    assert torch.equal(tasks.split(100)[1], te) and not torch.equal(tasks.split(100, seed=1)[1], te)


def test_targets_share_probability_between_correct_answers():
    t = tasks.targets([task([1]), task([0, 2])], 4)
    assert t.tolist() == [[0, 1, 0, 0], [0.5, 0, 0.5, 0]]
