import json

import numpy as np
import pytest

from conftest import task
from fly.training import continual
from fly.viz import export as export_viz


def test_forgetting():
    # after stage 0: 90% on it; after stage 1 — 60%; at the end — 50%. Stage 1: 80% -> 70%. Stage 2 is last.
    R = [[0.9, 0.3, 0.2], [0.6, 0.8, 0.3], [0.5, 0.7, 0.9]]
    assert continual.forgetting(R) == pytest.approx([0.4, 0.1])


def test_split_by_group_keeps_question_cards_together():
    items = [task([0], group=f"q{i // 3}") for i in range(30)]  # 10 questions with 3 cards each
    idx = list(range(6, 30))
    tr, te = continual.split_by_group(items, idx)
    assert sorted(tr.tolist() + te.tolist()) == idx
    groups = lambda ix: {items[i]["group"] for i in ix.tolist()}  # noqa: E731
    assert not groups(tr) & groups(te) and len(groups(te)) == 1
    assert len(continual.split_by_group(items, [0, 1, 2, 3])[1]) == 3, "экзамен не бывает пустым"


def test_heatmap_writes_png(tmp_path):
    path = tmp_path / "forgetting.png"
    continual.heatmap([[0.5, 0.3], [0.4, 0.6]], ["База", "DL"], "Забывание", path)
    assert path.read_bytes()[:4] == b"\x89PNG"


def test_export_brain(ann, tmp_path, monkeypatch):
    monkeypatch.setattr(export_viz, "VIZ", tmp_path)
    ann.loc[5, "pos_x"] = np.nan  # a neuron without an annotation is not drawn
    export_viz.export_brain(ann)
    raw = (tmp_path / "brain.bin").read_bytes()
    n = len(ann)
    xyz = np.frombuffer(raw[: n * 12], np.float32).reshape(n, 3)
    cls = np.frombuffer(raw[n * 12 :], np.uint8)
    assert len(raw) == n * 13 and cls[5] == 255 and (xyz[5] == 0).all()
    assert abs(np.delete(xyz, 5, 0).mean(0)).max() < 1e-3, "облако центрировано"
    meta = json.loads((tmp_path / "brain.json").read_text())
    assert meta["n"] == n and meta["classes"][0]["key"] == "orn"
