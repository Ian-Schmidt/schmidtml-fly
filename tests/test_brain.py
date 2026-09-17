import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

import fly
from conftest import N, toy_ann, toy_edges
from fly.model import brain as brain_module


def test_wiring_sign_belongs_to_neuron(edges):
    """Dale's law: the sign belongs to the neuron, so all its outgoing connections share a sign."""
    post, pre, syn, sign = fly.wiring(edges, N)
    assert len(post) == len(pre) == len(syn) == len(edges)
    assert set(np.unique(sign)) <= {-1.0, 0.0, 1.0}
    assert (sign[pre[pre % 5 == 0]] == -1).all() and (sign[pre[pre % 5 != 0]] == 1).all()


def test_weights_are_normalized_by_total_input(brain):
    """Every neuron gets a weighted mean of its inputs: the sum of |w| over its inputs is 1."""
    total = torch.zeros(N).index_add(0, brain.W.indices()[0], brain.W.values().abs())
    assert torch.allclose(total[total > 0], torch.ones_like(total[total > 0]), atol=1e-5)


def test_sense_channels(ann):
    idx, code = fly.sense(ann, "smell")
    assert idx.tolist() == list(range(30))
    assert code.max() + 1 == 10, "рецепторы одного типа делят канал (клубочек)"
    assert code[0] == code[10] == code[20]

    idx, code = fly.sense(ann, "sight")
    assert idx.tolist() == list(range(30, 70))
    assert code.tolist() == list(range(40)), "у фоторецептора канал свой: это пиксель"


def test_descending_and_classify(ann):
    assert fly.descending(ann).tolist() == list(range(280, 300))
    keys = [c[0] for c in fly.CLASSES]
    cls = fly.classify(ann)
    assert keys[cls[0]] == "orn", "приоритет: обонятельный рецептор — orn, а не sensory"
    assert keys[cls[30]] == "sensory" and keys[cls[70]] == "kc" and keys[cls[80]] == "mbon"
    assert keys[cls[100]] == "optic" and keys[cls[290]] == "dn" and keys[cls[200]] == "central"


def test_smell_drives_only_receptors(brain):
    emb = F.normalize(torch.randn(3, 16), dim=-1)
    i = brain.smell(emb)
    assert i.shape == (N, 3)
    assert (i[30:] == 0).all() and (i[:30] >= 0).all() and i[:30].sum() > 0
    assert torch.equal(i[0], i[10]), "рецепторы одного клубочка получают один и тот же ток"


def test_senses_inject_equal_total_current(edges, ann):
    """40 photoreceptors must not shout down 30 olfactory ones: scale is inverse to the sense's size."""
    senses = {k: fly.sense(ann, k) for k in ("smell", "sight")}
    b = fly.FlyBrain(*fly.wiring(edges, N), senses, fly.descending(ann), emb_dim=16)
    assert b.q_sense == "smell" and b.a_sense == "sight"
    assert b.channels("smell") == 10 and b.channels("sight") == 40
    assert float(b.get_buffer("scale_smell") * 30) == pytest.approx(float(b.get_buffer("scale_sight") * 40))


def test_forward_trace_and_schedule(brain, quiz):
    q, a, _ = quiz
    score, trace = brain(q, a[:, 0], record=True)
    assert score.shape == (8,) and trace.shape == (fly.STEPS, N, 8)
    assert (trace >= 0).all() and (trace <= 1).all(), "частота разрядов ограничена"
    _, other = brain(q, a[:, 1], record=True)
    assert torch.equal(trace[: fly.ANSWER_AT], other[: fly.ANSWER_AT]), "до ANSWER_AT муха нюхает только вопрос"
    assert not torch.equal(trace[-1], other[-1])


def test_choose_shapes_and_gradients(brain, quiz):
    q, a, _ = quiz
    scores, trace = brain.choose(q, a, record=True)
    assert scores.shape == (8, 4) and trace.shape == (fly.STEPS, N, 32)
    assert torch.allclose(scores, brain.choose(q, a))
    scores.sum().backward()
    assert {n for n, p in brain.named_parameters() if p.grad is not None} == {
        "log_gain", "bias", "readout.weight", "readout.bias"}, "проводка не обучается"


def test_brain_memorizes_toy_quiz(brain, quiz):
    q, a, t = quiz
    opt = torch.optim.Adam(brain.parameters(), 3e-2)
    for _ in range(150):
        loss = F.cross_entropy(brain.choose(q, a), t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert (brain.choose(q, a).argmax(1) == t.argmax(1)).all(), loss.item()


def test_load_connectome_filters_weak_edges(tmp_path, monkeypatch):
    ids = np.array([11, 22, 33])
    pd.DataFrame({"id": ids, "x": 0}).to_csv(tmp_path / "Completeness_783.csv", index=False)
    pd.DataFrame({"Presynaptic_Index": [0, 1, 2], "Postsynaptic_Index": [1, 2, 0],
                  "Connectivity": [fly.MIN_SYN - 1, fly.MIN_SYN, 40], "Excitatory": [1, -1, 1],
                  "extra": 0}).to_parquet(tmp_path / "Connectivity_783.parquet")
    rows = toy_ann().head(3).assign(root_id=[33, 11, 22])  # annotation order differs from neuron order
    rows.to_csv(tmp_path / "annotations.tsv", sep="\t", index=False)
    monkeypatch.setattr(brain_module, "DATA", tmp_path)

    edges, ann = fly.load_connectome()
    assert edges.Connectivity.tolist() == [fly.MIN_SYN, 40]
    assert ann.index.tolist() == [11, 22, 33] and ann.pos_x.tolist() == [1.0, 2.0, 0.0]


def test_build_on_toy_connectome(monkeypatch):
    monkeypatch.setattr(brain_module, "load_connectome", lambda: (toy_edges(), toy_ann()))
    b, ann = fly.build(senses=("smell", "sight"))
    assert b.n == N and b.senses == ["smell", "sight"] and len(ann) == N
    assert sum(p.numel() for p in b.parameters()) == 2 * N + 20 + 1
    assert fly.device() in {"mps", "cuda", "cpu"}
