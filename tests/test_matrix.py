import numpy as np

import fly
from conftest import N
from fly.viz import matrix


def test_binned_matrix_keeps_every_synapse(edges, ann):
    cls = fly.classify(ann)
    m, bounds = matrix.binned_matrix(edges, cls, bins=30)
    assert m.shape == (30, 30) and m.sum() == edges.Connectivity.sum()
    assert bounds[0] == 0 and bounds[-1] == 30 and (np.diff(bounds) >= 0).all()
    assert bounds[1] == 3, "30 обонятельных рецепторов из 300 — первые 3 бина из 30"


def test_class_matrix_rows_are_input_shares(edges, ann):
    share = matrix.class_matrix(edges, fly.classify(ann))
    sums = share.sum(1)
    assert share.shape == (len(fly.CLASSES),) * 2
    assert np.allclose(sums[sums > 0], 1) and (sums == 0).any(), "класс без нейронов в игрушечном мозге — нули"


def test_reciprocity_of_random_graph_is_small_but_not_zero(edges):
    assert 0 < matrix.reciprocity(edges) < 0.2


def test_plot_writes_png(edges, ann, tmp_path):
    path = tmp_path / "docs" / "matrix.png"
    matrix.plot(edges, fly.classify(ann), path, bins=N // 10)
    assert path.read_bytes()[:4] == b"\x89PNG"
