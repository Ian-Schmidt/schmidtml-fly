"""The fly brain's connectivity matrix as a picture.

    uv run fly-plot-matrix                       # -> docs/connectome-matrix.png

The model has one connectivity matrix for the whole brain — 138,639 × 138,639 with 2.7M non-zero cells.
It cannot be drawn as is: the screen has too few pixels and the matrix is 0.014% full. Hence two views:

- left — the matrix itself: neurons are sorted by class and squeezed into bins, brightness is the
  number of synapses that fell into a cell (log scale). The block structure is visible, and so is the
  fact that the brain is not layered: connections go every way, including backwards;
- right — a "class -> class" summary: which classes each class gets its input synapses from.
  A row sums to 100% — exactly how the matrix is normalized in the model (`FlyBrain.W`).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import fly  # noqa: E402
from fly.utils.config import load  # noqa: E402
from fly.utils.paths import ROOT  # noqa: E402

_CFG = load("viz")
DEFAULTS = _CFG["matrix"]
C = _CFG["palette"]
BG, INK, MUTE, LINE = C["bg"], C["ink"], C["mute"], C["line"]
CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("fly", [BG, C["blue_deep"], C["blue"], C["blue_pale"]])


def binned_matrix(edges: pd.DataFrame, cls: np.ndarray, bins: int = DEFAULTS["bins"]) -> tuple[np.ndarray, np.ndarray]:
    """Squeezes the connectivity matrix into a picture: neurons are sorted by class and split into equal bins.

    Args:
        edges: Connections from `fly.load_connectome`.
        cls: Class number of every neuron from `fly.classify`.
        bins: Side of the picture in cells.

    Returns:
        A `[bins, bins]` matrix: row — who receives, column — who sends, value — how many synapses fell
        into the cell. The second element holds the class boundaries in bins, `len(CLASSES) + 1` numbers.

    Example:
        >>> edges = pd.DataFrame({"Presynaptic_Index": [0, 3], "Postsynaptic_Index": [1, 0], "Connectivity": [5, 7]})
        >>> m, bounds = binned_matrix(edges, np.array([1, 0, 0, 1]), bins=2)  # neuron order: 1, 2 | 0, 3
        >>> m.tolist(), bounds[:3].tolist()
        ([[0.0, 5.0], [0.0, 7.0]], [0.0, 1.0, 2.0])
    """
    n = len(cls)
    rank = np.empty(n, np.int64)
    rank[np.argsort(cls, kind="stable")] = np.arange(n)  # position of the neuron after sorting by class
    scale = bins / n
    post = (rank[edges.Postsynaptic_Index.to_numpy()] * scale).astype(np.int64)
    pre = (rank[edges.Presynaptic_Index.to_numpy()] * scale).astype(np.int64)
    m = np.zeros((bins, bins))
    np.add.at(m, (post, pre), edges.Connectivity.to_numpy())
    bounds = np.concatenate([[0], np.cumsum(np.bincount(cls, minlength=len(fly.CLASSES)))]) * scale
    return m, bounds


def class_matrix(edges: pd.DataFrame, cls: np.ndarray) -> np.ndarray:
    """Computes which classes every class gets its input synapses from.

    Args:
        edges: Connections from `fly.load_connectome`.
        cls: Class number of every neuron from `fly.classify`.

    Returns:
        A `[C, C]` matrix: row — receiving class, column — source class, value — share of the receiver's
        input synapses. A row sums to 1; a class with no inputs gets zeros.

    Example:
        >>> edges = pd.DataFrame({"Presynaptic_Index": [0, 1], "Postsynaptic_Index": [1, 1], "Connectivity": [6, 2]})
        >>> class_matrix(edges, np.array([0, 1]))[1, :2].tolist()  # class 1 gets 75% of its input from class 0
        [0.75, 0.25]
    """
    c = len(fly.CLASSES)
    m = np.zeros((c, c))
    np.add.at(m, (cls[edges.Postsynaptic_Index.to_numpy()], cls[edges.Presynaptic_Index.to_numpy()]),
              edges.Connectivity.to_numpy())
    total = m.sum(1, keepdims=True)
    return np.divide(m, total, out=np.zeros_like(m), where=total > 0)


def reciprocity(edges: pd.DataFrame) -> float:
    """Computes the share of connections that have a reverse one: A -> B and B -> A.

    Args:
        edges: Connections from `fly.load_connectome`.

    Returns:
        A share from 0 to 1. In a layered (feedforward) network it is zero — a direct measure of how
        recurrent the brain is.

    Example:
        >>> reciprocity(pd.DataFrame({"Presynaptic_Index": [0, 1, 1], "Postsynaptic_Index": [1, 0, 2]}))
        0.6666666666666666
    """
    pre, post = edges.Presynaptic_Index.to_numpy(np.int64), edges.Postsynaptic_Index.to_numpy(np.int64)
    n = int(max(pre.max(), post.max())) + 1
    return float(np.isin(pre * n + post, post * n + pre).mean())


def plot(edges: pd.DataFrame, cls: np.ndarray, path: Path, bins: int = DEFAULTS["bins"]) -> None:
    """Draws both views of the connectivity matrix in the schmidtml palette.

    Args:
        edges: Connections from `fly.load_connectome`.
        cls: Class number of every neuron from `fly.classify`.
        path: Where to save the PNG (1920×1080).
        bins: Side of the left picture in cells.
    """
    m, bounds = binned_matrix(edges, cls, bins)
    share = class_matrix(edges, cls)
    keys = [c.key for c in fly.CLASSES]
    fig, (left, right) = plt.subplots(1, 2, figsize=(19.2, 10.8), dpi=100, facecolor=BG,
                                      gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.22})

    left.imshow(np.log1p(m), cmap=CMAP, interpolation="nearest")
    for b in bounds[1:-1]:
        left.axhline(b - 0.5, color=LINE, lw=0.6)
        left.axvline(b - 0.5, color=LINE, lw=0.6)
    mid = (bounds[:-1] + bounds[1:]) / 2
    wide = np.diff(bounds) > bins * DEFAULTS["min_label_frac"]  # narrow classes get no label: labels would overlap
    left.set_xticks(mid[wide], [k for k, w in zip(keys, wide) if w], color=MUTE, fontsize=11)
    left.set_yticks(mid[wide], [k for k, w in zip(keys, wide) if w], color=MUTE, fontsize=11)
    left.set_xlabel("кто передаёт", color=MUTE, fontsize=12)
    left.set_ylabel("кто принимает", color=MUTE, fontsize=12)
    n_neurons, n_edges = (f"{x:,}".replace(",", " ") for x in (len(cls), len(edges)))
    left.set_title(f"Матрица связей: {n_neurons} × {n_neurons} нейронов, {n_edges} связей",
                   color=INK, loc="left", fontsize=16, weight="bold", pad=14)

    right.imshow(share, cmap=CMAP, vmin=0, vmax=share.max())
    for i, row in enumerate(share):
        for j, v in enumerate(row):
            if v >= DEFAULTS["min_share"]:
                right.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=11,
                           color=BG if v > share.max() * 0.7 else INK)
    right.set_xticks(range(len(keys)), keys, color=MUTE, fontsize=11, rotation=30, ha="right")
    right.set_yticks(range(len(keys)), keys, color=MUTE, fontsize=11)
    right.set_xlabel("от какого класса", color=MUTE, fontsize=12)
    right.set_title("Откуда класс получает входные синапсы", color=INK, loc="left", fontsize=16, weight="bold", pad=14)

    for ax in (left, right):
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def main() -> None:
    """Entry point of `fly-plot-matrix`: draws the connectivity matrix of the real connectome.

    Raises:
        FileNotFoundError: If the connectome is not downloaded.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ROOT / DEFAULTS["out"])
    p.add_argument("--bins", type=int, default=DEFAULTS["bins"], help="сторона картинки матрицы в ячейках")
    args = p.parse_args()
    edges, ann = fly.load_connectome()
    plot(edges, fly.classify(ann), args.out, args.bins)
    print(f"{args.out}: взаимных связей {reciprocity(edges):.1%}")


if __name__ == "__main__":
    main()
