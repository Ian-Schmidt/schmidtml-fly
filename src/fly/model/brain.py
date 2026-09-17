"""The Drosophila connectome (FlyWire v783, 138,639 neurons) as a recurrent network.

The input is the fly's senses (SENSES): the question goes in through the first sense, the answer option
through the last one. `build(senses=("smell",))` delivers both texts as odours (53 glomeruli);
`("smell", "sight")` delivers the question as an odour and the answer "through the eyes"
(10,855 photoreceptors), which brings the optic lobes into play.

The wiring is real and never changes: who connects to whom, how many synapses, and the sign of each
synapse (acetylcholine excites, GABA/glutamate inhibit — Dale's law). Only the output "loudness" of
each neuron and its threshold are trained — two numbers per neuron — plus a linear readout from the
descending neurons, which run from the brain to the body.

The fly's task is a choice, as in a T-maze: the question and every answer option become odours, the
fly smells the question, then the question plus an option, and the descending neurons say how
strongly it is drawn to that option. It picks the most attractive one.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, NamedTuple, overload

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from fly.utils.config import load
from fly.utils.paths import DATA

_CFG = load("model")
MIN_SYN: int = _CFG["min_syn"]  # стандартный порог FlyWire: 2,7 млн связей, 63% всех синапсов
STEPS: int = _CFG["steps"]  # от рецептора до нисходящего нейрона 5–7 синапсов, сигналу нужно время дойти
ANSWER_AT: int = _CFG["answer_at"]  # до этого шага муха нюхает только вопрос, дальше вопрос + вариант
READ_LAST: int = _CFG["read_last"]  # выбор считывается по последним шагам
ALPHA: float = _CFG["alpha"]  # dt / tau
GAIN: float = _CFG["gain"]  # подобрано так, чтобы при старте были активны ~29% нисходящих нейронов
EMB_DIM: int = _CFG["emb_dim"]  # размерность эмбеддингов e5-small


class Sense(NamedTuple):
    """One of the fly's senses used as an input channel.

    Attributes:
        cell_class: Value of `cell_class` in the FlyWire annotations for this sense's receptors.
        by_type: Whether receptors of the same type share a channel. In smell, taste and
            mechanosensation neurons of one type carry the same receptor and respond identically, so
            channel = type. Every photoreceptor has its own channel: it is a pixel with its own column
            in the optic lobe.
    """

    cell_class: str
    by_type: bool


class NeuronClass(NamedTuple):
    """A class of neurons used for brain statistics and for colouring the visualization.

    Attributes:
        key: Short key (`orn`, `kc`, `dn`…) — the class is logged to MLflow metrics under this name.
        label: Caption in the visualization legend.
        color: Colour in the visualization, hex.
        belongs: Annotations → boolean mask "the neuron belongs to the class"; None means "everything else".
    """

    key: str
    label: str
    color: str
    belongs: Callable[[pd.DataFrame], pd.Series] | None


class Wiring(NamedTuple):
    """The brain's wiring as an edge list.

    Attributes:
        post: Index of the receiving neuron for every connection.
        pre: Index of the sending neuron for every connection.
        syn: Number of synapses in the connection.
        sign: Sign of every **neuron** (+1 excites, −1 inhibits, 0 — no outgoing connections).
    """

    post: np.ndarray
    pre: np.ndarray
    syn: np.ndarray
    sign: np.ndarray


SenseInput = tuple[Tensor, Tensor]
"""Input of one sense: (indices of the receptor neurons, channel number of each)."""

SENSES: dict[str, Sense] = {key: Sense(**spec) for key, spec in _CFG["senses"].items()}


def _membership(spec: dict[str, Any]) -> Callable[[pd.DataFrame], pd.Series] | None:
    """Builds the rule "the neuron belongs to the class" from a config entry.

    Args:
        spec: An entry of `classes` in `configs/model.json`: `column` plus one condition — `equals`,
            `isin` or `startswith`. Without `column` the class means "everything else".

    Returns:
        A function "annotations → boolean mask", or None for the "everything else" class.

    Raises:
        ValueError: If the entry has `column` but no condition.
    """
    if "column" not in spec:
        return None
    column = spec["column"]
    if "equals" in spec:
        return lambda a: a[column] == spec["equals"]
    if "isin" in spec:
        return lambda a: a[column].isin(spec["isin"])
    if "startswith" in spec:
        return lambda a: a[column].fillna("").str.startswith(spec["startswith"])
    raise ValueError(f"класс {spec['key']}: нужно одно из условий equals, isin, startswith")


# порядок задаёт приоритет, последний класс — все остальные
CLASSES: list[NeuronClass] = [
    NeuronClass(spec["key"], spec["label"], spec["color"], _membership(spec)) for spec in _CFG["classes"]
]


def load_connectome() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reads the FlyWire connectome from `data/` (see the README for how to download the files).

    Returns:
        A pair `(edges, ann)`. `edges` — connections with at least `MIN_SYN` synapses and the columns
        `Presynaptic_Index`, `Postsynaptic_Index`, `Connectivity`, `Excitatory`. `ann` — neuron
        annotations in the same order as the indices in `edges`; a neuron without an annotation has
        empty fields.

    Raises:
        FileNotFoundError: If the connectome files have not been downloaded.
    """
    ids = pd.read_csv(DATA / "Completeness_783.csv").iloc[:, 0].to_numpy()
    cols = ["Presynaptic_Index", "Postsynaptic_Index", "Connectivity", "Excitatory"]
    edges = pd.read_parquet(DATA / "Connectivity_783.parquet", columns=cols)
    edges = edges[edges.Connectivity >= MIN_SYN]
    ann = pd.read_csv(
        DATA / "annotations.tsv", sep="\t", low_memory=False,
        usecols=["root_id", "pos_x", "pos_y", "pos_z", "super_class", "cell_class", "cell_type"],
    ).set_index("root_id").reindex(ids)
    return edges, ann


def sense(ann: pd.DataFrame, key: str) -> SenseInput:
    """Finds the receptors of a sense and assigns them channels.

    Args:
        ann: Neuron annotations from `load_connectome`.
        key: Name of the sense — a key of `SENSES`: `smell`, `sight`, `touch` or `taste`.

    Returns:
        Indices of the sense's neurons and the channel number of each.

    Raises:
        KeyError: If `SENSES` has no such sense.

    Example:
        >>> ann = pd.DataFrame({"cell_class": ["olfactory"] * 3 + ["ALPN"], "cell_type": ["a", "b", "a", "x"]})
        >>> idx, code = sense(ann, "smell")
        >>> idx.tolist(), code.tolist()  # the two receptors of type "a" share one channel (glomerulus)
        ([0, 1, 2], [0, 1, 0])
    """
    cls, by_type = SENSES[key]
    mask = ann.cell_class.to_numpy() == cls
    code = pd.factorize(ann.cell_type[mask].fillna(f"{key}_unknown"))[0] if by_type else np.arange(mask.sum())
    return torch.tensor(np.flatnonzero(mask)), torch.tensor(code)


def descending(ann: pd.DataFrame) -> Tensor:
    """Finds the model's output — the descending neurons, the only channel from the brain to the body.

    Args:
        ann: Neuron annotations from `load_connectome`.

    Returns:
        Indices of the descending neurons.
    """
    return torch.tensor(np.flatnonzero(ann.super_class.to_numpy() == "descending"))


def classify(ann: pd.DataFrame) -> np.ndarray:
    """Assigns every neuron to one class from `CLASSES`.

    Args:
        ann: Neuron annotations from `load_connectome`.

    Returns:
        Class number for every neuron, `uint8`. If a neuron matches several classes, the one listed
        earlier in `CLASSES` wins; a neuron matching none falls into the last class.
    """
    cls = np.full(len(ann), len(CLASSES) - 1, np.uint8)
    for i, neuron_class in reversed(list(enumerate(CLASSES))):
        if neuron_class.belongs:
            cls[neuron_class.belongs(ann).to_numpy(bool)] = i
    return cls


def wiring(edges: pd.DataFrame, n: int) -> Wiring:
    """Turns the edge table into arrays for a sparse matrix.

    Args:
        edges: Connections from `load_connectome`.
        n: Number of neurons in the brain.

    Returns:
        Who signals to whom, how many synapses, and the sign of every neuron. The sign belongs to the
        neuron, not to the connection (Dale's law): a neuron has one transmitter, so all its outputs
        share a sign.
    """
    pre = edges.Presynaptic_Index.to_numpy()
    post = edges.Postsynaptic_Index.to_numpy()
    syn = edges.Connectivity.to_numpy().astype(np.float32)
    sign = np.zeros(n, np.float32)
    sign[pre] = edges.Excitatory.to_numpy()
    return Wiring(post, pre, syn, sign)


class FlyBrain(torch.nn.Module):
    """A rate-based recurrent network on the fly connectome.

    The wiring is fixed. What is trained: `log_gain` (output loudness of a neuron), `bias`
    (excitability threshold) and `readout` (a linear readout from the descending neurons).

    Attributes:
        W: Sparse connectivity matrix `[N, N]`, normalized by each neuron's total input.
        outputs: Indices of the descending neurons the choice is read from.
        senses: Input senses, in order.
        q_sense: Sense that delivers the question — the first one in `senses`.
        a_sense: Sense that delivers the answer option — the last one in `senses`.
        gain: Global gain; tuned so that the signal reaches the descending neurons.
        n: Number of neurons.

    Example:
        >>> edges = pd.DataFrame({"Presynaptic_Index": [0, 1, 2], "Postsynaptic_Index": [1, 2, 0],
        ...                       "Connectivity": [5, 9, 7], "Excitatory": [1, -1, 1]})
        >>> senses = {"smell": (torch.tensor([0]), torch.tensor([0]))}
        >>> brain = FlyBrain(*wiring(edges, 3), senses, outputs=torch.tensor([2]), emb_dim=4)
        >>> brain.choose(torch.randn(2, 4), torch.randn(2, 3, 4)).shape  # 2 questions, 3 options each
        torch.Size([2, 3])
    """

    W: Tensor
    outputs: Tensor

    def __init__(
        self,
        post: np.ndarray,
        pre: np.ndarray,
        syn: np.ndarray,
        sign: np.ndarray,
        senses: Mapping[str, SenseInput],
        outputs: Tensor,
        emb_dim: int = EMB_DIM,
        gain: float = GAIN,
        seed: int = 0,
    ) -> None:
        """Assembles the brain from its wiring.

        Args:
            post: Index of the receiving neuron for every connection.
            pre: Index of the sending neuron for every connection.
            syn: Number of synapses in the connection.
            sign: Sign of every neuron; the first four arguments are the `Wiring` returned by `wiring()`.
            senses: Input senses from `sense()`; order matters: the first gets the question, the last the option.
            outputs: Indices of the descending neurons from `descending()`.
            emb_dim: Dimensionality of the text embeddings.
            gain: Global gain of neuron outputs.
            seed: Seed of the random "nose" projection.
        """
        super().__init__()
        n = len(sign)
        # каждый нейрон получает взвешенное среднее входов: сигнал не взрывается, но и не глохнет
        in_total = np.bincount(post, weights=syn, minlength=n)
        w = sign[pre] * syn / in_total[post]
        self.register_buffer("W", torch.sparse_coo_tensor(
            torch.tensor(np.stack([post, pre])), torch.tensor(w, dtype=torch.float32), (n, n)).coalesce())
        self.register_buffer("outputs", outputs)
        # «Нос» каждого чувства: фиксированная случайная проекция текста на его каналы. Не обучается.
        g = torch.Generator().manual_seed(seed)
        # Чувства вливают одинаковый суммарный ток: иначе 10 855 фоторецепторов просто перекрикивают
        # 2 279 обонятельных, и вклад второго текста теряется.
        mean_n = sum(len(idx) for idx, _ in senses.values()) / len(senses)
        for key, (idx, code) in senses.items():
            self.register_buffer(f"in_{key}", idx)
            self.register_buffer(f"code_{key}", code)
            self.register_buffer(f"scale_{key}", torch.tensor(mean_n / len(idx), dtype=torch.float32))
            self.register_buffer(f"nose_{key}", torch.randn(int(code.max()) + 1, emb_dim, generator=g))
        self.senses = list(senses)
        self.q_sense, self.a_sense = self.senses[0], self.senses[-1]
        self.gain = gain
        self.log_gain = torch.nn.Parameter(torch.zeros(n))  # громкость выхода нейрона
        self.bias = torch.nn.Parameter(torch.zeros(n))  # порог возбудимости
        self.readout = torch.nn.Linear(len(outputs), 1)
        self.n = n

    def channels(self, key: str) -> int:
        """Counts the channels of a sense: 54 glomeruli for smell, 10,855 photoreceptors for sight.

        Args:
            key: Name of a sense from `self.senses`.

        Returns:
            Number of channels.
        """
        return int(self.get_buffer(f"code_{key}").max()) + 1

    def code(self, emb: Tensor, key: str) -> Tensor:
        """Turns text into what the fly actually receives as input.

        Args:
            emb: Text embeddings `[B, D]`.
            key: Name of the sense.

        Returns:
            Activity of the sense's channels `[B, channels]`, non-negative.
        """
        return torch.relu(emb @ self.get_buffer(f"nose_{key}").T)

    def smell(self, emb: Tensor, key: str | None = None) -> Tensor:
        """Feeds text into the receptors of a sense.

        Args:
            emb: Text embeddings `[B, D]`.
            key: Name of the sense; defaults to the question's sense.

        Returns:
            Input current `[N, B]`: non-zero only at this sense's receptors, and all receptors of one
            channel get the same current — like receptors of one glomerulus in a real fly.
        """
        key = key or self.q_sense
        idx, code = self.get_buffer(f"in_{key}"), self.get_buffer(f"code_{key}")
        drive = self.code(emb, key)[:, code].T * self.get_buffer(f"scale_{key}")
        return torch.zeros(self.n, emb.shape[0], device=emb.device).index_add(0, idx, drive)

    def out_gain(self) -> Tensor:
        """Computes the output multiplier of every neuron: trainable "loudness" × global gain.

        Returns:
            Multiplier `[N, 1]`.
        """
        return torch.exp(self.log_gain)[:, None] * self.gain

    @overload
    def forward(self, q: Tensor, a: Tensor, record: Literal[False] = False) -> Tensor: ...
    @overload
    def forward(self, q: Tensor, a: Tensor, record: Literal[True]) -> tuple[Tensor, Tensor]: ...
    def forward(self, q: Tensor, a: Tensor, record: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """One "sniff": the fly smells the question and, from step `ANSWER_AT`, the question plus the option.

        Args:
            q: Question embeddings `[B, D]`.
            a: Option embeddings `[B, D]`, one per question.
            record: Whether to return whole-brain activity per step (needed by the visualization and stats).

        Returns:
            Attractiveness of the option `[B]`; with `record=True` — a pair with the activity `[STEPS, N, B]`.
        """
        i_q = self.smell(q, self.q_sense)
        i_qa = i_q + self.smell(a, self.a_sense)
        out_gain = self.out_gain()
        v = torch.zeros_like(i_q)
        r = torch.zeros_like(i_q)
        read, trace = torch.zeros(len(self.outputs), i_q.shape[1], device=i_q.device), []
        for t in range(STEPS):
            i = i_q if t < ANSWER_AT else i_qa
            v = v + ALPHA * (-v + torch.sparse.mm(self.W, out_gain * r) + self.bias[:, None] + i)
            r = torch.tanh(torch.relu(v))  # частота разрядов в [0, 1)
            if t >= STEPS - READ_LAST:
                read = read + r[self.outputs]
            if record:
                trace.append(r)
        score = self.readout((read / READ_LAST).T).squeeze(-1)
        return (score, torch.stack(trace)) if record else score

    @overload
    def choose(self, q: Tensor, a: Tensor, record: Literal[False] = False) -> Tensor: ...
    @overload
    def choose(self, q: Tensor, a: Tensor, record: Literal[True]) -> tuple[Tensor, Tensor]: ...
    def choose(self, q: Tensor, a: Tensor, record: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """Scores all answer options: each one gets its own independent sniff.

        Args:
            q: Question embeddings `[B, D]`.
            a: Option embeddings `[B, K, D]`.
            record: Whether to return brain activity per step.

        Returns:
            Option scores `[B, K]` — the fly picks the `argmax`; with `record=True` — a pair with the
            activity `[STEPS, N, B * K]`.
        """
        b, k, d = a.shape
        q, a = q.repeat_interleave(k, 0), a.reshape(b * k, d)
        if record:
            score, trace = self(q, a, record=True)
            return score.view(b, k), trace
        return self(q, a).view(b, k)


def build(senses: Sequence[str] = ("smell",), seed: int = 0, device: str = "cpu") -> tuple[FlyBrain, pd.DataFrame]:
    """Builds the brain from the downloaded connectome.

    Args:
        senses: Input senses: the first gets the question, the last gets the answer option.
        seed: Seed of the random "nose" projection.
        device: torch device, see `device()`.

    Returns:
        The brain on the requested device and the neuron annotations (needed by stats and visualization).

    Raises:
        FileNotFoundError: If the connectome files have not been downloaded.
        KeyError: If a sense is missing from `SENSES`.

    Example:
        >>> brain, ann = build(senses=("smell", "sight"), device=device())  # doctest: +SKIP
        >>> brain.n, sum(p.numel() for p in brain.parameters())  # doctest: +SKIP
        (138639, 278578)
    """
    edges, ann = load_connectome()
    brain = FlyBrain(*wiring(edges, len(ann)), {k: sense(ann, k) for k in senses}, descending(ann), seed=seed)
    return brain.to(device), ann


def device() -> str:
    """Picks the fastest available device.

    Returns:
        `"mps"` on Apple Silicon, otherwise `"cuda"`, otherwise `"cpu"`.
    """
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
