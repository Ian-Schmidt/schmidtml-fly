"""Exam levels. Every level boils down to a choice among options.

A level is a JSON list of tasks: {"q": str, "options": [str], "answer": [int]} — see `Task`. Named levels
live in `levels/<name>.json` (not in git: the question data is proprietary); any other file works too:
`fly-train --task path/to/level.json`.
"""

import hashlib
import json
from pathlib import Path
from typing import NamedTuple, NotRequired, TypedDict

import torch
import torch.nn.functional as F
from torch import Tensor

from fly.utils.config import load
from fly.utils.paths import CACHE, LEVELS

_CFG = load("data")
ENCODER: str = _CFG["encoder"]
TEST_FRAC: float = _CFG["test_frac"]


class Task(TypedDict):
    """One exam task: a question and the options the fly chooses from.

    Attributes:
        q: Question text.
        options: Answer options; every task of a level has the same number of them.
        answer: Indices of the correct options — one or several.
        id: Task identifier in the source.
        topic: Topic of the task; in the staged experiments — the key of a roadmap stage from `STAGES`.
        group: The question a card belongs to; the exam in `fly-continual` is split by it.
    """

    q: str
    options: list[str]
    answer: list[int]
    id: NotRequired[str]
    topic: NotRequired[str]
    group: NotRequired[str]


class Stage(NamedTuple):
    """A roadmap stage.

    Attributes:
        key: Short key of the stage.
        title: Name — also the answer option in the topic-sorting level.
        label: Short caption for charts.
        categories: Identifiers of the flashcard categories assigned to the stage.
    """

    key: str
    title: str
    label: str


class Split(NamedTuple):
    """A split of a level.

    Attributes:
        train: Indices of the tasks the fly learns from.
        test: Indices of the exam — tasks the fly has never seen.
    """

    train: Tensor
    test: Tensor


# порядок этапов — порядок, в котором муха проходит роадмап в `fly-continual`; ключ этапа — поле `topic` задачи
STAGES = [Stage(s["key"], s["title"], s["label"]) for s in _CFG["stages"]]


def load(task: str) -> list[Task]:
    """Loads a level by name or from a file.

    Args:
        task: Level name — then `levels/<name>.json` is read — or a path to a JSON list of tasks.

    Returns:
        The tasks of the level.

    Raises:
        ValueError: If tasks have different numbers of options — they cannot be batched.
        FileNotFoundError: If there is neither a level with that name nor a file at that path.
    """
    named = LEVELS / f"{task}.json"
    items = json.loads((named if named.exists() else Path(task)).read_text())
    if len({len(it["options"]) for it in items}) != 1:
        raise ValueError(f"{task}: у всех задач должно быть одинаковое число вариантов")
    return items


def fingerprint(items: list[Task]) -> str:
    """Computes the fingerprint of a level.

    Args:
        items: The tasks of the level.

    Returns:
        10 hex characters; they change if a single question, option or correct answer changes.
        MLflow runs with different fingerprints must not be compared directly.

    Example:
        >>> fingerprint([{"q": "2 + 2?", "options": ["3", "4"], "answer": [1]}])
        'c30e675a05'
    """
    return hashlib.md5(json.dumps([[it["q"], it["options"], it["answer"]] for it in items]).encode()).hexdigest()[:10]


def embed(items: list[Task], task: str) -> tuple[Tensor, Tensor]:
    """Text → "odour": embeddings of the questions and options.

    The texts are read by a frozen e5 (this is the fly's nose, not its brain). The encoder runs once
    per data version: the result is cached in `cache/` under the level's fingerprint.

    Args:
        items: The tasks of the level.
        task: Level name or path to its file — goes into the cache file name.

    Returns:
        Centred and normalized embeddings: questions `[n, D]` and options `[n, K, D]`.
    """
    cache = CACHE / f"{Path(task).stem}-{fingerprint(items)}.pt"
    if cache.exists():
        q, a = torch.load(cache)
    else:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(ENCODER)
        enc = lambda xs: torch.tensor(model.encode(xs, batch_size=_CFG["encode_batch"], normalize_embeddings=True))  # noqa: E731
        q = enc(["query: " + it["q"] for it in items])
        a = enc(["passage: " + o for it in items for o in it["options"]]).view(len(items), -1, q.shape[1])
        cache.parent.mkdir(exist_ok=True)
        torch.save((q, a), cache)
    # у e5 все тексты смотрят почти в одну сторону (косинус ~0.75) — без центрирования все запахи одинаковые
    center = torch.cat([q, a.flatten(0, 1)]).mean(0)
    return F.normalize(q - center, dim=-1), F.normalize(a - center, dim=-1)


def split(n: int, test_frac: float = TEST_FRAC, seed: int = 0) -> Split:
    """Splits a level into training and exam.

    Args:
        n: Number of tasks.
        test_frac: Share of the exam.
        seed: Seed of the permutation.

    Returns:
        Disjoint training and exam indices.

    Example:
        >>> train, test = split(10)
        >>> len(train), len(test)
        (8, 2)
    """
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    k = int(n * test_frac)
    return Split(perm[k:], perm[:k])


def targets(items: list[Task], k: int) -> Tensor:
    """Builds soft targets for cross-entropy.

    Args:
        items: The tasks of the level.
        k: Number of options per task.

    Returns:
        Targets `[n, k]`: with several correct answers the probability is shared equally among them.

    Example:
        >>> targets([{"q": "", "options": ["a", "b", "c", "d"], "answer": [0, 2]}], 4).tolist()
        [[0.5, 0.0, 0.5, 0.0]]
    """
    t = torch.zeros(len(items), k)
    for i, it in enumerate(items):
        t[i, it["answer"]] = 1
    return t / t.sum(1, keepdim=True)
