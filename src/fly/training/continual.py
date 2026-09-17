"""The "fly walks the roadmap and forgets" experiment.

The flashcards of the `cards` level are split into 7 stages in roadmap order (tasks.STAGES). The fly learns the
stages one by one and after each takes an exam on all of them — this shows how earlier material is
forgotten (catastrophic forgetting).

    uv run fly-continual                 # no replay
    uv run fly-continual --replay 0.3    # 30% of cards from past stages are mixed into each stage

Every launch is an MLflow run (tag experiment=continual): acc/<stage> per stage step (the forgetting
curve), final forgetting, the "after stage × exam on stage" matrix and its heatmap as artifacts, and
the model in the registry as fly-roadmap-<strategy>. A local copy lives in runs/roadmap-<strategy>/.
"""

import argparse
import json
import random
import statistics
import time
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.pytorch  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from mlflow.entities import SpanType  # noqa: E402

import fly  # noqa: E402
from fly.data import tasks  # noqa: E402
from fly.data.tasks import TEST_FRAC, Split, Task, targets  # noqa: E402
from fly.training.metrics import accuracy, evaluate  # noqa: E402
from fly.utils.config import load  # noqa: E402
from fly.utils.paths import PACKAGE, RUNS  # noqa: E402
from fly.utils.tracking import EXPERIMENT, TRACKING, stage  # noqa: E402

DEFAULTS = load("training")["continual"]  # значения аргументов fly-continual по умолчанию
C = load("viz")["palette"]
LABELS = {stage.key: stage.label for stage in tasks.STAGES}


def split_by_group(items: list[Task], idx: Sequence[int], test_frac: float = TEST_FRAC, seed: int = 0) -> Split:
    """Splits a stage's cards into training and exam by whole questions.

    Cards of one question never land in both training and exam: otherwise the fly would have seen the exam.

    Args:
        items: All tasks of the `cards` level; each has a `group` field.
        idx: Indices of the stage's tasks.
        test_frac: Share of **questions** (not cards) that go to the exam; at least one question.
        seed: Seed for shuffling the questions.

    Returns:
        Training and exam indices.
    """
    groups = sorted({items[i]["group"] for i in idx})
    random.Random(seed).shuffle(groups)
    test = set(groups[: max(1, int(len(groups) * test_frac))])
    return Split(torch.tensor([i for i in idx if items[i]["group"] not in test]),
                 torch.tensor([i for i in idx if items[i]["group"] in test]))


def forgetting(R: Sequence[Sequence[float]]) -> list[float]:
    """Computes forgetting — the standard continual-learning metric.

    Args:
        R: Square matrix: `R[s][j]` is the exam accuracy on stage `j` after training on stage `s`.

    Returns:
        Forgetting of every stage except the last (it has no time to be forgotten): the best accuracy
        on the stage after learning it minus the final one. Negative means the stage even improved.

    Example:
        >>> forgetting([[0.9, 0.2], [0.5, 0.8]])  # stage one: 90% right after learning it, 50% at the end
        [0.4]
    """
    last = len(R) - 1
    return [max(R[s][j] for s in range(j, last)) - R[last][j] for j in range(last)]


def heatmap(R: Sequence[Sequence[float]], names: Sequence[str], title: str, path: Path) -> None:
    """Draws the forgetting matrix in the schmidtml palette.

    Args:
        R: Accuracy matrix: rows — after which stage, columns — exam on which stage.
        names: Stage captions.
        title: Figure title.
        path: Where to save the PNG.
    """
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("fly", [C["card"], C["blue_deep"], C["blue"]])
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100, facecolor=C["bg"])
    ax.imshow(R, cmap=cmap, vmin=0.25, vmax=max(max(row) for row in R))
    for s, row in enumerate(R):
        for j, v in enumerate(row):
            ax.text(j, s, f"{v:.0%}", ha="center", va="center", fontsize=13, color=C["ink"] if j <= s else C["mute"])
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right", color=C["ink2"], fontsize=12)
    ax.set_yticks(range(len(names)), [f"после: {n}" for n in names], color=C["ink2"], fontsize=12)
    ax.set_xlabel("экзамен по этапу", color=C["mute"], fontsize=12)
    ax.set_title(title, color=C["ink"], loc="left", fontsize=17, weight="bold", pad=16)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor=C["bg"])
    plt.close(fig)


def main() -> None:
    """Entry point of `fly-continual`: the fly walks the roadmap stages one by one, with an exam on all after each.

    Raises:
        FileNotFoundError: If the connectome is not downloaded or `levels/cards.json` is missing.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--replay", type=float, default=DEFAULTS["replay"],
                   help="доля карточек прошлых этапов, подмешиваемых в обучение")
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"], help="эпох на этап")
    p.add_argument("--batch", type=int, default=DEFAULTS["batch"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--senses", default=DEFAULTS["senses"],
                   help="чувства входа через запятую: вопрос карточки первым, ответ последним (smell,sight)")
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    args = p.parse_args()

    torch.manual_seed(args.seed)
    dev = fly.device()
    strategy = f"replay{round(args.replay * 100)}" if args.replay else "naive"
    senses_tag = "" if args.senses == "smell" else "-" + args.senses.replace(",", "+")
    name = f"roadmap-{strategy}{senses_tag}" + (f"-s{args.seed}" if args.seed else "")
    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    keys = [s[0] for s in tasks.STAGES]
    mlflow.set_tracking_uri(TRACKING)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name=name) as run:
        rid = run.info.run_id
        mlflow.set_tags({"task": "cards", "experiment": "continual", "strategy": strategy})
        with stage("setup", SpanType.WORKFLOW, root=rid, **vars(args)) as span:
            items = tasks.load("cards")
            q, a = tasks.embed(items, "cards")
            t = targets(items, a.shape[1])
            splits = [split_by_group(items, [i for i, it in enumerate(items) if it["topic"] == k], seed=args.seed)
                      for k in keys]
            brain, _ = fly.build(senses=tuple(args.senses.split(",")), seed=args.seed, device=dev)
            span.set_outputs({"cards": len(items), "train": {k: len(tr) for k, (tr, _) in zip(keys, splits)}})
        cosine = [accuracy((q[te, None] * a[te]).sum(-1), t[te]) for _, te in splits]
        mlflow.log_params({
            **vars(args), "strategy": strategy, "data": tasks.fingerprint(items), "encoder": tasks.ENCODER,
            "stages": " → ".join(keys), "trainable": sum(p.numel() for p in brain.parameters()),
            "channels": sum(brain.channels(k) for k in brain.senses),
            **{f"n_train/{k}": len(tr) for k, (tr, _) in zip(keys, splits)},
            **{f"n_test/{k}": len(te) for k, (_, te) in zip(keys, splits)},
        })
        mlflow.log_metrics({"reference/chance": (t > 0).float().mean().item(),
                            **{f"reference/cosine/{k}": v for k, v in zip(keys, cosine)}}, step=0)
        print(f"{len(items)} карточек, {len(keys)} этапов | косинус e5 по этапам: " +
              " ".join(f"{k}={v:.2f}" for k, v in zip(keys, cosine)), flush=True)

        q, a, t = q.to(dev), a.to(dev), t.to(dev)
        opt = torch.optim.Adam(brain.parameters(), args.lr)
        R, step = [], 0
        for s, key in enumerate(keys):
            with stage(f"stage-{s + 1}-{key}", SpanType.WORKFLOW, root=rid, stage=key, epochs=args.epochs) as span:
                t0, losses = time.time(), []
                old = torch.cat([splits[j][0] for j in range(s)]) if s else splits[0][0][:0]  # прошлых этапов ещё нет
                for _ in range(args.epochs):
                    pool = splits[s][0]
                    if args.replay and s:  # повторение: каждую эпоху новая случайная выборка из прошлых этапов
                        pool = torch.cat([pool, old[torch.randperm(len(old))[: int(args.replay * len(pool))]]])
                    for b in pool[torch.randperm(len(pool))].split(args.batch):
                        loss = F.cross_entropy(brain.choose(q[b], a[b]), t[b])
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        step += 1
                        losses.append(loss.item())
                        mlflow.log_metric("batch/loss", losses[-1], step=step)
                accs = [evaluate(brain, q, a, t, te, args.batch * 2)["acc"] for _, te in splits]
                R.append(accs)
                seen = statistics.mean(accs[: s + 1])
                mlflow.log_metrics({**{f"acc/{k}": v for k, v in zip(keys, accs)}, "acc/seen_mean": seen,
                                    "stage/loss": statistics.mean(losses), "stage/sec": time.time() - t0}, step=s + 1)
                span.set_outputs({"acc": dict(zip(keys, accs)), "seen_mean": seen})
                print(f"stage {s + 1} {key:<9} " + " ".join(f"{k}={v:.2f}" for k, v in zip(keys, accs)) +
                      f" | seen={seen:.3f} sec={time.time() - t0:.0f}", flush=True)

        forget = forgetting(R)
        summary = {"final/mean_acc": statistics.mean(R[-1]), "forgetting/mean": statistics.mean(forget),
                   "learned/mean": statistics.mean(R[s][s] for s in range(len(keys))),
                   **{f"forgetting/{k}": v for k, v in zip(keys, forget)}}
        mlflow.log_metrics(summary)
        (out / "matrix.json").write_text(json.dumps(
            {"stages": keys, "R": R, "forgetting": dict(zip(keys, forget)), "cosine": dict(zip(keys, cosine)),
             "summary": summary, "args": vars(args)}, indent=1, ensure_ascii=False))
        title = f"Что муха помнит после каждого этапа · {strategy}"
        heatmap(R, [LABELS[k] for k in keys], title, out / "forgetting.png")
        with stage("save_model", SpanType.WORKFLOW, root=rid, registered_model=f"fly-{name}") as span:
            mlflow.log_artifact(str(out / "matrix.json"))
            mlflow.log_artifact(str(out / "forgetting.png"))
            info = mlflow.pytorch.log_model(brain.cpu(), name="model", serialization_format="pickle",
                                            code_paths=[str(PACKAGE)], registered_model_name=f"fly-{name}",
                                            step=len(keys))
            span.set_outputs({"model_uri": info.model_uri, "version": info.registered_model_version})
        print(" ".join(f"{k}={v:.3f}" for k, v in summary.items() if "/" in k and not k.startswith("forgetting/")) +
              f" | модель fly-{name} v{info.registered_model_version}")


if __name__ == "__main__":
    main()
