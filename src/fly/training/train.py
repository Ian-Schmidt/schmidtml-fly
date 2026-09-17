"""Training the fly.

    uv run fly-train --task quiz                      # level 1: quizzes
    uv run fly-train --task interview --init runs/quiz/brain.pt   # level 2
    uv run fly-train --task quiz --seed 1             # repeat with another seed -> runs/quiz-s1
    uv run fly-train --task quiz --senses smell,sight # question as an odour, answer option "through the eyes"
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5050   # experiments -> http://127.0.0.1:5050

Every launch is a run in the MLflow experiment "schmidtml-fly": parameters (arguments, model constants,
data fingerprint), per-batch and per-epoch metrics, traces of the stages (setup, every epoch, saving),
the artifacts history.json and brain.pt, and the model itself in the registry as fly-<level>.
The same is kept locally in runs/<level>[-s<seed>]/. Details — DOCS.md, section 6.
"""

import argparse
import json
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torch.nn.functional as F
from mlflow.entities import SpanType

import fly
from fly.data import tasks
from fly.data.tasks import split, targets
from fly.training.baselines import references
from fly.training.metrics import accuracy, brain_stats, evaluate
from fly.utils.config import load
from fly.utils.paths import PACKAGE, RUNS
from fly.utils.tracking import EXPERIMENT, TRACKING, stage

_CFG = load("training")
DEFAULTS = _CFG["train"]  # default values of the fly-train arguments
PROBE: int = _CFG["probe"]  # how many exam questions we probe brain activity on after every epoch


def main() -> None:
    """Entry point of `fly-train`: trains the fly on a level and writes everything to MLflow and `runs/`.

    Raises:
        FileNotFoundError: If the connectome is not downloaded or the level file is missing.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--task", default=DEFAULTS["task"], help="quiz | interview | путь к JSON третьего уровня")
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--batch", type=int, default=DEFAULTS["batch"], help="вопросов в батче (× число вариантов прогонов)")
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--senses", default=DEFAULTS["senses"],
                   help="чувства входа через запятую: вопрос первым, вариант ответа последним (smell,sight)")
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"], help="меняет разбиение, «нос» и порядок батчей")
    p.add_argument("--init", help="продолжить с чекпойнта прошлого уровня")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    dev = fly.device()
    tag = Path(args.task).stem + ("" if args.senses == "smell" else "-" + args.senses.replace(",", "+"))
    registry = f"fly-{tag}"
    name = tag + (f"-s{args.seed}" if args.seed else "")
    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name=name) as run:
        rid = run.info.run_id
        mlflow.set_tag("task", args.task)

        with stage("setup", SpanType.WORKFLOW, root=rid, device=dev, **vars(args)):
            with stage("load_task", task=args.task) as span:
                items = tasks.load(args.task)
                span.set_outputs({"items": len(items), "fingerprint": tasks.fingerprint(items)})
            with stage("embed", SpanType.EMBEDDING, encoder=tasks.ENCODER,
                       texts=len(items) * (1 + len(items[0]["options"]))) as span:
                q, a = tasks.embed(items, args.task)
                span.set_outputs({"questions": list(q.shape), "options": list(a.shape)})
            t = targets(items, a.shape[1])
            tr, te = split(len(items), seed=args.seed)
            with stage("build_connectome", min_syn=fly.MIN_SYN, senses=args.senses, init=args.init) as span:
                brain, ann = fly.build(senses=tuple(args.senses.split(",")), seed=args.seed, device=dev)
                if args.init:
                    brain.load_state_dict(torch.load(args.init)["params"], strict=False)
                trainable = sum(p.numel() for p in brain.parameters())
                span.set_outputs({"neurons": brain.n, "edges": brain.W._nnz(), "trainable": trainable,
                                  "channels": {k: brain.channels(k) for k in brain.senses}})
            with stage("references", SpanType.EVALUATOR, n_train=len(tr), n_test=len(te)) as span:
                ref = references(brain, q, a, t, tr, te)
                span.set_outputs(ref)
        print(f"{len(items)} задач: {len(tr)} учим, {len(te)} экзамен | без мухи: " +
              ", ".join(f"{k}={v:.3f}" for k, v in ref.items()), flush=True)

        mlflow.log_params({
            **vars(args), "data": tasks.fingerprint(items), "encoder": tasks.ENCODER, "n_train": len(tr),
            "channels": sum(brain.channels(k) for k in brain.senses),
            "n_test": len(te), "neurons": brain.n, "edges": brain.W._nnz(), "min_syn": fly.MIN_SYN,
            "trainable": trainable, "gain": brain.gain, "alpha": fly.ALPHA,
            "steps": fly.STEPS, "answer_at": fly.ANSWER_AT, "read_last": fly.READ_LAST,
        })
        # ty still infers TypedDict.items() values as object although every field is a float
        mlflow.log_metrics({f"reference/{k}": v for k, v in ref.items()}, step=0)  # ty: ignore[invalid-argument-type]
        cls = torch.tensor(fly.classify(ann), device=dev)
        q, a, t = q.to(dev), a.to(dev), t.to(dev)
        probe = te[:PROBE]
        opt = torch.optim.Adam(brain.parameters(), args.lr)
        history, step = [], 0

        for epoch in range(1, args.epochs + 1):
            with stage(f"epoch-{epoch}", SpanType.WORKFLOW, root=rid, epoch=epoch) as epoch_span:
                t0, losses, grads = time.time(), [], []
                with stage("train", batch=args.batch, lr=args.lr) as span:
                    for b in tr[torch.randperm(len(tr))].split(args.batch):
                        scores = brain.choose(q[b], a[b])
                        loss = F.cross_entropy(scores, t[b])
                        opt.zero_grad()
                        loss.backward()
                        # the gradient norm is only measured: the threshold is infinite, there is no clipping
                        grads.append(torch.nn.utils.clip_grad_norm_(brain.parameters(), float("inf")).item())
                        opt.step()
                        step += 1
                        losses.append(loss.item())
                        mlflow.log_metrics({"batch/loss": losses[-1], "batch/acc": accuracy(scores.detach(), t[b]),
                                            "batch/grad_norm": grads[-1]}, step=step)
                    span.set_outputs({"batches": len(losses), "mean_loss": sum(losses) / len(losses),
                                      "max_grad_norm": max(grads)})
                with stage("evaluate", SpanType.EVALUATOR, n_train=len(tr), n_test=len(te)) as span:
                    m = {"train": evaluate(brain, q, a, t, tr, args.batch * 2),
                         "test": evaluate(brain, q, a, t, te, args.batch * 2)}
                    span.set_outputs(m)
                with stage("brain_stats", SpanType.EVALUATOR, questions=len(probe)) as span:
                    stats = brain_stats(brain, q[probe], a[probe], cls)
                    span.set_outputs(stats)

                row = {"epoch": epoch, "batch_loss": sum(losses) / len(losses),
                       **{f"{s}_{k}": v for s, mm in m.items() for k, v in mm.items()},
                       "sec": round(time.time() - t0), "brain": stats}
                history.append(row)
                mlflow.log_metrics({  # ty: ignore[invalid-argument-type]
                    "epoch/batch_loss": row["batch_loss"], "epoch/sec": row["sec"],
                    **{f"{k}/{s}": v for s, mm in m.items() for k, v in mm.items()},
                    **{f"brain_{k}/{key}": v for key, st in stats.items() for k, v in st.items()},
                }, step=epoch)
                print(f"epoch={epoch} batch_loss={row['batch_loss']:.3f} test_loss={row['test_loss']:.3f} "
                      f"train_acc={row['train_acc']:.3f} test_acc={row['test_acc']:.3f} "
                      f"test_p={row['test_p_correct']:.3f} kc_active={stats['kc']['active']:.2f} "
                      f"dn_rate={stats['dn']['rate']:.3f} sec={row['sec']}", flush=True)

                with stage("checkpoint", path=str(out)) as span:
                    params = {k: v.cpu() for k, v in brain.named_parameters()}
                    torch.save({"params": params, "task": args.task, "senses": brain.senses, "seed": args.seed,
                                "train": tr.tolist(), "test": te.tolist()}, out / "brain.pt")
                    (out / "history.json").write_text(
                        json.dumps({"args": vars(args), "references": ref, "history": history}, indent=1))
                    span.set_outputs({"brain.pt bytes": (out / "brain.pt").stat().st_size})
                epoch_span.set_outputs({k: v for k, v in row.items() if k != "brain"})

        with stage("save_model", SpanType.WORKFLOW, root=rid, registered_model=registry) as root:
            with stage("log_artifacts", files=["history.json", "brain.pt"]):
                mlflow.log_artifact(str(out / "history.json"))
                mlflow.log_artifact(str(out / "brain.pt"))
            with stage("log_model", format="pickle", code_paths=["fly/"]) as span:
                # pickle rather than pt2: torch.export cannot trace 32 steps with a sparse matrix.
                # the fly package is stored next to the model, so it loads without this repository too.
                info = mlflow.pytorch.log_model(
                    brain.cpu(), name="model", serialization_format="pickle", code_paths=[str(PACKAGE)],
                    registered_model_name=registry, step=args.epochs)
                span.set_outputs({"model_uri": info.model_uri, "version": info.registered_model_version})
            registered = f"{registry} v{info.registered_model_version}"
            root.set_outputs({"model_uri": info.model_uri, "registered": registered})
        print(f"модель: {info.model_uri} -> реестр {registry}, версия {info.registered_model_version}")


if __name__ == "__main__":
    main()
