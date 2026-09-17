"""Records brain activity for the 3D visualization (viz/index.html).

    uv run fly-export-viz --run runs/quiz --n 4
    cd viz && python3 -m http.server 8000      # → http://localhost:8000/?run=quiz
"""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import fly
from fly.data import tasks
from fly.data.tasks import targets
from fly.utils.config import load
from fly.utils.paths import VIZ

_CFG = load("viz")
DEFAULTS = _CFG["export"]  # значения аргументов fly-export-viz по умолчанию
TITLES: dict[str, str] = _CFG["titles"]


def export_brain(ann: pd.DataFrame) -> None:
    """Writes the neuron cloud for three.js: `viz/data/brain.bin` and `brain.json`.

    Args:
        ann: Neuron annotations from `fly.load_connectome`. Neurons without coordinates (~600) get
            class 255 and are not drawn.
    """
    cls = fly.classify(ann)
    voxel_nm = np.array(_CFG["voxel_nm"], np.float32)
    xyz = ann[["pos_x", "pos_y", "pos_z"]].to_numpy(np.float32) * voxel_nm / 1000  # воксели → мкм
    missing = np.isnan(xyz).any(1)  # ~600 нейронов без аннотации — не рисуем
    xyz -= np.nanmean(xyz, 0)
    xyz[:, 1] *= -1  # у FlyWire y растёт вниз
    xyz[missing], cls[missing] = 0, 255
    VIZ.mkdir(parents=True, exist_ok=True)
    (VIZ / "brain.bin").write_bytes(xyz.tobytes() + cls.tobytes())
    meta = {"n": len(ann), "classes": [{"key": c.key, "name": c.label, "color": c.color} for c in fly.CLASSES]}
    (VIZ / "brain.json").write_text(json.dumps(meta, ensure_ascii=False))


def main() -> None:
    """Entry point of `fly-export-viz`: records brain activity on a few questions for `viz/`.

    Raises:
        FileNotFoundError: If the `--run` checkpoint is missing or the connectome is not downloaded.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=DEFAULTS["run"])
    p.add_argument("--n", type=int, default=DEFAULTS["n"], help="сколько вопросов записать (~18 МБ на вопрос)")
    p.add_argument("--split", default=DEFAULTS["split"], choices=["test", "train"],
                   help="test — вопросы, которых муха не видела")
    args = p.parse_args()

    run = Path(args.run)
    ckpt = torch.load(run / "brain.pt")
    dev = fly.device()
    brain, ann = fly.build(senses=tuple(ckpt.get("senses", ("smell",))), seed=ckpt["seed"], device=dev)
    brain.load_state_dict(ckpt["params"], strict=False)
    export_brain(ann)

    items = tasks.load(ckpt["task"])
    q, a = tasks.embed(items, ckpt["task"])
    t = targets(items, a.shape[1])
    frames, questions = [], []
    with torch.no_grad():
        for i in ckpt[args.split][: args.n]:
            score, trace = brain.choose(q[i : i + 1].to(dev), a[i : i + 1].to(dev), record=True)  # trace [T, N, K]
            live = brain.readout(trace[:, brain.outputs].permute(0, 2, 1)).squeeze(-1).T  # [K, T]
            frames.append((trace.permute(2, 0, 1) * 255).round().to(torch.uint8).cpu().numpy())  # [K, T, N]
            s = score[0].cpu()
            it = items[i]
            questions.append({
                "id": it["id"], "topic": it.get("topic", ""), "q": it["q"], "options": it["options"],
                "answer": it["answer"], "chosen": int(s.argmax()), "correct": bool(t[i, s.argmax()] > 0),
                "probs": [round(x, 4) for x in s.softmax(0).tolist()],
                "live": [[round(x, 4) for x in row] for row in live.cpu().tolist()],
            })

    hist = json.loads((run / "history.json").read_text()) if (run / "history.json").exists() else {}
    # точность на экзамене скачет между эпохами на несколько пунктов — показываем среднее за последние 5, как в DOCS
    tail = hist.get("history", [])[-5:]
    exam = {"mean": statistics.mean(r["test_acc"] for r in tail), "sd": statistics.stdev(r["test_acc"] for r in tail),
            "epochs": [tail[0]["epoch"], tail[-1]["epoch"]]} if len(tail) > 1 else None
    meta = {"title": TITLES.get(ckpt["task"], Path(ckpt["task"]).stem), "split": args.split,
            "steps": fly.STEPS, "answerAt": fly.ANSWER_AT, "k": a.shape[1], "questions": questions,
            "exam": exam, "references": hist.get("references", {})}
    (VIZ / f"{run.name}.bin").write_bytes(np.stack(frames).tobytes())
    (VIZ / f"{run.name}.json").write_text(json.dumps(meta, ensure_ascii=False))
    right = sum(x["correct"] for x in questions)
    print(f"{run.name}: {len(questions)} вопросов, муха ответила верно на {right} → viz/data/{run.name}.*")


if __name__ == "__main__":
    main()
