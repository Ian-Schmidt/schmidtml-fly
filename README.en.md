# 🪰 Teaching a fruit fly to pass ML quizzes

**English** · [Русский](README.md)

**A real Drosophila connectome is trained to solve machine-learning quizzes**

![tests](https://github.com/Ian-Schmidt/schmidtml-fly/actions/workflows/tests.yml/badge.svg) ![lint](https://github.com/Ian-Schmidt/schmidtml-fly/actions/workflows/lint.yml/badge.svg) ![Python](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2FIan-Schmidt%2Fschmidtml-fly%2Fmain%2Fpyproject.toml&query=%24.project.requires-python&label=python&logo=python&logoColor=white&color=3776AB) ![Version](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2FIan-Schmidt%2Fschmidtml-fly%2Fmain%2Fpyproject.toml&query=%24.project.version&label=version&prefix=v&color=3987e5) ![Coverage](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2FIan-Schmidt%2Fschmidtml-fly%2Fmain%2Fpyproject.toml&query=%24.tool.coverage.report.fail_under&label=coverage&suffix=%25&color=2ea44f) ![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white) ![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white) ![Sentence Transformers](https://img.shields.io/badge/🤗%20sentence--transformers-e5--small-FFD21E) ![three.js](https://img.shields.io/badge/three.js-3D%20viz-000000?logo=threedotjs&logoColor=white)  
![FlyWire](https://img.shields.io/badge/data-FlyWire%20v783-8A2BE2)

![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json) ![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json) ![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)

The FlyWire v783 Drosophila connectome (138,639 neurons, 2.7M connections with >= 5 synapses) is turned
into a recurrent network, and that network is trained to solve schmidtml quizzes. The wiring is real and
never changes: only two numbers per neuron and a linear readout are trained — 278,578 parameters. A 3D
visualization shows how the "odour" of a question and an answer travels through the fly's brain.

![Connectivity matrix of the fly brain: the whole 138,639 x 138,639 matrix on the left; on the right, where each neuron class gets its input from](docs/connectome-matrix.png)

## Results

The same fly, the same 278,578 trainable numbers — the only thing that changes is the sense the text is fed through.


| task                                           | smell, 54 channels | sight, 10,855 channels | chance |
| ---------------------------------------------- | ------------------ | ---------------------- | ------ |
| Level 1 — site quizzes (746 questions)         | 45.6%              | **60.8%**              | 30.5%  |
| Level 2 — interview questions (537)            | 49.9%              | **83.2%**              | 25.0%  |
| Sorting questions by roadmap stage (758)       | 37.2%              | **53.1%**              | 14.3%  |


Accuracy on the exam — questions the fly has never seen; mean over the last 5 epochs, a single seed.

## Running

```bash
uv sync
mkdir -p data && cd data
curl -LO https://github.com/philshiu/Drosophila_brain_model/raw/main/Connectivity_783.parquet
curl -LO https://github.com/philshiu/Drosophila_brain_model/raw/main/Completeness_783.csv
curl -L -o annotations.tsv https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv
cd ..

uv run fly-train --task quiz                                  # level 1: site quizzes (~40 s/epoch on an M4 Pro)
uv run fly-train --task interview --init runs/quiz/brain.pt   # level 2, the fly keeps learning
uv run fly-train --task quiz --seed 1                         # repeat with another seed —> runs/quiz-s1
uv run fly-train --task quiz --senses sight                   # text "through the eyes": 10,855 channels instead of 54
uv run fly-train --task topics --epochs 15                    # the "sorting by topic" experiment
uv run fly-train --task level.json                            # your own question set (format below)
uv run fly-continual                                          # the "roadmap and forgetting" experiment
uv run fly-continual --replay 0.3                             # same, replaying 30% of past cards
uv run fly-plot-matrix                                        # connectivity matrix —> docs/connectome-matrix.png
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5050   # http://127.0.0.1:5050 (5000 is taken by AirPlay on macOS)
```

The question data is proprietary and not part of this repository: a level is a `levels/<name>.json` file with a
list of tasks in the format below, built separately. Text embeddings are computed once per data version and cached in
`cache/`. Every launch is a run in the MLflow experiment `schmidtml-fly` (parameters, metrics, stage
traces, artifacts), and the trained model is registered in the Model Registry as `fly-<level>` and
loaded with `mlflow.pytorch.load_model("models:/fly-quiz/1")`. A local copy lives in
`runs/<level>[-s<seed>]/`: `brain.pt` and `history.json`.

Format of a custom set: `[{"q": "...", "options": ["...", "...", "...", "..."], "answer": [0]}]`.

## Layout

```
src/fly/
├── model/brain.py          connectome —> FlyBrain: senses, dynamics, readout from descending neurons
├── data/tasks.py           loading levels from JSON, cached e5 embeddings, the train/exam split
├── training/
│   ├── train.py            supervised training (fly-train)
│   ├── continual.py        roadmap stages one by one and the forgetting metric (fly-continual)
│   ├── metrics.py          accuracy, loss, brain statistics per neuron class
│   └── baselines.py        no-fly baselines, including the input ceiling
├── viz/
│   ├── export.py           activity recording for the 3D scenes (fly-export-viz)
│   └── matrix.py           the connectivity matrix as a picture (fly-plot-matrix)
├── utils/                  config loading, project paths, MLflow tracking
└── configs/                every number and name that is not code:
    ├── model.json          brain dynamics (steps, gain), senses, neuron classes
    ├── data.json           encoder, paths, roadmap stages, exam share
    ├── training.json       MLflow, default command arguments, no-fly baselines
    └── viz.json            palette, captions, parameters of the connectivity-matrix picture
tests/                      pytest on a toy brain of 300 neurons
docs/                       images for the README
viz/                        three.js: the neuron cloud and the fly at its desk
```

## Development

```bash
uv run pytest               # tests and docstring examples, ~7 s; neither the connectome nor question data is needed
uv run ruff check .         # linter; also runs as a pre-commit hook
uv run ty check             # types
uv run pre-commit install   # once after cloning
```

Coverage is held at 100% by force (`fail_under = 100` in `pyproject.toml`): if coverage drops, the tests fail.
Only the `main()` functions are excluded: those are hours of training on the real connectome with MLflow.

## Visualization

```bash
uv run fly-export-viz --run runs/quiz --n 4   # ~18 MB per question, questions from the exam
cd viz && python3 -m http.server 8000
# http://localhost:8000/?run=quiz            — a cloud of 138,639 neurons, activity step by step
# http://localhost:8000/desk.html?run=quiz   — the fly at its desk solving a quiz, a brain hologram above it
```

Two pages on the same data:

- `index.html` — the fly brain up close: 138k neurons, the answer options and their "attraction";
- `desk.html` — a low-poly fly sits at a desk in a neon night room and solves a quiz on a monitor, with a
brain hologram above it and a NEURAL REPLAY panel below. Antennae, wings, head and the choice all move
according to the recorded activity.

Shared URL parameters: `run` — which run, `q` — which question to start from, `speed` — simulation steps
per second (6 by default), `clean` — no UI (for screen recording). `desk.html` only: `pixel` — pixel size
(2 by default, `1` — no pixelation), `shot` — starting camera shot (0–3), `noholo` — no hologram.

Keys: space — pause, <——> — question, H — UI; in `desk.html` also C — next camera shot, B — hologram.
`--split train` in the export takes the questions the fly was trained on.

## Data sources

- Zheng et al., *A complete electron microscopy volume of the brain of adult Drosophila melanogaster*, Cell 2018.
- Dorkenwald et al., *Neuronal wiring diagram of an adult brain*, Nature 2024.
- Schlegel et al., *Whole-brain annotation and multi-connectome cell typing of Drosophila*, Nature 2024.
- Eckstein et al., *Neurotransmitter classification from electron microscopy images at synaptic sites in Drosophila melanogaster*, Cell 2024.
- Shiu et al., *A Drosophila computational brain model reveals sensorimotor processing*, Nature 2024.
- Lappalainen et al., *Connectome-constrained networks predict neural activity across the fly visual system*, Nature 2024.
- Caron et al., *Random convergence of olfactory inputs in the Drosophila mushroom body*, Nature 2013.
- Aso et al., *The neuronal architecture of the mushroom body provides a logic for associative learning*, eLife 2014.
- Dasgupta, Stevens, Navlakha, *A neural algorithm for a fundamental computing problem*, Science 2017.
- Liu & Wilson, *Glutamate is an inhibitory neurotransmitter in the Drosophila olfactory system*, PNAS 2013.
- Tully & Quinn, *Classical conditioning and retention in normal and mutant Drosophila melanogaster*, J Comp Physiol A 1985.
- Wang et al., *Multilingual E5 Text Embeddings: A Technical Report*, 2024.

