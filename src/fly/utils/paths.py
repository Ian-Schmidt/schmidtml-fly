"""Where everything that is not code lives: the connectome, the exam levels, caches and run outputs."""

from pathlib import Path

from fly.utils.config import load

_PATHS = load("data")["paths"]  # paths relative to the repository root

PACKAGE = Path(__file__).resolve().parents[1]  # stored next to the model in MLflow so it loads without the repository
ROOT = PACKAGE.parents[1]
DATA = ROOT / _PATHS["data"]  # the FlyWire connectome, downloaded as described in the README
CACHE = ROOT / _PATHS["cache"]  # e5 embeddings
VIZ = ROOT / _PATHS["viz"]
DOCS = ROOT / _PATHS["docs"]
RUNS = Path(_PATHS["runs"])  # relative to the working directory, like the MLflow database
LEVELS = ROOT / _PATHS["levels"]  # exam levels: levels/<name>.json, never committed
