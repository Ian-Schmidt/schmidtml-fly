"""Where everything that is not code lives: the connectome, the exam levels, caches and run outputs."""

from pathlib import Path

from fly.utils.config import load

_PATHS = load("data")["paths"]  # пути относительно корня репозитория

PACKAGE = Path(__file__).resolve().parents[1]  # кладётся рядом с моделью в MLflow, чтобы она грузилась без репозитория
ROOT = PACKAGE.parents[1]
DATA = ROOT / _PATHS["data"]  # коннектом FlyWire, скачивается по README
CACHE = ROOT / _PATHS["cache"]  # эмбеддинги e5
VIZ = ROOT / _PATHS["viz"]
DOCS = ROOT / _PATHS["docs"]
RUNS = Path(_PATHS["runs"])  # относительно рабочей папки, как и база MLflow
LEVELS = ROOT / _PATHS["levels"]  # уровни экзамена: levels/<имя>.json, в git не попадают
