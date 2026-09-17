import json

import pytest

from fly.model import brain
from fly.utils import config


@pytest.mark.parametrize("name", ["model", "data", "training", "viz"])
def test_configs_are_valid_json_and_cached(name):
    assert config.load(name) == json.loads((config.CONFIGS / f"{name}.json").read_text())
    assert config.load(name) is config.load(name), "повторное чтение не ходит на диск"


def test_unknown_config_is_an_error():
    with pytest.raises(FileNotFoundError):
        config.load("нет-такого")


def test_schedule_in_model_config_is_consistent():
    cfg = config.load("model")
    assert cfg["answer_at"] < cfg["steps"] - cfg["read_last"], "вариант должен прийти до окна считывания"
    assert [c["key"] for c in cfg["classes"]][-1] == "other" and "column" not in cfg["classes"][-1]


def test_class_rule_without_condition_is_an_error():
    with pytest.raises(ValueError, match="kc"):
        brain._membership({"key": "kc", "column": "cell_class"})
