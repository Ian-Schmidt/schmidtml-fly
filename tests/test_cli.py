from importlib.metadata import entry_points

import pytest

SCRIPTS = [ep for ep in entry_points(group="console_scripts") if ep.name.startswith("fly-")]


def test_all_scripts_are_installed():
    assert {ep.name for ep in SCRIPTS} == {"fly-train", "fly-continual", "fly-export-viz", "fly-plot-matrix"}


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda ep: ep.name)
def test_script_shows_help(script, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", [script.name, "--help"])
    with pytest.raises(SystemExit) as exit_info:
        script.load()()
    assert exit_info.value.code == 0 and "usage: " + script.name in capsys.readouterr().out
