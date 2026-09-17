import mlflow

from fly.utils import tracking


def test_stage_records_span(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")  # не в рабочую базу экспериментов
    with tracking.stage("setup", task="quiz") as span:
        span.set_outputs({"items": 3})
    assert span.name == "setup" and span.inputs == {"task": "quiz"} and span.outputs == {"items": 3}
