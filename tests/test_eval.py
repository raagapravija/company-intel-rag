import importlib.util
from pathlib import Path

EVAL = Path(__file__).parents[1] / "eval"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, EVAL / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_only_verified_questions_with_answers_count():
    run = _load("run_ragas")
    qs = [
        {"id": "a", "verified": True, "reference_answer": "x"},
        {"id": "b", "verified": False, "reference_answer": "x"},
        {"id": "c", "verified": True, "reference_answer": None},
    ]
    assert [q["id"] for q in run.ready_questions(qs)] == ["a"]
    # On the real file, only human-verified questions with references are scored.
    shipped = run.load_questions()
    ready_ids = {q["id"] for q in run.ready_questions(shipped)}
    assert ready_ids == {
        q["id"] for q in shipped if q.get("verified") and q.get("reference_answer")
    }
    assert not any(q["id"] in ready_ids for q in shipped if not q.get("verified"))


def test_router_accuracy():
    run = _load("run_ragas")
    assert run.router_accuracy(["hybrid"], ["hybrid"]) == 1.0
    assert run.router_accuracy(["hybrid"], ["hybrid", "graph"]) == 0.5
    assert run.router_accuracy([], ["graph"]) == 0.0


def test_summary_by_type():
    run = _load("run_ragas")
    records = [
        {"type": "lookup", "scores": {"faithfulness": 1.0, "router_accuracy": 1.0}},
        {"type": "lookup", "scores": {"faithfulness": 0.5, "router_accuracy": 1.0}},
        {"type": "numeric", "scores": {"faithfulness": None, "router_accuracy": 0.0}},
    ]
    s = run.summarize(records)
    assert s["lookup"]["faithfulness"] == 0.75
    assert s["lookup"]["n"] == 2
    assert s["ALL"]["n"] == 3
    assert "faithfulness" not in s["numeric"]


def test_next_question_id():
    review = _load("review")
    assert review.next_question_id([{"id": "q007"}, {"id": "q002"}, {"id": "cand-x"}]) == "q008"
    assert review.next_question_id([]) == "q001"
