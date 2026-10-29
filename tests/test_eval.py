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


def test_retrieval_scores():
    run = _load("run_ragas")
    s = run.retrieval_scores(["g"], ["x", "g", "y"], ["x", "y", "g"])
    assert s == {"retrieval_hit": 1.0, "rerank_hit": 1.0, "mrr": 1 / 3}
    # Found by hybrid search but dropped by the reranker
    s = run.retrieval_scores(["g"], ["g", "x"], ["x"])
    assert s == {"retrieval_hit": 1.0, "rerank_hit": 0.0, "mrr": 0.0}
    assert run.retrieval_scores([], ["g"], ["g"]) == {}


def test_same_text_matches_repeated_paragraph_not_neighbour():
    rr = _load("run_retrieval")
    a = "Our profit margins vary due to a number of factors including product mix. "
    b = "We operate in a highly competitive market that might affect our pricing. "
    c = "Because we own much of our manufacturing capacity, many costs are fixed. "
    gold = a + b + c
    assert rr.same_text(gold, "Item 1A. Risk Factors. " + a + b + c)  # other year, shifted
    assert not rr.same_text(gold, c + "An unrelated sentence about distributors and their pay.")
    assert not rr.same_text("short.", "short.")  # nothing long enough to compare


def test_score_ranking():
    rr = _load("run_retrieval")
    gold = ["acc1:1A:5"]
    top = [("acc1:1A:4", "x"), ("acc2:1A:5", "y"), ("acc1:1A:5", "z")]
    s = rr.score_ranking(gold, ["gold text"], top, [cid for cid, _ in top])
    assert s["hit@1"] == 0.0 and s["hit@3"] == 1.0 and s["mrr"] == 1 / 3
    assert s["recall@30"] == 1.0 and s["text_hit@8"] == 1.0
    assert s["same_filing@8"] == 2 / 3
    miss = rr.score_ranking(gold, ["gold text"], [("acc2:7:1", "other")], ["acc2:7:1", *gold])
    assert miss["hit@8"] == 0.0 and miss["recall@30"] == 1.0 and miss["text_hit@8"] == 0.0


def test_retrieval_summary_groups():
    rr = _load("run_retrieval")
    recs = [
        {"type": "lookup", "fiscal_year": 2024, "scores": {"vector": {"mrr": 1.0}}},
        {"type": "lookup", "fiscal_year": 2025, "scores": {"vector": {"mrr": 0.0}}},
    ]
    s = rr.summarize(recs)
    assert s["ALL"]["vector"] == {"mrr": 0.5, "n": 2}
    assert s["FY2024"]["vector"]["mrr"] == 1.0
