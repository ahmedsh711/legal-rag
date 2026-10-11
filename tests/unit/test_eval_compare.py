"""Promotion rule for a candidate index: no worse than production, per language."""

import json

from legalrag.eval.compare import compare_runs, main


def metrics(mrr=1.0, hit1=1.0, errors=0, n=28, en_mrr=None):
    block = {"n": n, "errors": errors, "hit_at_1": hit1, "mrr": mrr}
    en = {**block, "mrr": en_mrr if en_mrr is not None else mrr}
    return {"all": block, "ar": block, "en": en}


def test_an_equal_candidate_is_promoted():
    verdict = compare_runs(metrics(), metrics(), margin=0.02)
    assert verdict["promote"] and verdict["reasons"] == []


def test_a_drop_within_the_margin_is_allowed():
    assert compare_runs(metrics(mrr=0.985), metrics(), margin=0.02)["promote"]


def test_a_drop_in_one_language_blocks_promotion():
    verdict = compare_runs(metrics(en_mrr=0.9), metrics(), margin=0.02)
    assert not verdict["promote"]
    assert verdict["reasons"] == ["en mrr 0.900 < production 1.000 - 0.02"]


def test_errors_or_a_different_question_count_block_promotion():
    assert not compare_runs(metrics(errors=2), metrics(), margin=0.02)["promote"]
    assert not compare_runs(metrics(n=20), metrics(), margin=0.02)["promote"]


def test_cli_prints_one_json_line(tmp_path, capsys):
    cand, prod = tmp_path / "cand.json", tmp_path / "prod.json"
    cand.write_text(json.dumps(metrics(mrr=0.5)), encoding="utf-8")
    prod.write_text(json.dumps(metrics()), encoding="utf-8")
    main(["--candidate", str(cand), "--production", str(prod)])
    verdict = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert verdict["promote"] is False and verdict["reasons"]
