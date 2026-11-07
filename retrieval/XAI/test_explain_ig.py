"""
Tests for explain_ig.py. Run:  python -m pytest -q test_explain_ig.py

Uses a tiny randomly-initialised BERT NLI model saved to a temp folder and
loaded through the real ClaimVerifier, so no download or GPU is needed.
The attributions of a random model are meaningless; these tests check the
plumbing (input order, label mapping, completeness, grouping, metrics).
"""
import json
import os

import pytest
import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

from eval_ig_dev import average_precision, build_examples, main as eval_main
from explain_ig import CLAIM, EVIDENCE, IGExplainer, to_html
from schemas import EvidencePassage, PassageLabel
from verifier import ClaimVerifier

WORDS = ("vitamin d supplementation reduced hip fractures in elderly adults the effect was dose "
         "dependent coffee causes heart disease not associated with moderate intake supplements "
         "prevent calcium did reduce fracture incidence older women . , <b>").split()


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny_nli")
    vocab = {t: i for i, t in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + WORDS + ["##s"])}
    tk = Tokenizer(models.WordPiece(vocab=vocab, unk_token="[UNK]"))
    tk.normalizer = normalizers.BertNormalizer(lowercase=True)
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tk.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])])
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]",
                                  cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2,
                     num_attention_heads=2, intermediate_size=64, num_labels=3,
                     max_position_embeddings=128,
                     id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
                     label2id={"entailment": 0, "neutral": 1, "contradiction": 2})
    torch.manual_seed(0)
    BertForSequenceClassification(cfg).save_pretrained(d)
    tok.save_pretrained(d)
    return str(d)


@pytest.fixture(scope="session")
def verifier(model_dir):
    return ClaimVerifier(checkpoint=model_dir, device="cpu")


@pytest.fixture(scope="session")
def explainer(verifier):
    return IGExplainer(verifier, n_steps=64)


CLAIM_TXT = "Vitamin D supplements prevent hip fractures."
PASSAGE = EvidencePassage(passage_id="p1", paper_id="d1", section="abstract",
                          text="Vitamin D supplementation reduced hip fractures in elderly adults. "
                               "The effect was dose dependent.")


def test_inputs_match_verifier(explainer, verifier):
    enc = explainer._encode(PASSAGE.text, CLAIM_TXT)
    ref = verifier.tokenizer(PASSAGE.text, CLAIM_TXT, max_length=verifier.max_length,
                             truncation=True)["input_ids"]
    assert enc["ids"] == ref
    # premise first (segment 0), claim second (segment 1)
    toks = verifier.tokenizer.convert_ids_to_tokens(enc["ids"])
    first_ev = enc["seq_ids"].index(0)
    first_claim = enc["seq_ids"].index(1)
    assert first_ev < first_claim and toks[first_ev] == "vitamin"


def test_probability_matches_verifier(explainer, verifier):
    exp = explainer.explain(CLAIM_TXT, PASSAGE)
    pv = verifier._classify_passage(CLAIM_TXT, PASSAGE)
    assert exp.target_probability == pytest.approx(pv.raw_probabilities[exp.target_class], abs=1e-5)
    best = max(pv.raw_probabilities, key=pv.raw_probabilities.get)
    assert exp.target_class == best  # default target = model's predicted class


def test_explicit_target_and_label_mapping(explainer):
    exp = explainer.explain(CLAIM_TXT, PASSAGE, target=PassageLabel.CONTRADICT)
    assert exp.target_label == PassageLabel.CONTRADICT and exp.target_class == "contradiction"
    exp = explainer.explain(CLAIM_TXT, PASSAGE, target=PassageLabel.SUPPORT)
    assert exp.target_class == "entailment"


def test_completeness_axiom(explainer):
    exp = explainer.explain(CLAIM_TXT, PASSAGE)
    total = sum(w.attribution for w in exp.words)
    assert total == pytest.approx(exp.target_probability - exp.baseline_probability, abs=2e-3)
    assert abs(exp.convergence_delta) < 2e-3


def test_word_grouping_and_offsets(explainer):
    exp = explainer.explain(CLAIM_TXT, PASSAGE)
    ev = [w for w in exp.words if w.segment == EVIDENCE]
    cl = [w for w in exp.words if w.segment == CLAIM]
    assert [w.text for w in cl] == ["Vitamin", "D", "supplements", "prevent", "hip", "fractures."]
    for w in ev:
        assert PASSAGE.text[w.start:w.end] == w.text
    for w in cl:
        assert CLAIM_TXT[w.start:w.end] == w.text
    assert max(abs(w.normalized) for w in exp.words) == pytest.approx(1.0)


def test_sentence_scores_partition_evidence(explainer):
    sents = ["Vitamin D supplementation reduced hip fractures in elderly adults.",
             "The effect was dose dependent.", "Coffee causes heart disease."]
    exp = explainer.explain(CLAIM_TXT, PASSAGE, sentences=sents)
    assert exp.evidence_text == " ".join(sents) and len(exp.sentence_scores) == 3
    ev_total = sum(w.attribution for w in exp.words if w.segment == EVIDENCE)
    assert sum(exp.sentence_scores) == pytest.approx(ev_total, abs=1e-6)


def test_truncation_flagged(verifier, model_dir):
    v = ClaimVerifier(checkpoint=model_dir, device="cpu", tokenizer_max_length=16)
    exp = IGExplainer(v, n_steps=8).explain(CLAIM_TXT, PASSAGE)
    assert exp.truncated
    assert v._classify_passage(CLAIM_TXT, PASSAGE).truncated  # agrees with the verifier


def test_faithfulness_metrics(explainer):
    exp, faith = explainer.explain_with_faithfulness(CLAIM_TXT, PASSAGE, n_random=2)
    assert faith["p_full"] == pytest.approx(exp.target_probability, abs=1e-5)
    for m in ("comprehensiveness", "sufficiency", "comprehensiveness_random", "sufficiency_random"):
        assert set(faith[m]) == {0.01, 0.05, 0.10, 0.20, 0.50}
        assert all(-1 <= v <= 1 for v in faith[m].values())
    _, again = explainer.explain_with_faithfulness(CLAIM_TXT, PASSAGE, n_random=2)
    assert again["aopc_comprehensiveness_random"] == faith["aopc_comprehensiveness_random"]


def test_explain_response(model_dir):
    v = ClaimVerifier(checkpoint=model_dir, device="cpu", threshold=0.0)
    papers = [{"paper_id": "d1", "title": "t"}, {"paper_id": "d2", "title": "u"}]
    p2 = EvidencePassage(passage_id="p2", paper_id="d2", section="abstract",
                         text="Coffee was not associated with heart disease.")
    resp = v.verify(CLAIM_TXT, papers, {"d1": [PASSAGE], "d2": [p2]})
    exps = IGExplainer(v, n_steps=8).explain_response(resp)
    decisive = {pid for pv in resp.per_paper_verdicts
                for pid in pv.supporting_passage_ids + pv.contradicting_passage_ids}
    assert {e.passage_id for e in exps} == decisive
    labels = {pv.passage_id: pv.label for pv in resp.per_passage_labels}
    assert all(e.target_label == labels[e.passage_id] for e in exps)


def test_html_escapes(explainer):
    p = EvidencePassage(passage_id="x", paper_id="y", section="abstract", text="<b> coffee causes")
    out = to_html(explainer.explain("coffee causes heart disease", p, target=PassageLabel.SUPPORT))
    assert "<b> coffee" not in out and "&lt;b&gt;" in out


def test_average_precision():
    assert average_precision([0.9, 0.1, 0.5], {0}) == 1.0
    assert average_precision([0.1, 0.9, 0.5], {0}) == pytest.approx(1 / 3)
    assert average_precision([0.9, 0.8, 0.1], {0, 2}) == pytest.approx((1 + 2 / 3) / 2)


def test_eval_script_end_to_end(verifier, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    corpus = [{"doc_id": 1, "title": "a", "abstract": [
        "Vitamin D supplementation reduced hip fractures in elderly adults.",
        "The effect was dose dependent."]},
              {"doc_id": 2, "title": "b", "abstract": [
        "Moderate coffee intake was not associated with heart disease.", "Coffee causes."]}]
    claims = [{"id": 1, "claim": CLAIM_TXT, "cited_doc_ids": [1],
               "evidence": {"1": [{"sentences": [0], "label": "SUPPORT"}]}},
              {"id": 2, "claim": "Coffee causes heart disease.", "cited_doc_ids": [2],
               "evidence": {"2": [{"sentences": [0], "label": "CONTRADICT"}]}},
              {"id": 3, "claim": "NEI claim.", "evidence": {}, "cited_doc_ids": [1]}]
    (data / "corpus.jsonl").write_text("\n".join(json.dumps(c) for c in corpus))
    (data / "claims_dev.jsonl").write_text("\n".join(json.dumps(c) for c in claims))
    assert len(build_examples(str(data))) == 2
    out = tmp_path / "out"
    summary = eval_main(["--data-dir", str(data), "--out-dir", str(out), "--n-steps", "8"],
                        verifier=verifier)
    assert summary["all"]["n"] == 2
    for f in ("per_example.jsonl", "summary.json", "summary.md"):
        assert (out / f).exists()
