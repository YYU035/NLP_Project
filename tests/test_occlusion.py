import pytest
pytest.importorskip('torch');pytest.importorskip('captum');pytest.importorskip('transformers')
from tests.test_integrated_gradients import model_dir, verifier, PASSAGE, CLAIM_TXT
from research_verifier.interpretability.occlusion import explain_occlusion
from research_verifier.interpretability.service import explain_checked


def test_occlusion_scores_are_actual_masked_word_probability_differences(verifier):
    import torch
    result=explain_occlusion(verifier,CLAIM_TXT,PASSAGE,max_words=4,batch_size=1)
    assert not result['coverage']['complete'] and result['kind']=='word_occlusion'
    word=result['words'][0]
    enc=verifier.tokenizer(PASSAGE.text,CLAIM_TXT,return_tensors='pt')
    enc['input_ids'][0,word['token_indices']]=verifier.tokenizer.mask_token_id
    target=next(i for i,l in verifier.label_mapping.items() if l.value==result['target_label'])
    with torch.no_grad():masked=float(verifier.model(**enc).logits.softmax(-1)[0,target])
    assert word['attribution']==pytest.approx(result['target_probability']-masked,abs=1e-6)
    again=explain_occlusion(verifier,CLAIM_TXT,PASSAGE,max_words=4,batch_size=4)
    assert [w['attribution'] for w in again['words']]==pytest.approx([w['attribution'] for w in result['words']],abs=1e-6)
    for w in result['words']:
        original=PASSAGE.text if w['segment']=='evidence' else CLAIM_TXT
        assert original[w['start']:w['end']].strip()==w['text']


def test_occlusion_refuses_empty_overbudget_and_bad_budgets(verifier,monkeypatch):
    with pytest.raises(ValueError):explain_occlusion(verifier,' ',PASSAGE)
    with pytest.raises(ValueError):explain_occlusion(verifier,CLAIM_TXT,PASSAGE,max_words=0)
    monkeypatch.setattr(verifier,'max_length',2)
    with pytest.raises(ValueError,match='budget'):explain_occlusion(verifier,CLAIM_TXT,PASSAGE)


def test_adaptive_ig_matches_original_probability_and_respects_budget(verifier):
    result=explain_checked(verifier,CLAIM_TXT,PASSAGE,steps=(72,),integration='adaptive')
    pred=verifier._classify_passage(CLAIM_TXT,PASSAGE)
    assert result['target_probability']==pytest.approx(pred.raw_probabilities[result['target_class']],abs=1e-6)
    assert result['quadrature']['evaluations']<=72
    assert result['numerical_status']=='passed'
    assert result['quadrature']['input_probability_parity_error']<=1e-6
