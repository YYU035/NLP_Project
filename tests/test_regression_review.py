import csv
import json
import pytest
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlparse
from evaluation.regression_review import select_review_cases, reaggregate_scope


def test_review_selection_covers_transitions_and_is_independent_of_input_order():
    transitions=[('SUPPORTED','INSUFFICIENT'),('CONTRADICTED','INSUFFICIENT'),('CONTRADICTED','SUPPORTED'),('CONTRADICTED','CONFLICTING'),('INSUFFICIENT','SUPPORTED')]
    reg=[{'claim_id':str(i+10),'gold':g,'prediction':p,'condition':'yuyue_frozen','correct':False,'baseline_correct':True} for i,(g,p) in enumerate(transitions)]
    reg.append({**reg[0],'claim_id':'999'})
    extra=[{'claim_id':'20','gold':'SUPPORTED','condition':'yuyue_frozen','correct':True,'baseline_correct':False},
           {'claim_id':'21','gold':'INSUFFICIENT','condition':'yuyue_frozen','correct':True,'baseline_correct':False},
           {'claim_id':'22','gold':'CONTRADICTED','condition':'yuyue_frozen','correct':False,'baseline_correct':False}]
    a=select_review_cases(reg,reg+extra)
    assert a==select_review_cases(list(reversed(reg)),list(reversed(reg+extra)))
    assert len(a)==8 and '999' not in {cid for cid,_ in a}


def test_scope_counterfactual_uses_recorded_paper_verdicts_without_model_initialization(monkeypatch):
    from research_verifier.verification.engine import ClaimVerifier
    from research_verifier.verification.schemas import VerificationResponse
    def fail(*args,**kwargs):raise AssertionError('Counterfactual must not initialize an NLI model')
    monkeypatch.setattr(ClaimVerifier,'__init__',fail)
    response=VerificationResponse.model_validate({'claim':'Recorded claim','per_paper_verdicts':[{'paper_id':'a','verdict':'SUPPORTED','confidence':.9},{'paper_id':'b','verdict':'INSUFFICIENT','confidence':0.}],
        'model_settings':{'checkpoint':'recorded-test','revision':'fixed','tokenizer_max_length':512,'threshold':.5},'per_passage_labels':[]})
    assert reaggregate_scope(response,{'a','b'})=='SUPPORTED'
    assert reaggregate_scope(response,{'b'})=='INSUFFICIENT'
    assert reaggregate_scope(response,set())=='INSUFFICIENT'


def test_generated_review_has_blank_human_fields_and_portable_offline_links():
    root=Path(__file__).parents[1]/'evaluation/regression_review_20261111'
    with (root/'REVIEW_FORM.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==9
    diagnostics=json.loads((root/'case_diagnostics.json').read_text(encoding='utf-8'))
    assert all(d['diagnosis'] in ('recovered_baseline_error','persistent_error_not_a_regression') for d in diagnostics if not d['is_regression'])
    assert all(row[field]=='' for row in rows for field in ['evidence_supports_verdict','citation_correct','context_helpful','reviewer_name','issue_notes'])
    class Links(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True);self.in_quote=False;self.parts=[];self.quotes=[]
        def handle_data(self,data):
            if self.in_quote:self.parts.append(data)
        def handle_endtag(self,tag):
            if tag=='blockquote':self.quotes.append(''.join(self.parts));self.parts=[];self.in_quote=False
        def handle_starttag(self,tag,attrs):
            assert tag!='script'
            if tag=='blockquote':self.in_quote=True;self.parts=[]
            for name,value in attrs:
                assert not name.lower().startswith('on')
                if name not in ['href','src']:continue
                if value.startswith('https:'):
                    assert value=='https://arxiv.org/pdf/1703.01365';continue
                path=(self.current.parent/urlparse(value).path).resolve()
                assert path.is_relative_to(root.resolve()) and path.is_file(),value
    parser=Links()
    for page in root.rglob('*.html'):
        parser.current=page;parser.quotes=[];parser.feed(page.read_text(encoding='utf-8'))
        if page.stem.isdigit():
            from research_verifier.explanations.replay import import_snapshot
            for arm in ['baseline','yuyue_frozen']:
                bundle=import_snapshot((root/'replays'/f'{page.stem}-{arm}.json').read_bytes())
                assert all(p.text in parser.quotes for p in bundle.snapshot.verification.evidence_passages)


def test_rebuild_refuses_to_overwrite_existing_human_review(monkeypatch,tmp_path):
    import evaluation.regression_review as module
    monkeypatch.setattr(module,'DEST',tmp_path)
    form=tmp_path/'REVIEW_FORM.csv';original='case_id,evidence_supports_verdict,issue_notes\n1,yes,My source review\n'
    form.write_text(original,encoding='utf-8')
    with pytest.raises(ValueError,match='refusing to overwrite human input'):module.main()
    assert form.read_text(encoding='utf-8')==original
