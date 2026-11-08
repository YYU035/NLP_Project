from types import SimpleNamespace
import pytest
from ranking_evaluation.evidence_selection import EvidenceSpan
from ranking_evaluation.verifier_budget import enforce_verifier_budget
from ranking_evaluation.pdf_evidence import document_from_pages,select_full_text


class Tokenizer:
    def __init__(self):self.calls=[]
    def encode(self,text,claim,**kwargs):
        self.calls.append((text,claim,kwargs))
        return list(range(len(text.split())+len(claim.split())+3))


def span(text):return EvidenceSpan('p',(0,),0,len(text),text,1,'p:s0')


def test_exact_pair_orientation_boundary_and_no_silent_truncation():
    tokenizer=Tokenizer();claim='claim'
    fitting=span(' '.join(['word']*508));too_long=span(' '.join(['word']*509))
    result=enforce_verifier_budget(claim,[fitting,too_long],tokenizer)
    assert result.accepted==(fitting,)
    assert result.checked[0]['pair_tokens']==512
    assert result.rejected[0]['pair_tokens']==513
    assert result.rejected[0]['text_modified'] is False
    assert all(c[1]==claim and c[2]=={'add_special_tokens':True,'truncation':False} for c in tokenizer.calls)


def test_long_original_claim_is_not_shortened():
    claim=' '.join(['claim']*510)
    result=enforce_verifier_budget(claim,[span('evidence')],Tokenizer())
    assert not result.accepted
    assert result.rejected[0]['reason']=='original_claim_exceeds_pair_budget'
    with pytest.raises(ValueError):enforce_verifier_budget('claim',[],None)
    with pytest.raises(ValueError):enforce_verifier_budget('claim',[],Tokenizer(),max_pair_tokens=1024)


def test_fulltext_page_identity_and_bounded_person4_payload():
    document=document_from_pages('p',['First page text.','Second page exact text.'],source_reference='upstream:doc1')
    selector=SimpleNamespace(select_abstract=lambda claim,pid,text:[EvidenceSpan(pid,(0,),0,len(text),text,1,pid+':s')])
    result=select_full_text('claim',document,selector,Tokenizer(),top_k=1)
    assert len(result['passages'])==1
    assert result['passages'][0]['section']=='full_text'
    assert result['source_spans'][0]['page']==1
    assert result['source_spans'][0]['text']==document['pages'][0]['text']
    assert result['review_status']=='human_review_pending'


def test_local_pdf_extraction_has_source_hash_pages_and_no_ocr_claim(tmp_path):
    pymupdf=pytest.importorskip('pymupdf')
    from ranking_evaluation.pdf_evidence import extract_pdf
    path=tmp_path/'source.pdf'
    document=pymupdf.open();page=document.new_page()
    page.insert_text((50,60),'Left column first sentence.')
    page.insert_text((320,60),'Right column second sentence.')
    document.save(path);document.close()
    result=extract_pdf(path,'p')
    assert result['pages'][0]['page']==1
    assert 'Left column' in result['pages'][0]['text']
    assert result['source_sha256']
    assert result['extraction']['sort'] is False
    blank=tmp_path/'blank.pdf';document=pymupdf.open();document.new_page();document.save(blank);document.close()
    with pytest.raises(ValueError,match='OCR'):
        extract_pdf(blank,'empty')


def test_review_validator_never_counts_blank_or_edited_forms_as_human_review(tmp_path):
    import csv
    from ranking_evaluation.scripts.validate_pdf_review import validate
    row={'sample_id':'synthetic-1','claim':'synthetic claim','paper_id':'p','page':'1','start':'0','end':'3',
         'passage_id':'p:1','passage':'ABC','reviewer':'','reviewed_at':'','source_matches':'',
         'relevant_to_claim':'','context_sufficient':'','within_input_limit':''}
    def write(path,record):
        with path.open('w',newline='') as h:
            writer=csv.DictWriter(h,fieldnames=list(record));writer.writeheader();writer.writerow(record)
    write(tmp_path/'human-review.csv',row)
    completed=tmp_path/'completed.csv';write(completed,row)
    with pytest.raises(ValueError,match='Reviewer'):validate(tmp_path,completed)
    row.update(reviewer='Synthetic test reviewer',reviewed_at='2026-11-07T00:00:00Z',
               source_matches='yes',relevant_to_claim='no',context_sufficient='uncertain',within_input_limit='yes')
    write(completed,row)
    result=validate(tmp_path,completed)
    assert result['ratings']['relevant_to_claim']=={'no':1}
    row['passage']='changed';write(completed,row)
    with pytest.raises(ValueError,match='source changed'):validate(tmp_path,completed)


def test_overlong_claim_remains_operational_even_if_semantic_selection_is_empty():
    result=enforce_verifier_budget(' '.join(['claim']*510),[],Tokenizer())
    assert not result.accepted
    assert result.rejected[0]['passage_id'] is None
    assert result.rejected[0]['reason']=='original_claim_exceeds_pair_budget'
