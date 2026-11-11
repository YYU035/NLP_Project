"""Actual-model language stress cases and bounded PDF failure/success checks."""
import json,time,hashlib
from pathlib import Path
from evaluation.validation.freeze import ROOT, OUT

def main():
    from research_verifier.pipeline import local_verifier
    from research_verifier.verification.schemas import EvidencePassage
    from research_verifier.ui.pdf import verify_pdf
    from research_verifier.contracts.models import make_context
    from research_verifier.explanations.templates import build_template
    from research_verifier.explanations.grounding import validate_explanation
    from research_verifier.explanations.replay import export_snapshot, import_snapshot
    import pymupdf as fitz
    v=local_verifier();rows=[]
    a='In mice, treatment A reduced mortality by exactly20% compared with placebo.'
    b='In this trial, treatment B did not reduce blood pressure compared with placebo.'
    c='The survey found an association between exposure X and outcome Y. It did not establish that X causes Y.'
    d='Only adults were studied; children were not included. A statistically significant decrease in symptoms was observed in adults.'
    cases=[('literal',a,'Treatment A reduced mortality in mice.','SUPPORT'),('paraphrase',a,'Treatment A decreased the death rate in mice.','SUPPORT'),
        ('negation',a,'Treatment A did not reduce mortality in mice.','CONTRADICT'),('number_change',a,'Treatment A reduced mortality by exactly60% in mice.','CONTRADICT'),
        ('population_change',a,'Treatment A reduced mortality in humans.','NEUTRAL'),('outcome_change',a,'Treatment A reduced body weight in mice.','NEUTRAL'),
        ('unsupported_universal',a,'Treatment A always reduces mortality in every human.','NEUTRAL'),
        ('explicit_null_result',b,'Treatment B did not reduce blood pressure.','SUPPORT'),('reverse_null_result',b,'Treatment B reduced blood pressure.','CONTRADICT'),
        ('absence_is_not_increase',b,'Treatment B increased blood pressure.','NEUTRAL'),
        ('association',c,'Exposure X is associated with outcome Y.','SUPPORT'),('reverse_association',c,'Exposure X is not associated with outcome Y.','CONTRADICT'),
        ('causality_not_established',c,'Exposure X causes outcome Y.','NEUTRAL'),('unstudied_children',d,'The treatment improved symptoms in children.','NEUTRAL'),
        ('studied_adults',d,'Symptoms decreased in the studied adults.','SUPPORT')]
    for name,evidence,claim,expected in cases:
        passage=EvidencePassage(passage_id=name,paper_id='synthetic-stress',section='abstract',text=evidence)
        t=time.perf_counter();pred=v._classify_passage(claim,passage)
        rows.append({'case_id':name,'category':'synthetic_language','claim':claim,'evidence':evidence,'expected':expected,
            'actual':pred.label.value,'confidence':pred.confidence,'passed':pred.label.value==expected,'raw_probabilities':pred.raw_probabilities,'seconds':time.perf_counter()-t,
            'scope':'Author-constructed NLI stress expectation, not real scientific benchmark labels.'})
    # Repeated exact pairs: IDs and metadata may change, model input/outputs must not.
    repeated=[]
    for row in rows[:5]:
        p=EvidencePassage(passage_id='repeat',paper_id='repeat',section='abstract',text=row['evidence'])
        pred=v._classify_passage(row['claim'],p)
        delta=max(abs(pred.raw_probabilities[k]-value) for k,value in row['raw_probabilities'].items())
        repeated.append({'case_id':row['case_id'],'max_probability_difference':delta,'passed':pred.label.value==row['actual'] and delta<=1e-6})
    directory=OUT/'pdf';directory.mkdir(exist_ok=True)
    doc=fitz.open();page=doc.new_page();page.insert_textbox(fitz.Rect(40,40,270,220),'LEFT COLUMN\nPlants require water to survive.\nThis is a constructed integration test.',fontsize=11)
    page.insert_textbox(fitz.Rect(320,40,560,220),'RIGHT COLUMN\nThe study contains no claim about children.\nReading order must remain inspectable.',fontsize=11)
    valid=doc.tobytes();doc.close()
    scanned=fitz.open();page=scanned.new_page();page.draw_rect(fitz.Rect(10,10,200,200),fill=(.5,.5,.5));image_only=scanned.tobytes();scanned.close()
    encrypted=fitz.open(stream=valid,filetype='pdf');encrypted_bytes=encrypted.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256,owner_pw='validation-owner',user_pw='validation-reader');encrypted.close()
    too_many=fitz.open()
    for _ in range(201):too_many.new_page()
    too_many_bytes=too_many.tobytes();too_many.close()
    payloads=[('multi_column_text',valid,False),('image_only_no_ocr',image_only,True),('encrypted',encrypted_bytes,True),('malformed',b'%PDF-not-a-real-pdf',True),('too_many_pages',too_many_bytes,True),('oversized',b'%PDF-'+b'x'*(20*1024*1024),True)]
    pdf_rows=[]
    for name,payload,should_fail in payloads:
        t=time.perf_counter()
        try:
            snapshot=verify_pdf(payload,'Plants require water to survive.')
            ctx=make_context(snapshot,list(snapshot.papers),snapshot.provisional);ex=build_template(ctx);saved=export_snapshot(ctx,ex)
            good=validate_explanation(ex,ctx).displayable and import_snapshot(saved).snapshot.fingerprint==snapshot.fingerprint
            if not should_fail:(directory/(name+'-replay.json')).write_text(saved,encoding='utf-8')
            pdf_rows.append({'case_id':name,'expected':'reject' if should_fail else 'page_preserving_verification','passed':good and not should_fail,
                'actual':'accepted','pages':[p.page for p in snapshot.verification.evidence_passages],'status':snapshot.provisional.status,'verdict':snapshot.provisional.aggregated_verdict,'seconds':time.perf_counter()-t})
        except Exception as exc:
            pdf_rows.append({'case_id':name,'expected':'reject' if should_fail else 'page_preserving_verification','passed':should_fail,'actual':'rejected','error_type':type(exc).__name__,'reason':str(exc),'seconds':time.perf_counter()-t})
    # One original scientific PDF, independent of fabricated layout fixtures.
    real_row={'case_id':'real_arxiv_pdf','source':'https://arxiv.org/pdf/1703.01365','scope':'Extraction/page/span/replay functionality, not verified scientific correctness.'}
    try:
        import requests
        path=directory/'arxiv-1703.01365.pdf'
        if not path.exists():
            with requests.get(real_row['source'],timeout=60,stream=True) as response:
                response.raise_for_status();total=0
                with path.open('wb') as handle:
                    for chunk in response.iter_content(1024*1024):
                        total+=len(chunk)
                        if total>20*1024*1024:raise ValueError('Real PDF exceeds20MiB')
                        handle.write(chunk)
        real=verify_pdf(path.read_bytes(),'Integrated Gradients satisfies Sensitivity and Implementation Invariance.')
        ctx=make_context(real,list(real.papers),real.provisional);ex=build_template(ctx);saved=export_snapshot(ctx,ex)
        assert validate_explanation(ex,ctx).displayable and import_snapshot(saved).snapshot.fingerprint==real.fingerprint
        real_row.update(passed=True,actual='verified_and_replayed',verdict=real.provisional.aggregated_verdict,pages=[p.page for p in real.verification.evidence_passages],sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (directory/'real-arxiv-replay.json').write_text(saved,encoding='utf-8')
    except Exception as exc:real_row.update(passed=False,actual='unavailable_or_failed',reason=f'{type(exc).__name__}: {exc}')
    pdf_rows.append(real_row)
    report={'synthetic_nli_cases':rows,'synthetic_nli_passed':sum(r['passed'] for r in rows),'synthetic_nli_total':len(rows),
        'repeatability':repeated,'pdf_cases':pdf_rows,'limits':'Synthetic expectations are diagnostic, not dataset accuracy; real PDF scientific meaning/reading order still needs inspection.'}
    (OUT/'stress.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'synthetic_passed':report['synthetic_nli_passed'],'synthetic_total':len(rows),'repeatability_passed':sum(r['passed'] for r in repeated),'pdf_passed':sum(r['passed'] for r in pdf_rows),'pdf_total':len(pdf_rows)}),flush=True)
    return report
if __name__=='__main__':main()
