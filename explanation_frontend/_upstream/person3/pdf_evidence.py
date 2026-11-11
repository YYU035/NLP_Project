"""Local full-text/PDF evidence selection with exact page text references."""

from dataclasses import asdict
import hashlib
from pathlib import Path

from .evidence_selection import EvidenceSpan
from .verifier_budget import enforce_verifier_budget, VERIFIER_CONTRACT


def extract_pdf(path, paper_id, *, source_url=None, max_pages=200):
    """Extract a local PDF only; encrypted/scanned/empty PDFs fail explicitly.

    Offsets are into each original PyMuPDF-extracted page string, not a claim
    that PDF bytes are plain text. Reading order requires separate visual QC.
    """
    import pymupdf
    path=Path(path)
    if not paper_id or not path.is_file():raise ValueError('Existing local PDF and paper ID required')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    with pymupdf.open(path) as document:
        if document.needs_pass:raise ValueError('Encrypted PDF requires an unlocked source')
        if len(document)>max_pages:raise ValueError('PDF exceeds explicit page limit')
        pages=[{'page':index+1,'text':page.get_text('text',sort=False)} for index,page in enumerate(document)]
    if not any(p['text'].strip() for p in pages):raise ValueError('No extractable PDF text; OCR is not implemented')
    for page in pages:page['text_sha256']=hashlib.sha256(page['text'].encode()).hexdigest()
    return {'paper_id':paper_id,'source_file':str(path.resolve()),'source_url':source_url,
            'source_sha256':digest,'pages':pages,'extraction':{'library':'PyMuPDF','version':pymupdf.VersionBind,'sort':False,'reading_order':'PDF content stream; visual review required'},
            'limitations':['PDF layout/reading order requires visual review',
                           'No OCR or reconstruction of image tables',
                           'Context windows never silently cross page boundaries']}


def select_full_text(claim, document, selector, tokenizer, *, top_k=3):
    """Select a bounded set of whole passages; exact native P4 payload plus refs."""
    if type(top_k) is not int or not 1<=top_k<=3:
        raise ValueError('top_k must be 1..3, matching Person 4 maximum votes per paper')
    candidates=[];references={}
    for page in document['pages']:
        for span in selector.select_abstract(claim,document['paper_id'],page['text']):
            pid=f"{document['paper_id']}:pdf:p{page['page']}:{span.start}:{span.end}:{hashlib.sha256(span.text.encode()).hexdigest()[:12]}"
            converted=EvidenceSpan(span.paper_id,span.sentence_indices,span.start,span.end,span.text,span.score,pid)
            candidates.append(converted)
            references[pid]={'page':page['page'],'start':span.start,'end':span.end,
                'page_text_sha256':page['text_sha256'],'source_sha256':document['source_sha256'],
                'source_url':document.get('source_url'),'sentence_index_origin':'heuristic_page_segmentation'}
    candidates.sort(key=lambda span:(-span.score,references[span.passage_id]['page'],span.start))
    decision=enforce_verifier_budget(claim,candidates,tokenizer)
    accepted=list(decision.accepted)[:top_k]
    from .semantic_evidence import person4_passages
    native=person4_passages(accepted)
    for row in native:row['section']='full_text'
    output={'schema_version':'person3-fulltext-1','claim':claim,'paper_id':document['paper_id'],
        'evidence_type':'full_text','selection_configuration':asdict(selector.config) if hasattr(selector,'config') else {'selector':'injected'},'passages':native,
        'source_spans':[{**asdict(span),**references[span.passage_id]} for span in accepted],
        'budget_checks':list(decision.checked),'rejected_passages':list(decision.rejected),
        'verifier_contract':VERIFIER_CONTRACT,'source_sha256':document['source_sha256'],
        'extraction':document.get('extraction',{'method':'caller_supplied_page_text'}),
        'status':'selected' if accepted else 'verifier_budget_rejected' if decision.rejected else 'no_passage_above_threshold',
        'review_status':'human_review_pending','limitations':document.get('limitations',[])}
    for span in output['source_spans']:
        page=next(p for p in document['pages'] if p['page']==span['page'])
        if page['text'][span['start']:span['end']]!=span['text']:raise ValueError('PDF source span mismatch')
    return output


def document_from_pages(paper_id,pages,*,source_reference):
    """Allow already extracted full text from Person 1 without requiring a PDF."""
    if not paper_id or not source_reference or isinstance(pages, str) or any(not isinstance(t,str) for t in pages):
        raise ValueError('Paper ID, source reference and page text strings are required')
    digest=hashlib.sha256('\f'.join(pages).encode()).hexdigest()
    return {'paper_id':paper_id,'source_file':source_reference,'source_url':None,'source_sha256':digest,
            'pages':[{'page':i+1,'text':t,'text_sha256':hashlib.sha256(t.encode()).hexdigest()} for i,t in enumerate(pages)],
            'extraction':{'method':'upstream_page_text'},'limitations':['Page text supplied upstream; extraction authenticity is not independently verified']}
