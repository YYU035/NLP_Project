"""Deterministic A3 vector poster: exact text, no generated/OCR technical wording."""
from pathlib import Path
import pymupdf as fitz
import json

ROOT=Path(__file__).resolve().parent
W,H=420*72/25.4,297*72/25.4
NAVY=(.075,.13,.20);TEAL=(.03,.43,.41);GRAY=(.35,.40,.44);LIGHT=(.84,.88,.88)
COLORS=[(.10,.44,.64),(.08,.47,.43),(.69,.45,.09),(.38,.30,.62),(.65,.29,.37)]
doc=fitz.open();page=doc.new_page(width=W,height=H)
page.draw_rect(page.rect,fill=(.975,.978,.965),color=None)
checks=[]

def text(x,y,w,h,value,size=10,color=NAVY,bold=False,align=0):
    rect=fitz.Rect(x,y,x+w,y+h)
    remaining=page.insert_textbox(rect,value,fontname='hebo' if bold else 'helv',fontsize=size,color=color,align=align,lineheight=1.18)
    checks.append({'text':value,'font':size,'box':[x,y,w,h],'remaining':remaining})
    if remaining<0:raise ValueError(f'Overflow {remaining}: {value[:65]}')

def box(x,y,w,h,fill=(1,1,1),color=LIGHT):page.draw_rect(fitz.Rect(x,y,x+w,y+h),fill=fill,color=color,width=.8)

def arrow(x1,y1,x2,y2,color=TEAL):
    page.draw_line((x1,y1),(x2,y2),color=color,width=1.5)
    if x2>x1:
        page.draw_line((x2-5,y2-3),(x2,y2),color=color,width=1.5);page.draw_line((x2-5,y2+3),(x2,y2),color=color,width=1.5)
    elif y2>y1:
        page.draw_line((x2-3,y2-5),(x2,y2),color=color,width=1.5);page.draw_line((x2+3,y2-5),(x2,y2),color=color,width=1.5)

text(30,20,1000,43,'FROM CLAIM TO EVIDENCE',27,bold=True)
text(31,61,1040,23,'Five teammates. One explainable scientific research pipeline.',14,TEAL,bold=True)
text(31,89,1125,18,'Like five detectives: find the books, sort them, choose the clues, check the claim, and show your working.',11.5)
box(30,116,1130,28,fill=(.90,.95,.94),color=(.90,.95,.94))
text(42,123,1100,18,'START: one scientific claim  |  public paper search  |  imported scientific records  |  your local text PDF',10.7,TEAL,bold=True)

stages=[
 ('RAGHAVI','1  FIND PAPERS','The librarian finds papers and remembers where they came from.',
  'OpenAlex / OpenAIRE / arXiv\nspaCy concepts + queries\nCanonical IDs + duplicate handling\nTitles, abstracts and source records\nPreserve the exact original claim',
  'OUTPUT: typed candidate papers'),
 ('NAILA','2  RETRIEVE','One helper matches words. Another matches meaning.',
  'BM25: keyword relevance\nSBERT: MiniLM text embeddings\nCosine similarity compares meaning\nrank-bm25 + NumPy\nSentence-Transformers',
  'OUTPUT: two ordered paper lists'),
 ('YU YUE','3  RANK + SELECT','Combine the lists. Pick exact sentences the checker can read.',
  'RRF: sum of 1 / (60 + rank)\nFrozen MiniLM sentence selector\nScore >=0.50; top3 passages/paper\nKeep exact source spans\nWhole claim + passage <=512 tokens\nReject oversized whole spans',
  'OUTPUT: bounded evidence + IDs'),
 ('CHONG YAO','4  CHECK THE CLAIM','Does each clue agree, disagree, or tell us too little?',
  'Local DeBERTa-v3-large NLI\nPassage = premise; claim = hypothesis\nSUPPORT / CONTRADICT / NEUTRAL\nConfidence threshold:0.50\nAggregate the selected papers\nKeep both sides of a conflict',
  'OUTPUT: verdict + cached predictions'),
 ('KIREETI','5  EXPLAIN + EXPLORE','Show the exact clue. Let people check the answer.',
  'Streamlit interface + Pydantic\nGrounded quotation templates\nValidated citations + source/page IDs\nCompare selected-paper subsets\nJSON export / cached replay\nHuman explanation review',
  'OUTPUT: inspectable, scoped result')]
for i,(owner,title,kid,tech,output) in enumerate(stages):
    x=30+i*230;box(x,158,210,236)
    page.draw_rect(fitz.Rect(x,158,x+210,179),fill=COLORS[i],color=COLORS[i])
    text(x+10,162,190,16,owner,10,(1,1,1),True)
    text(x+10,189,190,30,title,12.3,bold=True)
    text(x+10,224,190,45,kid,10.7,GRAY)
    text(x+10,278,190,93,tech,9.7)
    text(x+10,374,190,16,output,8.4,TEAL,True)
    if i<4:arrow(x+213,255,x+227,255)
text(36,402,1115,20,'AGGREGATED ANSWERS: SUPPORTED / CONTRADICTED / CONFLICTING / INSUFFICIENT. Processing failure is a separate state.',9.4,TEAL,True)

# Shared branches: explicitly label direction and the different XAI techniques.
box(30,432,415,94)
text(42,441,390,17,'EXPLAIN MODEL BEHAVIOUR',11.2,bold=True)
text(42,463,390,43,'Stage4 -> word-MASK occlusion -> Stage5\nCover one word; measure the class-score change. Green = toward\nthe target; red = away. Up to96 words; exact offsets + coverage.',9.5)
text(42,508,390,15,'Captum IG is experimental: failed numerical checks stay hidden.',8.8,GRAY)
box(465,432,325,94)
text(477,441,301,17,'LOCAL PDF SIDE ROUTE',11.2,bold=True)
text(477,463,301,58,'PDF -> PyMuPDF -> Stage3 -> shared verifier/UI\nExact extracted text, page numbers and source SHA-256.\nText PDFs only; scanned/encrypted/empty files fail visibly.\nA PDF link is not proof that full text was inspected.',9.5)
box(810,432,350,94,fill=(.90,.95,.94))
text(822,441,326,17,'CHANGE SELECTION, REUSE THE THINKING',10.7,TEAL,True)
text(822,463,326,57,'Stage5 -> cached aggregation in Stage4 -> Stage5\nNo new NLI inference for paper-subset changes.\nSave JSON -> validate claim / evidence / hashes -> replay.\nConfidence and word scores are not scientific truth.',9.5)

text(30,545,1120,23,'MEASURED RESULTS - different questions, never one pooled accuracy',14,bold=True)
for x in (30,415,800):box(x,577,360,140)
text(42,586,336,18,'FIND RELEVANT PAPERS: RRF RECALL@10',11.1,bold=True)
for y,label,score,note in [(613,'SciFact - biomedical','95.9%','188 dev claims;112 no-positive exclusions;5,183 papers'),
                         (644,'CSFCube - computer science','39.2%','26 fold1-test tasks; judged pools;4,207 records'),
                         (675,'RECIPER - materials science','93.7%','1,024 questions;343 papers; one known source target')]:
    text(42,y,250,18,label,10.3);text(294,y-2,83,22,score,14,TEAL,True,2);text(42,y+17,336,14,note,8.3,GRAY)
text(427,586,336,18,'CHOOSE EVIDENCE + VERIFY CLAIMS',11.1,bold=True)
text(427,612,336,15,'SciFact sentence micro-F1 (188 eligible dev claims)',9.9)
text(427,632,336,20,'0.179 RRF top5  |  0.444 oracle gold papers',12,TEAL,True)
text(427,657,336,18,'Verifier audit:72.4% correct (131/181 pairs)',11.8,bold=True)
text(427,681,336,29,'28 truncated pairs excluded. Gold-cited full abstracts with\nSUPPORT/CONTRADICT gold only; not end-to-end accuracy.',9,GRAY)
text(812,586,336,18,'EXPLAIN + TEST + RESPOND',11.1,bold=True)
text(812,611,336,18,'Occlusion AP0.833 vs random0.438',11.5,TEAL,True)
text(812,632,336,26,'Top-rationale hit75% vs17.5%; only4 pairs from3 abstracts.\nExploratory gold-target study; not IG or general accuracy.',8.9,GRAY)
text(812,663,336,18,'156 frontend +452 owner-suite tests passed',10.9,bold=True)
text(812,685,336,25,'100 cached subset changes:0 new NLI calls; median1.14ms.\nLocal, preloaded-model observation; not a production SLA.',8.9,GRAY)
text(31,724,1127,15,'Recall@10: does a known relevant paper reach the first10? Ranking/evidence results are pinned owner experiments, not independently rerun in full by the frontend.',8.5,GRAY)

page.draw_line((30,747),(1160,747),color=LIGHT,width=1)
rail=[('INTERFACE + CONTRACTS','Python3.12 / Streamlit / Pydantic / pandas\nAltair + PyArrow / typed handoffs'),
      ('SEARCH + RANKING','paper_search / spaCy / HTTP clients\nrank-bm25 / NumPy / Sentence-Transformers'),
      ('LOCAL MODELS','PyTorch / Transformers / MiniLM / DeBERTa NLI\nQwen0.5B optional + gated; HF setup downloads'),
      ('DOCUMENTS + RELIABILITY','PyMuPDF / Captum IG / pytest / Matplotlib\nJSON + SHA-256 / local cache / background futures')]
for i,(title,body) in enumerate(rail):
    x=30+i*285;text(x,759,270,14,title,8.8,TEAL,True);text(x,780,270,30,body,8.7)
text(30,818,1130,15,'No hosted model inference. Original claims and exact evidence preserved. Scientific/source review and actual five-person recording still require humans.',8.4,GRAY)
# File/source provenance is embedded as metadata and a clickable annotation.
page.insert_link({'kind':fitz.LINK_URI,'from':fitz.Rect(30,818,1160,838),'uri':'https://github.com/YYU035/NLP_Project/tree/kireeti/explanation_frontend/poster'})
doc.set_metadata({'title':'From Claim to Evidence - A3 integrated NLP project poster','author':'NLP Project team / Kireeti integration','subject':'Verified five-person architecture and measured results; snapshot9 November2026, code4ca3a22','keywords':'A3, NLP, RRF, BM25, SBERT, DeBERTa, occlusion, reproducibility'})
doc.save(ROOT/'PROJECT_POSTER_A3.pdf',deflate=True)
page.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(ROOT/'PROJECT_POSTER_PREVIEW.png')
(ROOT/'PROJECT_POSTER_A3.svg').write_text(page.get_svg_image(text_as_path=False),encoding='utf-8')
(ROOT/'LAYOUT_CHECKS.json').write_text(json.dumps({'page_mm':[420,297],'page_points':[W,H],'pages':1,'overflow_count':0,'boxes':checks},indent=2),encoding='utf-8')
print('Built one exact A3 landscape page;',len(checks),'text boxes fit')
