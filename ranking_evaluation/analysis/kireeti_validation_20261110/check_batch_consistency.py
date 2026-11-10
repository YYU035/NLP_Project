import json,sys,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
root=Path.cwd();sys.path.insert(0,str(root))
from ranking_evaluation.analysis.kireeti_validation_20261110.batch_verification import classify_batched
from explanation_frontend._upstream.claim_verifier.verifier import ClaimVerifier
from explanation_frontend._upstream.claim_verifier.schemas import EvidencePassage
v=object.__new__(ClaimVerifier);v.device='mps';v.max_length=512;v.threshold=.5
p=root/'.cache/ranking/verifier-model';torch.set_num_threads(4)
v.tokenizer=AutoTokenizer.from_pretrained(str(p),local_files_only=True)
v.model=AutoModelForSequenceClassification.from_pretrained(str(p),local_files_only=True,dtype=torch.float32).to('mps').eval();v.label_mapping=v._build_label_mapping()
source=root/'ranking_evaluation/analysis/kireeti_validation_20261110/mps-replay-nli/predictions.jsonl'
pairs=[];expected=[]
for row in map(json.loads,source.open()):
 labels={r['passage_id']:r for r in row['response']['per_passage_labels']}
 for ps in row['passages'].values():
  for raw in ps:pairs.append((row['claim'],EvidencePassage.model_validate(raw)));expected.append(labels[raw['passage_id']])
start=time.monotonic();got=classify_batched(v,pairs,batch_size=4);torch.mps.synchronize();elapsed=time.monotonic()-start
labels=sum(g.label.value==e['label'] for g,e in zip(got,expected));delta=max(abs(g.confidence-e['confidence']) for g,e in zip(got,expected))
r={'pairs':len(pairs),'label_matches':labels,'max_confidence_delta':delta,'seconds':elapsed,'batch_size':4};print(json.dumps(r));assert labels==len(pairs)
(root/'ranking_evaluation/analysis/kireeti_validation_20261110/batch-consistency.json').write_text(json.dumps(r,indent=2)+'\n')
