"""Render a local review form; all human judgements start blank."""
import csv
import html
import json
from pathlib import Path


def render(directory):
    directory=Path(directory)
    with (directory/'human-review.csv').open(newline='') as h:rows=list(csv.DictReader(h))
    fields=['source_matches','relevant_to_claim','context_sufficient','within_input_limit']
    parts=['<!doctype html><meta charset="utf-8"><title>PDF evidence human review</title>',
        '<style>body{font:16px system-ui;max-width:1080px;margin:36px auto;padding:16px}article{border-top:2px solid #aaa;margin:36px 0;padding-top:20px}pre{white-space:pre-wrap;background:#f4f6f9;padding:18px}img{width:100%;border:1px solid #ddd}label{display:block;margin:10px}input,select,textarea,button{font:inherit;padding:7px}strong{color:#a32}textarea{width:95%}</style>',
        '<h1>PDF evidence: separate manual review sample</h1>',
        '<p><strong>Human review is pending.</strong> Automatic source/token checks are not relevance or scientific-quality labels.</p>',
        '<p>Compare each passage with its original PDF page. Mark yes, no or uncertain for every check. A negative rating is valid and must not be changed just to pass.</p>',
        '<label>Your reviewer name <input id="reviewer" autocomplete="name"></label>',
        '<button onclick="downloadReview()">Download completed review CSV</button>',
        '<p>The file is downloaded locally. No ratings are sent automatically and no reviewer is pre-filled.</p>']
    for index,row in enumerate(rows):
        parts += ['<article><h2>'+html.escape(row['sample_id'])+'</h2><p>'+html.escape(row['claim'])+'</p>',
            '<p>'+html.escape(row['paper_id'])+'; PDF page '+row['page']+'; pair tokens '+row['pair_tokens']+'/512.</p>',
            '<a href="'+html.escape(row['pdf'])+'#page='+row['page']+'" target="_blank">Original PDF</a>',
            '<pre>'+html.escape(row['passage'])+'</pre>']
        for field in fields:
            parts.append('<label>'+field.replace('_',' ')+' <select id="'+field+'-'+str(index)+'"><option value="">Choose…</option><option>yes</option><option>no</option><option>uncertain</option></select></label>')
        parts += ['<label>Notes<textarea id="notes-'+str(index)+'" rows="2"></textarea></label>',
            '<img loading="lazy" src="'+html.escape(row['image'])+'" alt="Original PDF page"></article>']
    data=json.dumps(rows,ensure_ascii=False).replace('<',chr(92)+'u003c')
    parts += ['<script>const reviewRows='+data+';const reviewFields='+json.dumps(fields)+';',r'''
function downloadReview(){
 const reviewer=document.getElementById('reviewer').value.trim();
 if(!reviewer){alert('Enter your reviewer name.');return;}
 const completed=reviewRows.map((row,i)=>{
   const result={...row,reviewer,reviewed_at:new Date().toISOString()};
   for(const field of reviewFields){result[field]=document.getElementById(field+'-'+i).value;}
   result.notes=document.getElementById('notes-'+i).value;
   return result;
 });
 if(completed.some(row=>reviewFields.some(field=>!row[field]))){alert('Complete each rating; uncertain is allowed.');return;}
 const columns=Object.keys(completed[0]);
 const quote=value=> '"'+String(value??'').replaceAll('"','""')+'"';
 const csv=[columns.map(quote).join(','),...completed.map(row=>columns.map(k=>quote(row[k])).join(','))].join(String.fromCharCode(13,10));
 const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
 const link=document.createElement('a');link.href=url;link.download='human-review-completed.csv';link.click();
 setTimeout(()=>URL.revokeObjectURL(url),1000);
}
</script>''']
    (directory/'index.html').write_text('\n'.join(parts))
    print(directory/'index.html')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    render(p.parse_args().directory)
