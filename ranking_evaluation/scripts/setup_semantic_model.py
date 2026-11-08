"""Explicit fixed-revision semantic-model setup with actual file hashes."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.model_dir.exists():parser.error('Destination exists; choose a separate directory')
    config=json.loads(args.config.read_text())
    from huggingface_hub import snapshot_download
    snapshot_download(config['model_id'],revision=config['model_revision'],local_dir=args.model_dir,
        allow_patterns=['config.json','modules.json','model.safetensors','pytorch_model.bin',
                        'sentence_bert_config.json','config_sentence_transformers.json',
                        'tokenizer.json','tokenizer_config.json','special_tokens_map.json',
                        'vocab.txt','1_Pooling/config.json','README.md'])
    if not any((args.model_dir/name).is_file() for name in ('model.safetensors','pytorch_model.bin')):
        raise ValueError('Pinned checkpoint has no supported weight file')
    files={}
    for path in sorted(args.model_dir.rglob('*')):
        if path.is_file() and '.cache' not in path.relative_to(args.model_dir).parts:
            with path.open('rb') as handle:files[path.relative_to(args.model_dir).as_posix()]=hashlib.file_digest(handle,'sha256').hexdigest()
    manifest={'model_id':config['model_id'],'revision':config['model_revision'],'files_sha256':files}
    (args.model_dir/'download_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(args.model_dir)


if __name__=='__main__':main()
