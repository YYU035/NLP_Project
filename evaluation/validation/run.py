"""One-command software, dataset and real-model validation; retain completed frozen studies."""
import argparse,json,os,subprocess,sys
from evaluation.validation.freeze import ROOT, OUT

def command(module,*args,log=None):
    result=subprocess.run([sys.executable,'-m',module,*args],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    if log:(OUT/log).write_text(result.stdout+result.stderr,encoding='utf-8')
    print(result.stdout[-1600:],flush=True)
    if result.returncode:raise RuntimeError(f'{module} failed ({result.returncode}); see {log}')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--clean-setup',action='store_true');args=ap.parse_args()
    os.environ['PYTHONUTF8']='1';OUT.mkdir(parents=True,exist_ok=True)
    command('evaluation.xai_data',log='scifact_staging.log')
    command('evaluation.validation.freeze',log='freeze.log')
    command('evaluation.validation.stage_data',log='dataset_staging.log')
    command('pytest',str(ROOT/'tests'),'-q','-o',f'cache_dir={ROOT}/.pytest_cache','--basetemp',str(ROOT/'.cache/tmp/validation_final_pytest'),log='final_tests.log')
    command('pip','check',log='pip_check.log')
    jobs=[('other_datasets.json','other_datasets'),('scifact_full/summary.json','scifact'),('stress.json','stress'),('reliability.json','reliability_checks')]
    for artifact,module in jobs:
        if (OUT/artifact).is_file():print('Retaining completed frozen result:',artifact,flush=True)
        else:command('evaluation.validation.'+module,log=module+'.log')
    if args.clean_setup or not (OUT/'clean_setup.json').exists():command('evaluation.validation.clean_setup',log='clean_setup_console.log')
    clean=json.loads((OUT/'clean_setup.json').read_text(encoding='utf-8'))
    for key in ['dependency_install_exit_code','pip_check_exit_code','tests_exit_code','offline_model_smoke_exit_code']:
        if clean.get(key)!=0:raise RuntimeError('Clean setup did not pass: '+key)
    # Live providers may be rate-limited: preserve failures in report rather than a false scientific verdict.
    result=subprocess.run([sys.executable,'-m','research_verifier.runtime.smoke','--live','--limit','3'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    (OUT/'live_smoke.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    print('Fresh live-provider smoke exit:',result.returncode,flush=True)
    command('evaluation.validation.audit_artifacts',log='audit.log')
    command('evaluation.validation.report',log='report.log')
    print('Review outputs/validation/published/FINAL_TEST_REPORT.md and outputs/validation/review-kit.zip. Model errors remain measured errors; software completion is not scientific approval.',flush=True)
if __name__=='__main__':main()
