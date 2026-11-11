"""Recreate a separate environment; do not reuse the main venv or model junctions."""
import json,os,subprocess,sys,time,venv
from datetime import datetime
from pathlib import Path
from .freeze import ROOT,OUT

def main():
    env_root=(ROOT/('.tmp/clean_validation_'+datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    if not env_root.is_relative_to(ROOT.resolve()):raise ValueError('Invalid environment root')
    started=time.perf_counter();venv.EnvBuilder(with_pip=True).create(env_root)
    python=env_root/'Scripts/python.exe'
    env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',PIP_CACHE_DIR=str(ROOT/'.pip-cache'),TEMP=str(ROOT/'.tmp'),TMP=str(ROOT/'.tmp'),
        HF_HOME=str(ROOT/'.tmp/clean_validation_hf'),TORCH_HOME=str(ROOT/'.tmp/clean_validation_torch'),MPLCONFIGDIR=str(ROOT/'.model-cache/matplotlib'))
    count=int(env.get('GIT_CONFIG_COUNT','0'));env.update({f'GIT_CONFIG_KEY_{count}':'core.longpaths',f'GIT_CONFIG_VALUE_{count}':'true','GIT_CONFIG_COUNT':str(count+1)})
    command=[str(python),'-m','pip','install','--disable-pip-version-check','--no-input','--retries','1','--timeout','30','-q',
        '-r',str(ROOT/'requirements-discovery-lock.txt'),'-r',str(ROOT/'requirements-attribution.txt')]
    print('Installing exact requirements into',env_root,flush=True)
    result=subprocess.run(command,cwd=ROOT.parent,env=env,capture_output=True,text=True,encoding='utf-8',timeout=1200)
    (OUT/'clean_install.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    report={'kind':'Independent Python environment, no system-site-packages and no venv junction reuse','dependency_install_exit_code':result.returncode,
        'models':'Existing pinned local model files reused by explicit owned paths; no claim of a second full model download or another operating system.',
        'environment_path':str(env_root),'seconds':time.perf_counter()-started}
    if result.returncode==0:
        check=subprocess.run([str(python),'-m','pip','check'],env=env,capture_output=True,text=True,encoding='utf-8');report['pip_check_exit_code']=check.returncode
        env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        tests=subprocess.run([str(python),'-m','pytest',str(ROOT/'tests'),'-q','-o',f'cache_dir={ROOT}/.pytest_cache','--basetemp',str(ROOT/'.tmp/clean_validation_pytest')],cwd=ROOT.parent,env=env,capture_output=True,text=True,encoding='utf-8',timeout=600)
        (OUT/'clean_tests.log').write_text(tests.stdout+tests.stderr,encoding='utf-8');report['tests_exit_code']=tests.returncode;report['test_summary']=tests.stdout[-3000:]
        smoke=subprocess.run([str(python),'-m','explanation_frontend.pipeline_smoke'],cwd=ROOT.parent,env=env,capture_output=True,text=True,encoding='utf-8',timeout=300)
        (OUT/'clean_model_smoke.log').write_text(smoke.stdout+smoke.stderr,encoding='utf-8');report['offline_model_smoke_exit_code']=smoke.returncode
    report['seconds']=time.perf_counter()-started
    (OUT/'clean_setup.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    return report
if __name__=='__main__':main()
