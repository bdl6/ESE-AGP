"""Day03固定实验执行器：两组独立对照，训练全部成功后才统一评估测试集。"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[2]
NAMES=['day03_local_id','day03_global_id','day03_attrs_legacy','day03_attrs_matched']
LOCK=threading.Lock()
STATE={'stage':'training','test_evaluated':False,'jobs':{},'started_unix':time.time()}


def update():
    """调用方持锁，原子更新状态，避免读到半个JSON。"""
    STATE['updated_unix']=time.time()
    path=ROOT/'runs/day03_jobs_status.json'; temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(STATE,ensure_ascii=False,indent=2)); temporary.replace(path)


def verify_snapshot():
    """训练和评估均拒绝静默改变源码/配置；防止异步实验混用版本。"""
    protocol=json.loads((ROOT/'runs/day03_refactored_protocol.json').read_text())
    for name,expected in protocol['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Frozen source/config changed: '+name)


def command(args,gpu,logfile):
    """以同一venv启动命令并落盘stdout/stderr；不使用shell拼接命令。"""
    verify_snapshot()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    with logfile.open('w') as out:
        child=subprocess.Popen([sys.executable,'-u','-m','ese_agp',*args],cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT)
        with LOCK:
            STATE['active_pid_'+str(gpu)]=child.pid;update()
        return child.wait()


def train_sequence(names,gpu):
    """同GPU顺序运行，另一组独立GPU并行；失败即停止该序列。"""
    for name in names:
        run=ROOT/'runs'/name
        if run.exists(): raise FileExistsError('Refuse overwriting run '+str(run))
        with LOCK:
            STATE['jobs'][name]={'stage':'training','gpu':gpu,'started_unix':time.time()};update()
        code=command(['train','--config','configs/'+name+'.yaml'],gpu,ROOT/'runs'/(name+'.log'))
        with LOCK:
            STATE['jobs'][name].update(stage='trained' if code==0 else 'failed',exit_code=code,ended_unix=time.time());update()
        if code: return False
    return True


def finish_report():
    """只读取已生成机器结果，生成完整指标表，不手填或挑选最好数字。"""
    lines=['# Day03 正式实验结果','', '同一论文随机9:1数据，seed=42，各固定20轮，保存最后一轮。单种子结果不代表统计显著。', '', '|模型|HR@20|MRR@20|训练秒数|峰值分配显存MiB|','|---|---:|---:|---:|---:|']
    for name in NAMES:
        run=ROOT/'runs'/name
        m=json.loads((run/'evaluation_test.json').read_text());r=json.loads((run/'report.json').read_text())
        lines.append(f"|{name}|{m['hr@20']:.6f}|{m['mrr@20']:.6f}|{r['elapsed_seconds']:.1f}|{r['peak_cuda_bytes']/2**20:.1f}|")
    lines+=['','对照一仅比较 local_id/global_id；对照二仅比较 attrs_legacy/attrs_matched。不能跨数据协议比较旧数值。','所有数字来自 runs 对应目录；完整图模型、实体语义、剪枝、多视角仍未完成。']
    (ROOT/'docs/DAY3_RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    """启动前检查不可覆盖，完成全部训练后统一进行一次最终测试评估。"""
    os.chdir(ROOT)
    guard=ROOT/'runs/day03_runner.lock'
    fd=os.open(guard,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        for name in NAMES:
            if (ROOT/'runs'/name).exists(): raise FileExistsError(name)
        with LOCK: update()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            jobs=[pool.submit(train_sequence,['day03_global_id'],5),pool.submit(train_sequence,['day03_local_id','day03_attrs_legacy','day03_attrs_matched'],4)]
            ok=all([job.result() for job in jobs])
        if not ok: raise RuntimeError('Training failed; test evaluation skipped')
        with LOCK: STATE['stage']='final_test_evaluation';update()
        for name in NAMES:
            code=command(['evaluate','--config','configs/'+name+'.yaml','--split','test'],5,ROOT/'runs'/(name+'_evaluate.log'))
            if code: raise RuntimeError('Evaluation failed: '+name)
            with LOCK: STATE['jobs'][name]['stage']='evaluated';update()
        code=command(['baseline','--config','configs/day03_local_id.yaml','--split','test'],5,ROOT/'runs/day03_baselines.log')
        if code: raise RuntimeError('Baseline failed')
        finish_report()
        with LOCK: STATE.update(stage='complete',test_evaluated=True);update()
    except Exception as exc:
        with LOCK: STATE.update(stage='failed',error=str(exc));update()
        raise
    finally:
        guard.unlink(missing_ok=True)


if __name__=='__main__': main()
