"""Curvature-instrumented MLP LR-schedule runs (supercollapse@curv).

Replicates the paper's MLP schedule setting (D=1024, N=5, B=4096, lr=1e-3,
T=1e9 fourier tokens) with the hessian/noise diagnostics enabled:
top-10 preconditioned-GGN eigenpairs, per-mode noise powers, Hutchinson
trace, displacement overlaps, and function-space stats.

Launch:  xm launch ~/supercollapse/xm/curv_mlp_schedules.py
"""
from xm_local import Sweep, Job

BASE = (
    "python main.py -cn mlp "
    "wandb_project=super wandb_tag=mlp_schedules_curv wandb_mode=online "
    "model.D=1024 model.N=5 "
    "opt.B=4096 opt.lr=0.001 opt.eps=1e-8 opt.readout_lr_mult=1 "
    "opt.warmup_tokens=0 T=1000000000 T_eval=100000 num_evals=1000 seed=0 "
    "hessian.enabled=true "
)

SCHEDULES = [("const", 1.0), ("linear", 1.0), ("linear", 0.5), ("linear", 0.2)]

s = Sweep("mlp_schedules_curv")
for sched, frac in SCHEDULES:
    s.add(Job(
        cmd=BASE + f"opt.schedule={sched} opt.decay_frac={frac}",
        resource="v6e-1",
        workdir="~/supercollapse",
        restart_on_preemption=True,
    ))
s.launch()
