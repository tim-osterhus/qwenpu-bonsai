"""Bounded native-Windows eager circuit parity probe for pinned QwenPU."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent

def emit(**fields):
    print(json.dumps(fields), flush=True)

def measure(args):
    self = psutil.Process()
    cores = self.cpu_affinity()[:2]
    self.cpu_affinity(cores)
    assert self.cpu_affinity() == cores and len(cores) <= 2
    sys.path.insert(0, str(ROOT / "upstream"))
    import torch
    from chipformer.rv import core, elf, fastexec, ref, spec
    from chipformer.rv.asm import assemble

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = args.device
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        emit(phase="device", gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability())
    if args.elf:
        prog = elf.program(args.elf)
    elif args.fixture == "vx":
        prog = assemble("_start: li t1, 123\nli t2, -45\nvput zero, t1, t2\nvget a0, zero, zero\n"
                        "li a7, 93\necall")
    else:
        prog = assemble("_start: li a0, 19\nli a1, 23\nadd a0, a0, a1\nli a7, 93\necall")
    ref_machine = ref.Machine.from_program(prog)
    reference = ref_machine.run(args.steps + 1)[:args.steps + 1]
    vx = args.fixture != "rv"
    start = time.perf_counter()
    circ = core.build(vx=vx)
    core.snapshot(circ, ref.Machine.from_program(prog))
    emit(phase="graph", seconds=time.perf_counter() - start, affinity=cores, **circ.stats())
    start = time.perf_counter()
    ex = fastexec.Executor(circ, device=device, sparse=True)
    ex.load_init()
    if device == "cuda":
        torch.cuda.synchronize()
    emit(phase="setup", seconds=time.perf_counter() - start)
    got = [spec.tok("GO")]
    start = time.perf_counter()
    with torch.inference_mode():
        for want in reference[1:]:
            residual = ex.step(ex.embed(got[-1:])).cpu().numpy()
            token = core.predictions(circ, residual)[0]
            got.append(token)
            if token != want:
                break
    seconds = time.perf_counter() - start
    names = [spec.name(t) if t >= 0 else "INVALID" for t in got]
    expected = [spec.name(t) for t in reference]
    stats = {
        "phase": "result", "fixture": args.fixture, "device": device, "steps": len(got) - 1,
        "exact": got == reference, "seconds": seconds, "trace_tokens_per_second": (len(got) - 1) / seconds,
        "reference": expected, "generated": names, "reference_halted": ref_machine.halted,
        "wrom_bytes": len(prog.wrom),
    }
    if device == "cuda":
        stats.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                     cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                     cuda_free_bytes=torch.cuda.mem_get_info()[0])
    emit(**stats)
    return 0 if stats["exact"] else 1

def supervise(args):
    psutil.cpu_percent()
    baseline = []
    for _ in range(8):
        time.sleep(0.5)
        baseline.append(psutil.cpu_percent())
    emit(phase="baseline", cpu_average=sum(baseline) / len(baseline),
         ram_available=psutil.virtual_memory().available)
    if sum(baseline) / len(baseline) > 80 or psutil.virtual_memory().available < 6 * 1024**3:
        emit(phase="skipped", reason="CPU or RAM headroom")
        return 2
    start = time.perf_counter()
    child_log = ROOT / f"{args.fixture}-{args.device}-child.log"
    with child_log.open("w", encoding="utf-8") as log, subprocess.Popen([sys.executable, __file__, *sys.argv[1:], "--child"],
                          cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW) as p:
        ps = psutil.Process(p.pid)
        tree_peak = system_peak = rss = private = 0
        elapsed = cpu_previous = 0
        cpu_samples = []
        cpu_seen = {}
        reason = None
        while p.poll() is None:
            time.sleep(0.5)
            now = time.perf_counter() - start
            try:
                processes = [ps, *ps.children(recursive=True)]
            except psutil.NoSuchProcess:
                break
            for proc in processes:
                try:
                    cpu_seen[proc.pid] = sum(proc.cpu_times()[:2])
                    mem = proc.memory_info()
                    rss = max(rss, getattr(mem, "peak_wset", mem.rss))
                    private = max(private, getattr(mem, "peak_pagefile", 0))
                except psutil.NoSuchProcess:
                    continue
            cpu = sum(cpu_seen.values())
            pct = 100 * (cpu - cpu_previous) / ((now - elapsed) * psutil.cpu_count())
            cpu_samples.append(psutil.cpu_percent())
            tree_peak = max(pct, tree_peak)
            system_peak = max(cpu_samples[-1], system_peak)
            elapsed, cpu_previous = now, cpu
            if now > 150 or psutil.virtual_memory().available < 4 * 1024**3:
                reason = "wall-time or RAM guard"
                for proc in reversed(processes):
                    try:
                        proc.kill()
                    except psutil.NoSuchProcess:
                        pass
                break
        status = p.wait()
        print(child_log.read_text(encoding="utf-8", errors="replace"), end="", flush=True)
        emit(phase="supervisor", status=status, stop_reason=reason,
             elapsed=time.perf_counter() - start, worker_cpu_average=100 * cpu_previous / (max(elapsed, 0.001) * psutil.cpu_count()),
             worker_cpu_peak=tree_peak, system_cpu_average=sum(cpu_samples)/max(1,len(cpu_samples)),
             system_cpu_peak=system_peak, ram_peak_working_set_bytes=rss, ram_peak_commit_bytes=private)
        return status

if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("fixture", choices=["rv", "vx", "stories"])
    cli.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    cli.add_argument("--steps", type=int, default=48)
    cli.add_argument("--elf", type=Path)
    cli.add_argument("--child", action="store_true")
    args = cli.parse_args()
    assert args.fixture != "stories" or args.elf is not None
    raise SystemExit(measure(args) if args.child else supervise(args))



