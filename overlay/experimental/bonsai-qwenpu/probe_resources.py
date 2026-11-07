"""Bound and sample Windows native probe runs."""

import ctypes
from ctypes import wintypes
import os
import subprocess
import threading
import time

import psutil

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class ProbePreflightError(RuntimeError):
    """Available resources are below the threshold for starting a probe."""


def cap_processes():
    """Windows children inherit the affinity; the job reaps them if this worker exits."""
    if os.name != "nt":
        raise RuntimeError("The native probes require Windows.")
    process = psutil.Process()
    logical = psutil.cpu_count()
    count = min(2, int(logical * 0.15))
    if not count:
        raise RuntimeError("Too few logical processors to enforce the CPU budget.")
    cores = process.cpu_affinity()[:count]
    process.cpu_affinity(cores)
    if not cores or process.cpu_affinity() != cores:
        raise RuntimeError("Windows did not apply the worker CPU limit.")

    class BasicLimits(ctypes.Structure):
        _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                    ("flags", wintypes.DWORD), ("min_ws", ctypes.c_size_t),
                    ("max_ws", ctypes.c_size_t), ("processes", wintypes.DWORD),
                    ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
                    ("scheduling", wintypes.DWORD)]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [("basic", BasicLimits), ("io", ctypes.c_uint64 * 6),
                    ("memory", ctypes.c_size_t * 4)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
    ]
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    limits = ExtendedLimits()
    limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        error = ctypes.WinError(ctypes.get_last_error())
        kernel.CloseHandle(job)
        raise error
    if not kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess()):
        error = ctypes.WinError(ctypes.get_last_error())
        kernel.CloseHandle(job)
        raise error
    # Windows closes the job handle when this probe supervisor exits.
    return process, cores, logical


def gpu_memory():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
        creationflags=NO_WINDOW, text=True, timeout=4,
    )
    return tuple(map(int, output.splitlines()[0].split(",")))


class ResourceSampler:
    def __init__(self, process, logical, baseline):
        self.process = process
        self.logical = logical
        self.baseline = baseline
        self.stage = "circuit setup"
        self.stop = threading.Event()
        self.results = {}
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()

    def sample(self):
        previous = {}
        previous_time = time.monotonic()
        psutil.cpu_percent()
        while not self.stop.wait(0.65):
            now = time.monotonic()
            stage = self.stage
            rows = self.results.setdefault(stage, [])
            rss = private = cpu = 0
            for proc in [self.process, *self.process.children(recursive=True)]:
                try:
                    with proc.oneshot():
                        times = proc.cpu_times()
                        memory = proc.memory_info()
                except psutil.NoSuchProcess:
                    continue
                seconds = times.user + times.system
                cpu += max(0, seconds - previous.get(proc.pid, 0))
                previous[proc.pid] = seconds
                rss += memory.rss
                private += memory.private
            try:
                gpu_used, _ = gpu_memory()
            except (OSError, subprocess.SubprocessError, ValueError):
                gpu_used = None
            rows.append({
                "tree_cpu_pct": 100 * cpu / ((now - previous_time) * self.logical),
                "system_cpu_pct": psutil.cpu_percent(),
                "working_set_bytes": rss,
                "private_commit_bytes": private,
                "system_ram_used_bytes": psutil.virtual_memory().used,
                "gpu_mib": gpu_used,
            })
            previous_time = now

    def summary(self):
        self.stop.set()
        self.thread.join()
        return {
            "baseline": self.baseline,
            "stages": {
                stage: {key: max(values) for key in rows[0]
                        if (values := [row[key] for row in rows if row[key] is not None])}
                for stage, rows in self.results.items() if rows
            },
        }
