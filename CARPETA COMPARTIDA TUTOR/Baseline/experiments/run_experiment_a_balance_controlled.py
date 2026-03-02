#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experimento A variantes de balance: ejecución controlada y monitorizada.

Características:
  - Entorno tfm obligatorio (conda)
  - Log detallado con timestamp
  - Monitor de recursos (GPU, RAM) cada 30s en background
  - Archivo de progreso para monitor externo
  - Lock de ejecución exclusiva

Uso:
  .\run_experiment_a_balance_controlled.ps1
  # o
  conda activate tfm && python experiments/run_experiment_a_balance_controlled.py
"""

import os
import sys
import time
import threading
from pathlib import Path
from datetime import datetime

LOG_FILE = None
LOG_LOCK = threading.Lock()
MONITOR_ACTIVE = [True]
_ORIG_STDOUT = None


class TeeWriter:
    """Escribe a log y stdout."""
    def __init__(self, log_file, orig_stdout):
        self.log_file = log_file
        self.orig_stdout = orig_stdout

    def write(self, s):
        if self.log_file:
            with LOG_LOCK:
                self.log_file.write(s)
                self.log_file.flush()
        self.orig_stdout.write(s)
        self.orig_stdout.flush()

    def flush(self):
        if self.log_file:
            self.log_file.flush()
        self.orig_stdout.flush()


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg, also_print=True):
    s = f"[{_ts()}] {msg}"
    with LOG_LOCK:
        if LOG_FILE:
            LOG_FILE.write(s + "\n")
            LOG_FILE.flush()
    if also_print and _ORIG_STDOUT:
        _ORIG_STDOUT.write(s + "\n")
        _ORIG_STDOUT.flush()


def _get_resources():
    """Obtiene uso de GPU y RAM."""
    lines = []
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            lines.append(f"GPU: {r.stdout.strip()}")
    except Exception:
        lines.append("GPU: n/a")

    try:
        if sys.platform == "win32":
            r = subprocess.run(
                ["wmic", "OS", "get", "FreePhysicalMemory,TotalVisibleMemorySize", "/Value"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                free_mb = total_mb = 0
                for line in r.stdout.splitlines():
                    if "=" in line:
                        k, v = line.strip().split("=", 1)
                        if k == "FreePhysicalMemory":
                            free_mb = int(v) / 1024
                        elif k == "TotalVisibleMemorySize":
                            total_mb = int(v) / 1024
                if total_mb > 0:
                    lines.append(f"RAM: {free_mb:.0f}/{total_mb:.0f} MB libres")
        else:
            with open("/proc/meminfo") as f:
                d = dict(line.split() for line in f if ":" in line)
            free = int(d.get("MemAvailable", d.get("MemFree", 0))) // 1024
            total = int(d.get("MemTotal", 0)) // 1024
            lines.append(f"RAM: {free}/{total} MB libres")
    except Exception:
        lines.append("RAM: n/a")
    return " | ".join(lines)


def _monitor_loop(interval_sec=30):
    """Thread: escribe recursos al log cada interval_sec."""
    while MONITOR_ACTIVE[0]:
        _log(f"[MONITOR] {_get_resources()}", also_print=False)
        for _ in range(interval_sec):
            if not MONITOR_ACTIVE[0]:
                break
            time.sleep(1)


def main():
    global LOG_FILE, MONITOR_ACTIVE, _ORIG_STDOUT
    _ORIG_STDOUT = sys.stdout

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = results_dir / f"experiment_a_balance_run_{ts}.log"
    progress_path = results_dir / "experiment_a_balance_progress.txt"

    LOG_FILE = open(log_path, "w", encoding="utf-8")
    _log("=" * 70)
    _log("EXPERIMENTO A — VARIANTES DE BALANCE (ejecución controlada)")
    _log("=" * 70)
    _log(f"Log: {log_path}")
    _log(f"Progreso: {progress_path}")
    sys.stdout = TeeWriter(LOG_FILE, _ORIG_STDOUT)
    _log("Variantes: original, 10:1, 5:1, 1:1")
    _log("=" * 70)

    monitor_thread = threading.Thread(target=_monitor_loop, args=(30,), daemon=True)
    monitor_thread.start()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from experiments.execution_lock import acquire_lock, release_lock

    acquired = False
    if not acquire_lock("experiment_a_balance"):
        _log("ERROR: Otra ejecución en curso. Abortando.")
        sys.exit(1)
    acquired = True
    try:
        from experiments.run_experiment_a_all_balance_variants import main as _main_impl

        _log("Iniciando experimento...")
        t0 = time.time()
        _main_impl(progress_path=progress_path)
        elapsed = time.time() - t0
        _log(f"Experimento completado en {elapsed:.1f}s ({elapsed/60:.1f} min)")
        _log("=" * 70)

    except Exception as e:
        _log(f"ERROR: {e}", also_print=True)
        import traceback
        _log(traceback.format_exc(), also_print=False)
        raise
    finally:
        MONITOR_ACTIVE[0] = False
        if _ORIG_STDOUT:
            sys.stdout = _ORIG_STDOUT
        if LOG_FILE:
            LOG_FILE.close()
        if acquired:
            release_lock()


if __name__ == "__main__":
    main()
