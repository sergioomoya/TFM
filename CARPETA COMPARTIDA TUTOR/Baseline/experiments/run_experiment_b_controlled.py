#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experimento B: Ejecución controlada, monitorizada y optimizada.

OBJETIVO TFM: Mejorar AUPRC y CP@100 sobre el baseline (Experimento A).
  Baseline A: XGBoost AUPRC=0.690, CP100=0.296; RF CP100=0.297
  Meta: superar o igualar estas métricas con cost-sensitive/optimizaciones.

Características:
  - Log detallado (experiments/results/experiment_b_run_TIMESTAMP.log)
  - Monitor de recursos (GPU, CPU, RAM) cada 30s
  - Comparativa automática vs baseline
  - Optimizaciones: selección por CP@100, grid ampliado, threshold óptimo

Uso: conda activate tfm && python experiments/run_experiment_b_controlled.py
"""

import os
import sys
import time
import threading
import pickle
from pathlib import Path
from datetime import datetime

# Redirigir stdout a log + consola
LOG_FILE = None
LOG_LOCK = threading.Lock()
MONITOR_ACTIVE = [True]  # lista para mutabilidad en closure
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
    """Obtiene uso de GPU, RAM. Multiplataforma."""
    lines = []
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
                          capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            lines.append(f"GPU: {r.stdout.strip()}")
    except Exception:
        lines.append("GPU: n/a")
    try:
        if sys.platform == "win32":
            import subprocess
            free_mb = total_mb = 0
            r = subprocess.run(["wmic", "OS", "get", "FreePhysicalMemory,TotalVisibleMemorySize", "/Value"],
                               capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                for line in r.stdout.splitlines():
                    if "=" in line:
                        k, v = line.strip().split("=", 1)
                        if k == "FreePhysicalMemory":
                            free_mb = int(v) / 1024
                        elif k == "TotalVisibleMemorySize":
                            total_mb = int(v) / 1024
                if total_mb > 0:
                    lines.append(f"RAM: {free_mb:.0f} / {total_mb:.0f} MB libres")
        else:
            with open("/proc/meminfo") as f:
                d = dict(line.split() for line in f if ":" in line)
            free = int(d.get("MemAvailable", d.get("MemFree", 0))) // 1024
            total = int(d.get("MemTotal", 0)) // 1024
            lines.append(f"RAM: {free} / {total} MB libres")
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
    _ORIG_STDOUT = sys.stdout  # guardar antes de redirigir

    # Crear archivo de log
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = results_dir / f"experiment_b_run_{ts}.log"
    LOG_FILE = open(log_path, "w", encoding="utf-8")

    _log("=" * 70)
    _log("EXPERIMENTO B — EJECUCIÓN CONTROLADA Y MONITORIZADA")
    _log("=" * 70)
    _log(f"Log: {log_path}")
    sys.stdout = TeeWriter(LOG_FILE, _ORIG_STDOUT)
    _log(f"Objetivo: Mejorar AUPRC y CP@100 sobre baseline (Exp A)")
    _log(f"  Baseline A: XGBoost AUPRC=0.690 CP100=0.296 | RF CP100=0.297")
    _log("=" * 70)

    monitor_thread = threading.Thread(target=_monitor_loop, args=(30,), daemon=True)
    monitor_thread.start()

    results_b = None
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from experiments.run_experiment_b_standalone import _main_impl
        from experiments.execution_lock import acquire_lock, release_lock

        if not acquire_lock("experiment_b_controlled"):
            _log("ERROR: Otra ejecución en curso. Abortando.")
            sys.exit(1)

        _log("Iniciando Experimento B standalone...")
        t0 = time.time()
        results_b = _main_impl()
        elapsed = time.time() - t0
        _log(f"Experimento B completado en {elapsed:.1f}s")

        _log("")
        _log("=" * 70)
        _log("COMPARATIVA vs BASELINE (Experimento A)")
        _log("=" * 70)
        baseline_a = {
            "Logistic Regression": {"auprc": 0.6350, "cp100": 0.2929},
            "Random Forest": {"auprc": 0.6846, "cp100": 0.2971},
            "XGBoost": {"auprc": 0.6904, "cp100": 0.2961},
        }
        for variant_key, res in results_b.items():
            model_name = variant_key.split("_", 1)[1] if "_" in variant_key else variant_key
            ba = baseline_a.get(model_name, {})
            if not ba:
                for k in baseline_a:
                    if k in variant_key:
                        ba = baseline_a[k]
                        break
            auprc_d = res["auprc_mean"] - ba.get("auprc", 0)
            cp_d = res["cp100_mean"] - ba.get("cp100", 0)
            auprc_ok = "SUPERA" if auprc_d > 0 else ("IGUAL" if auprc_d == 0 else "EMPEORA")
            cp_ok = "SUPERA" if cp_d > 0 else ("IGUAL" if cp_d == 0 else "EMPEORA")
            _log(f"{variant_key}:")
            _log(f"  AUPRC {res['auprc_mean']:.4f} (A={ba.get('auprc',0):.4f}) delta={auprc_d:+.4f} [{auprc_ok}]")
            _log(f"  CP@100 {res['cp100_mean']:.4f} (A={ba.get('cp100',0):.4f}) delta={cp_d:+.4f} [{cp_ok}]")
            best = res.get("best_params", {})
            if best:
                relevant = {k.replace('clf__', ''): v for k, v in best.items()
                            if k.startswith('clf__') and k not in (
                                'clf__random_state', 'clf__use_label_encoder',
                                'clf__eval_metric', 'clf__n_jobs', 'clf__verbosity',
                                'clf__max_iter',
                            )}
                _log(f"  best_params: {relevant}")
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
        release_lock()

    return results_b


if __name__ == "__main__":
    main()
