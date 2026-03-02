#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuración de hardware para optimizar la ejecución de experimentos.

Detecta GPU NVIDIA, núcleos CPU para ajustar:
- XGBoost: tree_method='gpu_hist' (1.5.x) o device='cuda' (2.x) cuando hay GPU
- GridSearchCV: n_jobs según núcleos
"""

import os
import subprocess
from typing import Dict, Any


def _gpu_available() -> bool:
    """Comprueba si nvidia-smi funciona (GPU NVIDIA accesible)."""
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() == "-1":
        return False
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5)
        return r.returncode == 0 and "NVIDIA" in (r.stdout or "")
    except Exception:
        try:
            r = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
            return r.returncode == 0
        except Exception:
            return False


def _cpu_count() -> int:
    """Número de núcleos lógicos."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 4


def get_hw_config() -> Dict[str, Any]:
    """Retorna gpu_available, n_jobs, n_cpus."""
    gpu = _gpu_available()
    n = _cpu_count()
    return {
        "gpu_available": gpu,
        "n_cpus": n,
        "n_jobs": -1 if not gpu else max(1, n - 1),
    }


def get_xgboost_gpu_params() -> Dict[str, Any]:
    """
    Parámetros para XGBoost según GPU.
    XGBoost 1.5.x: tree_method='gpu_hist', gpu_id=0
    XGBoost 2.x:   device='cuda', tree_method='hist'
    Con XGBOOST_USE_CPU=1: fuerza CPU (evita NCCL en Docker/GridSearchCV).
    """
    if os.environ.get("XGBOOST_USE_CPU", "").strip() == "1":
        return {}
    cfg = get_hw_config()
    if not cfg["gpu_available"]:
        return {}
    try:
        import xgboost as xgb
        v = getattr(xgb, "__version__", "0.0")
        major = int(v.split(".")[0]) if v else 0
        if major >= 2:
            return {"device": "cuda", "tree_method": "hist"}
        return {"tree_method": "gpu_hist", "gpu_id": 0}
    except Exception:
        return {}
