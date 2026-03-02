#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Control de ejecución exclusiva: evita múltiples instancias del experimento.

Usa execution.lock en RESULTS_DIR. Si existe y el PID está activo, aborta.
Si el lock es antiguo (>2h) o el PID no existe, lo elimina y continúa.
"""

import os
import time
from pathlib import Path


def _get_lock_path() -> Path:
    results_dir = Path(__file__).resolve().parent / "results"
    return results_dir / "execution.lock"


def _pid_exists(pid: int) -> bool:
    """Comprueba si el proceso con PID existe."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def acquire_lock(script_name: str = "experiment") -> bool:
    """
    Intenta adquirir el lock exclusivo de ejecución.

    Returns:
        True si se adquirió; False si hay otra ejecución en curso.
    """
    lock_path = _get_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    if lock_path.exists():
        try:
            content = lock_path.read_text().strip().split()
            if len(content) >= 2:
                pid = int(content[0])
                ts = float(content[1])
                age_hours = (time.time() - ts) / 3600
                if _pid_exists(pid) and age_hours < 2:
                    return False  # Otra ejecución activa
        except (ValueError, IndexError):
            pass
        lock_path.unlink(missing_ok=True)

    lock_path.write_text(f"{os.getpid()} {time.time()} {script_name}")
    return True


def release_lock() -> None:
    """Libera el lock (llamar en finally)."""
    lock_path = _get_lock_path()
    try:
        if lock_path.exists():
            lock_path.unlink()
    except OSError:
        pass
