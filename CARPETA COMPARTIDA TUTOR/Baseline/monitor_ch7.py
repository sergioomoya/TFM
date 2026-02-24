#!/usr/bin/env python3
"""
Monitorizacion automatica para Chapter 7
Ciclo EMRR - Ejecucion Supervisada
"""

import subprocess
import time
import sys
from datetime import datetime, timedelta
import json

# Configuracion
CONTAINER_NAME = "baseline-ch7-gpu-run-67ab5a70d20f"
TOTAL_CELLS = 484
INTERVAL_MINUTES = 5
PROGRESS_FILE = "/app/execution_progress.txt"
COMPLETE_FILE = "/app/execution_complete.txt"
ERROR_FILE = "/app/execution_error.txt"
LOG_FILE = "monitor_ch7_py.log"

# Estado del monitoreo
state = {
    "start_time": datetime.now(),
    "last_cell": 0,
    "iterations": 0,
    "completed": False,
    "error": False
}

def log(message):
    """Escribe mensaje a consola y archivo"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {message}"
    print(entry)
    sys.stdout.flush()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry + "\n")

def run_docker_command(cmd):
    """Ejecuta comando docker y retorna salida"""
    try:
        full_cmd = f"docker exec {CONTAINER_NAME} {cmd}"
        result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=30)
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception as e:
        return None

def get_progress():
    """Obtiene progreso actual"""
    output = run_docker_command(f"cat {PROGRESS_FILE}")
    if output:
        # Parse "Celda X/484 | HH:MM:SS"
        parts = output.split("|")
        if parts and "Celda" in parts[0]:
            try:
                cell_part = parts[0].strip()
                current = int(cell_part.split("/")[0].replace("Celda", "").strip())
                return current, output
            except:
                return None, output
    return None, output

def get_gpu_status():
    """Obtiene estado de GPU"""
    output = run_docker_command(
        "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader"
    )
    return output

def check_completion():
    """Verifica si hay archivo de completado"""
    result = run_docker_command(f"test -f {COMPLETE_FILE} && echo 'COMPLETE'")
    return result == "COMPLETE"

def check_error():
    """Verifica si hay archivo de error o contenedor caido"""
    # Verificar archivo de error
    result = run_docker_command(f"test -f {ERROR_FILE} && echo 'ERROR'")
    if result == "ERROR":
        return True, "Archivo execution_error.txt detectado"

    # Verificar si contenedor sigue corriendo
    try:
        check = subprocess.run(
            f"docker ps --filter name={CONTAINER_NAME} --format '{{{{.Names}}}}'",
            shell=True, capture_output=True, text=True, timeout=10
        )
        if CONTAINER_NAME not in check.stdout:
            return True, "Contenedor detenido"
    except:
        return True, "No se pudo verificar estado del contenedor"

    return False, None

def get_estimated_time(current_cell):
    """Calcula tiempo restante estimado"""
    if current_cell == 0:
        return "Calculando..."

    elapsed = datetime.now() - state["start_time"]
    cells_remaining = TOTAL_CELLS - current_cell
    avg_time_per_cell = elapsed.total_seconds() / 60 / current_cell  # minutos por celda
    estimated_remaining = avg_time_per_cell * cells_remaining

    hours = int(estimated_remaining // 60)
    minutes = int(estimated_remaining % 60)

    return f"{hours}h {minutes}m"

def format_elapsed():
    """Formatea tiempo transcurrido"""
    elapsed = datetime.now() - state["start_time"]
    hours = int(elapsed.total_seconds() // 3600)
    minutes = int((elapsed.total_seconds() % 3600) // 60)
    return f"{hours}h {minutes}m"

def print_header():
    """Imprime encabezado inicial"""
    log("=" * 60)
    log("INICIANDO MONITORIZACION CHAPTER 7 - Ciclo EMRR")
    log("=" * 60)
    log(f"Contenedor: {CONTAINER_NAME}")
    log(f"Intervalo: {INTERVAL_MINUTES} minutos")
    log(f"Inicio: {state['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Total celdas estimadas: {TOTAL_CELLS}")
    log("=" * 60)
    log("")

def print_check(iteration):
    """Imprime reporte de comprobacion"""
    log("")
    log(f"--- Comprobacion #{iteration} ---")
    log(f"Tiempo transcurrido: {format_elapsed()}")

    # Progreso
    current_cell, raw_progress = get_progress()
    if current_cell:
        state["last_cell"] = current_cell
        percentage = round((current_cell / TOTAL_CELLS) * 100, 1)
        estimated = get_estimated_time(current_cell)
        log(f"[PROGRESO] Celda {current_cell}/{TOTAL_CELLS} ({percentage}%)")
        log(f"           Raw: {raw_progress}")
        log(f"[TIEMPO] Restante estimado: {estimated}")
    else:
        log(f"[!] No se pudo obtener progreso (raw: {raw_progress})")

    # GPU
    gpu = get_gpu_status()
    if gpu:
        parts = [p.strip() for p in gpu.split(",")]
        if len(parts) >= 4:
            log(f"[GPU] Util: {parts[0]} | VRAM: {parts[1]}/{parts[2]} | Temp: {parts[3]}")

            # Alertas
            try:
                util = int(parts[0].replace("%", ""))
                vram_used = int(parts[1].replace("MiB", ""))
                vram_total = int(parts[2].replace("MiB", ""))
                vram_pct = (vram_used / vram_total) * 100

                if util > 95:
                    log("[ALERTA] GPU al maximo - posible cuello de botella")
                if vram_pct > 90:
                    log("[ALERTA] VRAM casi llena - riesgo de OOM")
                if util < 10 and current_cell and current_cell < TOTAL_CELLS:
                    log("[ALERTA] GPU inactiva durante entrenamiento")
            except:
                pass
    else:
        log("[!] No se pudo obtener estado de GPU")

    # Estado contenedor
    error, error_msg = check_error()
    if error:
        state["error"] = True
        log(f"[ERROR DETECTADO] {error_msg}")
        return False

    if check_completion():
        state["completed"] = True
        log("[EXITO] EJECUCION COMPLETADA")
        return False

    remaining = TOTAL_CELLS - (current_cell or 0)
    log(f"[RESUMEN] {remaining} celdas restantes")
    log(f"--- Esperando {INTERVAL_MINUTES} minutos ---")

    return True

def print_final_report():
    """Imprime reporte final"""
    end_time = datetime.now()
    duration = end_time - state["start_time"]
    hours = int(duration.total_seconds() // 3600)
    minutes = int((duration.total_seconds() % 3600) // 60)

    log("")
    log("=" * 60)
    log("MONITORIZACION FINALIZADA - Reporte Final")
    log("=" * 60)
    log(f"Inicio: {state['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Fin: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Duracion total: {hours}h {minutes}m")
    log(f"Comprobaciones realizadas: {state['iterations']}")

    if state["completed"]:
        log("Estado final: COMPLETADO EXITOSAMENTE")
    elif state["error"]:
        log("Estado final: ERROR O INTERRUMPIDO")
    else:
        log("Estado final: MONITOR DETENIDO")

    # Archivos generados
    log("")
    log("Archivos de resultados:")
    try:
        result = subprocess.run(
            f"docker exec {CONTAINER_NAME} ls -la /app/results/ 2>/dev/null || echo 'No files'",
            shell=True, capture_output=True, text=True, timeout=10
        )
        if "No files" not in result.stdout:
            for line in result.stdout.strip().split("\n"):
                log(f"  {line}")
        else:
            log("  (No se encontraron archivos de resultados)")
    except Exception as e:
        log(f"  Error listando archivos: {e}")

    log("=" * 60)

def main():
    """Funcion principal de monitoreo"""
    print_header()

    try:
        while True:
            state["iterations"] += 1
            should_continue = print_check(state["iterations"])

            if not should_continue:
                break

            # Esperar intervalo
            time.sleep(INTERVAL_MINUTES * 60)

    except KeyboardInterrupt:
        log("\n[!] Monitorizacion interrumpida por usuario")
    except Exception as e:
        log(f"\n[ERROR] Error en monitor: {str(e)}")
    finally:
        print_final_report()

if __name__ == "__main__":
    main()
