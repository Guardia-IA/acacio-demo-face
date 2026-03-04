#!/usr/bin/env python3
"""
Utilidades para exportar modelos YOLO a TensorRT (.engine).

Separamos dos funciones para que puedas enchufar más lógica después:
- export_yolo_engine: para modelos YOLO "normales" (detección).
- export_yolo_pose_engine: para modelos YOLO pose.

Ambas usan la API oficial de Ultralytics:
    YOLO(weights).export(format="engine", ...)
"""

from pathlib import Path
from typing import Union
import shutil

from ultralytics import YOLO

try:
    import torch

    _HAS_CUDA = torch.cuda.is_available()
except Exception:
    _HAS_CUDA = False


PathLike = Union[str, Path]


def export_yolo_engine(
    weights: PathLike,
    imgsz: int = 640,
    device: str = "cuda",
    half: bool = True,
    dynamic: bool = False,
) -> Path:
    """
    Exporta un modelo YOLO de detección a TensorRT (.engine).

    - weights: ruta al .pt (yolo11n.pt, yolo11x.pt, etc.).
    - imgsz: tamaño de entrada (un solo entero → cuadrado).
    - device: "cuda", "cpu" o "cuda:0", etc.
    - half: FP16 si es True (recomendado en GPU moderneas).
    - dynamic: batch dinámico si True (más flexible, a veces menos óptimo).

    Devuelve la ruta al fichero .engine generado dentro de la carpeta ./engines.
    """
    weights_path = Path(weights)
    engines_dir = Path(__file__).resolve().parent / "engines"
    engines_dir.mkdir(parents=True, exist_ok=True)

    # Asegurarnos de que realmente hay CUDA si se pide "cuda"
    if device.startswith("cuda") and not _HAS_CUDA:
        print("[export_yolo_engine] CUDA no disponible; usando CPU para la exportación.")
        device = "cpu"
    else:
        print(f"[export_yolo_engine] Exportando con dispositivo '{device}'. CUDA disponible={_HAS_CUDA}")

    model = YOLO(str(weights_path))
    out = model.export(
        format="engine",
        imgsz=imgsz,
        device=device,
        half=half,
        dynamic=dynamic,
    )
    out_path = Path(out)
    final_path = engines_dir / f"{weights_path.stem}.engine"
    try:
        shutil.move(str(out_path), str(final_path))
    except Exception:
        # Si no podemos mover, al menos devolvemos la ruta original
        final_path = out_path
    return final_path


def export_yolo_pose_engine(
    weights: PathLike,
    imgsz: int = 640,
    device: str = "cuda",
    half: bool = True,
    dynamic: bool = False,
) -> Path:
    """
    Exporta un modelo YOLO de pose a TensorRT (.engine).

    La API es la misma que para detección; se separa en otra función
    solo para que puedas meter lógica específica de pose más adelante.
    """
    weights_path = Path(weights)
    engines_dir = Path(__file__).resolve().parent / "engines"
    engines_dir.mkdir(parents=True, exist_ok=True)

    if device.startswith("cuda") and not _HAS_CUDA:
        print("[export_yolo_pose_engine] CUDA no disponible; usando CPU para la exportación.")
        device = "cpu"
    else:
        print(f"[export_yolo_pose_engine] Exportando con dispositivo '{device}'. CUDA disponible={_HAS_CUDA}")

    model = YOLO(str(weights_path))
    out = model.export(
        format="engine",
        imgsz=imgsz,
        device=device,
        half=half,
        dynamic=dynamic,
    )
    out_path = Path(out)
    final_path = engines_dir / f"{weights_path.stem}.engine"
    try:
        shutil.move(str(out_path), str(final_path))
    except Exception:
        final_path = out_path
    return final_path


__all__ = ["export_yolo_engine", "export_yolo_pose_engine"]

