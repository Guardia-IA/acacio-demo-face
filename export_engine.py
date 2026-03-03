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

from ultralytics import YOLO


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

    Devuelve la ruta al fichero .engine generado.
    """
    weights_path = Path(weights)
    model = YOLO(str(weights_path))
    out = model.export(
        format="engine",
        imgsz=imgsz,
        device=device,
        half=half,
        dynamic=dynamic,
    )
    return Path(out)


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
    model = YOLO(str(weights_path))
    out = model.export(
        format="engine",
        imgsz=imgsz,
        device=device,
        half=half,
        dynamic=dynamic,
    )
    return Path(out)


__all__ = ["export_yolo_engine", "export_yolo_pose_engine"]

