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
import argparse
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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Exportar modelos YOLO a TensorRT (.engine). "
            "Modo: 'yolo', 'pose' o 'all'. Los .engine se guardan en ./engines"
        )
    )
    parser.add_argument(
        "mode",
        choices=["yolo", "pose", "all"],
        help="Qué exportar: solo detección (yolo), solo pose (pose) o ambos (all).",
    )
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default="yolo11x.pt",
        help="Pesos YOLO de detección (default: yolo11x.pt).",
    )
    parser.add_argument(
        "--pose-weights",
        type=str,
        default="yolo11x-pose.pt",
        help="Pesos YOLO pose (default: yolo11x-pose.pt).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Tamaño de entrada (lado) para exportar el engine (default: 640).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Dispositivo para la exportación: cuda, cuda:0, cpu... (default: cuda).",
    )
    parser.add_argument(
        "--no-half",
        action="store_true",
        help="Usar FP32 en vez de FP16 (half=False).",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Habilitar batch dinámico en el engine.",
    )

    args = parser.parse_args()
    half = not args.no_half

    if args.mode in ("yolo", "all"):
        det_path = export_yolo_engine(
            args.yolo_weights,
            imgsz=args.imgsz,
            device=args.device,
            half=half,
            dynamic=args.dynamic,
        )
        print(f"[export_engine] Engine YOLO detección generado en: {det_path}")

    if args.mode in ("pose", "all"):
        pose_path = export_yolo_pose_engine(
            args.pose_weights,
            imgsz=args.imgsz,
            device=args.device,
            half=half,
            dynamic=args.dynamic,
        )
        print(f"[export_engine] Engine YOLO pose generado en: {pose_path}")


__all__ = ["export_yolo_engine", "export_yolo_pose_engine"]

if __name__ == "__main__":
    main()

