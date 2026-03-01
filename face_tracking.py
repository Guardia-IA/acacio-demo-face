#!/usr/bin/env python3
"""
Captura de puntos faciales y parametrización de una cara desde un vídeo.
Usa InsightFace (ArcFace) para embeddings 512-d de alta calidad en identificación.
- Recibe: path del vídeo, id (entero), nombre (string).
- Recoge al menos 10 frames de vista frontal, 10 de lateral izquierdo y 10 de lateral derecho
  para una identificación más robusta (el otro lateral y mirar arriba/abajo son opcionales pero recomendables).
- Muestra en ventana Tkinter el vídeo con la cara (bbox + puntos clave) y barra de progreso por pose.
- Al terminar guarda la parametrización en face_tracking/<id>_<nombre>.pkl.
"""

import argparse
import pickle
import sys
import threading
from pathlib import Path
from queue import Empty, Queue

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk

# InsightFace (ArcFace): carga perezosa para no ralentizar el arranque
_app_insightface = None

def get_face_app():
    """Inicializa InsightFace FaceAnalysis una sola vez (descarga modelos en primer uso)."""
    global _app_insightface
    if _app_insightface is None:
        from insightface.app import FaceAnalysis
        # buffalo_l: buen equilibrio; antelopev2 es más nuevo. ctx_id=-1 = CPU.
        _app_insightface = FaceAnalysis(name="buffalo_l", root=str(Path.home() / ".insightface"))
        _app_insightface.prepare(ctx_id=-1, det_thresh=0.5, det_size=(320, 320))
    return _app_insightface

# Número de frames con cara válida que queremos recoger para la parametrización
NUM_FRAMES_META = 30
# Mínimo de frames por tipo de pose (frontal, lateral izq., lateral der.) para mejor identificación
MIN_FRAMES_POR_POSE = 10
# Cada cuántos frames extraemos embedding (para no saturar)
CADA_N_FRAMES = 2

# Índices de los 5 keypoints InsightFace: [ojo_izq, ojo_der, nariz, boca_izq, boca_der]
KPS_OJO_IZQ, KPS_OJO_DER, KPS_NARIZ = 0, 1, 2
# Umbral (fracción del ancho de la cara) para considerar lateral vs frontal
YAW_THRESHOLD = 0.12

DIR_FACE_TRACKING = Path(__file__).resolve().parent / "face_tracking"


def frame_bgr_a_photoimage(frame_bgr, max_ancho=960, max_alto=720):
    """Convierte frame BGR (OpenCV) a ImageTk.PhotoImage para Tkinter."""
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    h, w = frame_bgr.shape[:2]
    if w > max_ancho or h > max_alto:
        escala = min(max_ancho / w, max_alto / h)
        nw, nh = int(w * escala), int(h * escala)
        frame_bgr = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(image=Image.fromarray(rgb))


def dibujar_cara_insightface(frame_bgr, face):
    """
    Dibuja bbox y los 5 puntos clave (ojos, nariz, comisuras) de InsightFace sobre el frame BGR.
    face: objeto retornado por app.get() con .bbox y .kps.
    """
    if face is None:
        return
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2, lineType=cv2.LINE_AA)
    if hasattr(face, "kps") and face.kps is not None:
        for i in range(face.kps.shape[0]):
            x, y = int(face.kps[i][0]), int(face.kps[i][1])
            cv2.circle(frame_bgr, (x, y), 4, (0, 255, 255), -1, lineType=cv2.LINE_AA)


def clasificar_pose(face):
    """
    Clasifica la pose de la cara en 'frontal', 'izq' o 'der' usando la posición de la nariz
    respecto al centro de los ojos (yaw aproximado). Así aseguramos variedad en el vídeo.
    """
    if not hasattr(face, "kps") or face.kps is None or face.kps.shape[0] < 3:
        return "frontal"
    kps = face.kps
    centro_ojos_x = (kps[KPS_OJO_IZQ][0] + kps[KPS_OJO_DER][0]) / 2
    nariz_x = kps[KPS_NARIZ][0]
    bbox = face.bbox
    ancho = bbox[2] - bbox[0]
    margen = ancho * YAW_THRESHOLD
    if nariz_x < centro_ojos_x - margen:
        return "izq"   # cara girada hacia nuestra izquierda (vemos su lateral derecho)
    if nariz_x > centro_ojos_x + margen:
        return "der"   # cara girada hacia nuestra derecha (vemos su lateral izquierdo)
    return "frontal"


def run_face_tracking(path_video: Path, id_persona: int, nombre: str) -> bool:
    """
    Abre el vídeo, muestra ventana Tkinter con cara (bbox + puntos) y barra de progreso,
    recoge embeddings ArcFace (512-d) hasta tener NUM_FRAMES_META, y guarda en face_tracking/<id>_<nombre>.pkl.
    Devuelve True si se guardó correctamente.
    """
    path_video = Path(path_video)
    if not path_video.is_file():
        print(f"Error: no existe el vídeo '{path_video}'", file=sys.stderr)
        return False

    cap = cv2.VideoCapture(str(path_video))
    if not cap.isOpened():
        print(f"Error: no se pudo abrir el vídeo '{path_video}'", file=sys.stderr)
        return False

    nombre_safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in nombre).strip() or "usuario"
    nombre_safe = nombre_safe.replace(" ", "_")
    DIR_FACE_TRACKING.mkdir(parents=True, exist_ok=True)
    path_out = DIR_FACE_TRACKING / f"{id_persona}_{nombre_safe}.pkl"

    encodings_recogidos = []
    pose_buckets = {"frontal": [], "izq": [], "der": []}
    frame_queue = Queue(maxsize=2)
    playing = [True]

    root = tk.Tk()
    root.title(f"Face tracking (ArcFace) — {path_video.name} — ID {id_persona} ({nombre})")
    root.geometry("1000x720")
    root.configure(bg="#2b2b2b")

    main = tk.Frame(root, bg="#2b2b2b")
    main.pack(fill="both", expand=True, padx=8, pady=8)

    lbl_info = tk.Label(
        main,
        text=f"Recogiendo cara (ArcFace) para ID {id_persona} — {nombre}",
        font=("Segoe UI", 11),
        fg="white",
        bg="#2b2b2b",
    )
    lbl_info.pack(pady=(0, 2))
    lbl_hint = tk.Label(
        main,
        text="Incluye en el vídeo: vista frontal, lateral izquierdo y lateral derecho (y opcionalmente mirar arriba/abajo).",
        font=("Segoe UI", 9),
        fg="#888",
        bg="#2b2b2b",
        wraplength=700,
    )
    lbl_hint.pack(pady=(0, 4))

    lbl_frame = tk.Label(main, text="Frame: 0", font=("Segoe UI", 10), fg="#aaa", bg="#2b2b2b")
    lbl_frame.pack(pady=2)

    lbl_video = tk.Label(main, bg="#1e1e1e")
    lbl_video.pack(padx=4, pady=4)
    photo_ref = [None]

    progress_frame = tk.Frame(main, bg="#2b2b2b")
    progress_frame.pack(fill="x", pady=8)
    tk.Label(
        progress_frame,
        text="Progreso por pose (frontal | lateral izq. | lateral der.):",
        font=("Segoe UI", 9),
        fg="#ccc",
        bg="#2b2b2b",
    ).pack(anchor="w")
    progress_var = tk.DoubleVar(value=0.0)
    progress_bar = ttk.Progressbar(
        progress_frame,
        variable=progress_var,
        maximum=MIN_FRAMES_POR_POSE * 3,
        length=400,
        mode="determinate",
    )
    progress_bar.pack(fill="x", pady=4)
    lbl_progress = tk.Label(
        progress_frame,
        text="Frontal: 0 | Izq: 0 | Der: 0",
        font=("Segoe UI", 9),
        fg="#0f0",
        bg="#2b2b2b",
    )
    lbl_progress.pack(anchor="w")

    def worker():
        frame_idx = 0
        app = get_face_app()
        while playing[0]:
            ret, frame = cap.read()
            if not ret:
                n_total = sum(len(b) for b in pose_buckets.values())
                try:
                    frame_queue.put_nowait((None, pose_buckets, frame_idx, None, True))
                except Exception:
                    pass
                break

            faces = app.get(frame)
            frame_dibujo = frame.copy()
            if faces:
                face = faces[0]
                dibujar_cara_insightface(frame_dibujo, face)
                if frame_idx % CADA_N_FRAMES == 0 and hasattr(face, "embedding") and face.embedding is not None:
                    pose = clasificar_pose(face)
                    if len(pose_buckets[pose]) < MIN_FRAMES_POR_POSE:
                        pose_buckets[pose].append(face.embedding.copy())
                        encodings_recogidos.append(face.embedding.copy())
            nf, ni, nd = len(pose_buckets["frontal"]), len(pose_buckets["izq"]), len(pose_buckets["der"])
            try:
                frame_queue.put_nowait((frame_dibujo, {"frontal": nf, "izq": ni, "der": nd}, frame_idx, None, False))
            except Exception:
                pass
            if nf >= MIN_FRAMES_POR_POSE and ni >= MIN_FRAMES_POR_POSE and nd >= MIN_FRAMES_POR_POSE:
                try:
                    frame_queue.put_nowait((None, {"frontal": nf, "izq": ni, "der": nd}, frame_idx, None, True))
                except Exception:
                    pass
                break
            frame_idx += 1

        cap.release()

    def poll_queue():
        if not playing[0]:
            return
        try:
            while True:
                item = frame_queue.get_nowait()
                frame_dibujo, counts, frame_idx, _, fin = item
                nf = counts.get("frontal", 0)
                ni = counts.get("izq", 0)
                nd = counts.get("der", 0)
                n_total = nf + ni + nd
                if fin or frame_dibujo is None:
                    if fin:
                        progress_var.set(MIN_FRAMES_POR_POSE * 3)
                        lbl_progress.config(text=f"Frontal: {nf} | Izq: {ni} | Der: {nd} — Finalizado")
                        lbl_frame.config(text=f"Frame: {frame_idx} — Finalizado")
                        guardar_parametrizacion(encodings_recogidos, id_persona, nombre, path_out)
                    root.after(50, poll_queue)
                    return
                progress_var.set(n_total)
                lbl_progress.config(text=f"Frontal: {nf} | Izq: {ni} | Der: {nd}")
                lbl_frame.config(text=f"Frame: {frame_idx}")
                photo_ref[0] = frame_bgr_a_photoimage(frame_dibujo)
                if photo_ref[0]:
                    lbl_video.config(image=photo_ref[0])
        except Empty:
            pass
        root.after(40, poll_queue)

    def guardar_parametrizacion(encodings_list, id_persona, nombre, path_out):
        """Guarda en face_tracking/<id>_<nombre>.pkl el embedding medio (512-d) y metadatos para ArcFace."""
        if not encodings_list:
            return
        encodings_arr = np.array(encodings_list[:NUM_FRAMES_META], dtype=np.float32)
        encoding_mean = encodings_arr.mean(axis=0)
        # Normalizar para comparación por similitud coseno (opcional pero habitual con ArcFace)
        norm = np.linalg.norm(encoding_mean)
        if norm > 1e-6:
            encoding_mean = encoding_mean / norm
        data = {
            "id": id_persona,
            "nombre": nombre,
            "encoding": encoding_mean,
            "encodings": [encodings_arr[i] for i in range(len(encodings_arr))],
            "num_frames": len(encodings_arr),
            "model": "arcface_insightface",
        }
        with open(path_out, "wb") as f:
            pickle.dump(data, f)
        print(f"Parametrización guardada en: {path_out}")

    def on_cerrar():
        playing[0] = False
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_cerrar)
    tk.Button(root, text="Cerrar", command=on_cerrar, font=("Segoe UI", 10), cursor="hand2").pack(pady=6)

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    root.after(50, poll_queue)
    root.mainloop()

    return path_out.exists()


def main():
    parser = argparse.ArgumentParser(
        description="Captura cara desde un vídeo con ArcFace (InsightFace) y guarda la parametrización para identificación."
    )
    parser.add_argument("video", type=str, help="Ruta al fichero de vídeo")
    parser.add_argument("--id", dest="id_persona", type=int, required=True, help="ID numérico de la persona")
    parser.add_argument("--nombre", type=str, required=True, help="Nombre de la persona (para el fichero de salida)")
    args = parser.parse_args()

    ok = run_face_tracking(Path(args.video), args.id_persona, args.nombre)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
