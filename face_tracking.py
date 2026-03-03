#!/usr/bin/env python3
"""
Captura de puntos faciales y parametrización de una cara desde un vídeo.
Usa InsightFace (ArcFace) para embeddings 512-d de alta calidad en identificación.
- Recibe: path del vídeo, id (entero), nombre (string).
- Recoge al menos 10 frames de vista frontal, 10 de lateral izquierdo y 10 de lateral derecho
  para una identificación más robusta (el otro lateral y mirar arriba/abajo son opcionales pero recomendables).
- Muestra en ventana Tkinter el vídeo con la cara (bbox + puntos clave) y barra de progreso por pose.
- Al terminar guarda la parametrización en face_tracking/<id>_<nombre>.pkl
  y actualiza register_users.json con id, nombre, ruta del pkl, edad y género estimados.
"""

import argparse
import json
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

# Optimización CPU/GPU: detectar CUDA (para InsightFace) y limitar hilos en CPU
HAS_CUDA = False
INSIGHTFACE_CTX_ID = -1
try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
    INSIGHTFACE_CTX_ID = 0 if HAS_CUDA else -1
    torch.set_num_threads(min(4, torch.get_num_threads()))
except Exception:
    HAS_CUDA = False
    INSIGHTFACE_CTX_ID = -1

# InsightFace (ArcFace): carga perezosa para no ralentizar el arranque
_app_insightface = None

def get_face_app():
    """Inicializa InsightFace FaceAnalysis una sola vez (descarga modelos en primer uso)."""
    global _app_insightface
    if _app_insightface is None:
        from insightface.app import FaceAnalysis
        # buffalo_l; det_size mayor en registro para embeddings de mejor calidad
        _app_insightface = FaceAnalysis(name="buffalo_l", root=str(Path.home() / ".insightface"))
        try:
            _app_insightface.prepare(ctx_id=INSIGHTFACE_CTX_ID, det_thresh=0.4, det_size=(512, 512))
        except Exception:
            # Si falla en GPU, reintentar en CPU
            _app_insightface.prepare(ctx_id=-1, det_thresh=0.4, det_size=(512, 512))
    return _app_insightface

# Mínimo por pose para considerar "mínima calidad"; seguimos capturando hasta el máximo
MIN_FRAMES_POR_POSE = 8
# Máximo por pose: procesamos todo el vídeo hasta llegar aquí (ej. 20 s → muchas caras)
MAX_FRAMES_POR_POSE = 80
# Cada cuántos frames extraemos embedding
CADA_N_FRAMES = 4

# Índices de los 5 keypoints InsightFace: [ojo_izq, ojo_der, nariz, boca_izq, boca_der]
KPS_OJO_IZQ, KPS_OJO_DER, KPS_NARIZ = 0, 1, 2
# Umbral (fracción del ancho de la cara) para considerar lateral vs frontal
YAW_THRESHOLD = 0.12

BASE_DIR = Path(__file__).resolve().parent
DIR_FACE_TRACKING = BASE_DIR / "face_tracking"
REGISTER_USERS_PATH = BASE_DIR / "register_users.json"


def _load_registered_users():
    """Carga register_users.json como lista de dicts."""
    if not REGISTER_USERS_PATH.is_file():
        return []
    try:
        with open(REGISTER_USERS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def _save_registered_users(users):
    REGISTER_USERS_PATH.write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")


def registrar_usuario(id_persona: int, nombre: str, path_pkl: Path, edad: int | None = None, genero: str | None = None):
    """
    Registra un usuario en register_users.json con:
    - id: el id_persona pasado al script (se asume que empiezas en 1 y vas incrementando).
    - nombre
    - pkl_path
    Si ya existe una entrada con el mismo id o mismo pkl_path, se actualiza.
    """
    users = _load_registered_users()
    pkl_str = str(path_pkl)
    actualizado = False
    for u in users:
        if u.get("id") == id_persona or u.get("pkl_path") == pkl_str:
            u["id"] = id_persona
            u["nombre"] = nombre
            u["pkl_path"] = pkl_str
            if edad is not None:
                u["edad"] = int(edad)
            if genero is not None:
                u["genero"] = genero
            actualizado = True
            break
    if not actualizado:
        entry = {"id": id_persona, "nombre": nombre, "pkl_path": pkl_str}
        if edad is not None:
            entry["edad"] = int(edad)
        if genero is not None:
            entry["genero"] = genero
        users.append(entry)
    _save_registered_users(users)


def frame_bgr_a_photoimage(frame_bgr, max_ancho=800, max_alto=600):
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
    # Flag para no guardar dos veces la parametrización
    registro_guardado = [False]
    # Metadatos estimados del usuario (edad, género) usando InsightFace
    user_meta = {"edad": None, "genero": None}

    # --- UI principal: vídeo a la izquierda, info a la derecha ---
    root = tk.Tk()
    root.title(f"Face tracking (ArcFace) — {path_video.name}")
    root.geometry("1150x650")
    root.minsize(1150, 650)
    root.resizable(False, False)
    root.configure(bg="#202020")

    main = tk.Frame(root, bg="#202020")
    main.pack(fill="both", expand=True, padx=10, pady=10)

    # Columna izquierda: solo vídeo
    left = tk.Frame(main, bg="#202020")
    left.pack(side="left", fill="both", expand=True, padx=(0, 10))

    lbl_video = tk.Label(left, bg="#101010", bd=1, relief="sunken")
    lbl_video.pack(fill="both", expand=True)
    photo_ref = [None]

    # Columna derecha: datos del usuario y progreso
    right = tk.Frame(main, bg="#252525", bd=1, relief="ridge")
    right.pack(side="right", fill="y")

    lbl_title = tk.Label(
        right,
        text="Registro de usuario (ArcFace)",
        font=("Segoe UI", 13, "bold"),
        fg="white",
        bg="#252525",
    )
    lbl_title.pack(anchor="w", padx=12, pady=(10, 4))

    lbl_info = tk.Label(
        right,
        text=f"ID: {id_persona}   Nombre: {nombre}",
        font=("Segoe UI", 11),
        fg="#e0e0e0",
        bg="#252525",
    )
    lbl_info.pack(anchor="w", padx=12, pady=(0, 4))

    lbl_hint = tk.Label(
        right,
        text="Mueve la cara: frontal, izquierda, derecha, arriba, abajo.\n"
             "Se captura durante todo el vídeo hasta máxima calidad (máx. 80 por pose).",
        font=("Segoe UI", 9),
        fg="#aaaaaa",
        bg="#252525",
        justify="left",
        wraplength=340,
    )
    lbl_hint.pack(anchor="w", padx=12, pady=(0, 8))

    lbl_frame = tk.Label(
        right,
        text="Frame: 0",
        font=("Segoe UI", 10),
        fg="#bbbbbb",
        bg="#252525",
    )
    lbl_frame.pack(anchor="w", padx=12, pady=(0, 8))

    progress_frame = tk.Frame(right, bg="#252525")
    progress_frame.pack(fill="x", padx=12, pady=8)
    tk.Label(
        progress_frame,
        text="Progreso por pose (frontal | lateral izq. | lateral der.):",
        font=("Segoe UI", 10),
        fg="#dddddd",
        bg="#252525",
    ).pack(anchor="w")

    progress_var = tk.DoubleVar(value=0.0)
    progress_bar = ttk.Progressbar(
        progress_frame,
        variable=progress_var,
        maximum=100,
        length=280,
        mode="determinate",
    )
    progress_bar.pack(fill="x", pady=4)
    lbl_progress = tk.Label(
        progress_frame,
        text="Frontal: 0 | Izq: 0 | Der: 0",
        font=("Segoe UI", 10),
        fg="#80ff80",
        bg="#252525",
    )
    lbl_progress.pack(anchor="w")

    # Sexo y edad estimada (se rellena al finalizar)
    lbl_meta = tk.Label(
        right,
        text="Sexo: —    Edad estimada: —",
        font=("Segoe UI", 11),
        fg="#f0f0f0",
        bg="#252525",
    )
    lbl_meta.pack(anchor="w", padx=12, pady=(10, 4))

    # Botón cerrar debajo
    btn_close = tk.Button(
        right,
        text="Cerrar",
        font=("Segoe UI", 10),
        cursor="hand2",
        width=14,
    )
    btn_close.pack(anchor="center", pady=(16, 12))

    def worker():
        frame_idx = 0
        total_frames = None
        app = get_face_app()
        last_face = None
        while playing[0]:
            ret, frame = cap.read()
            if total_frames is None:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames <= 0:
                    total_frames = 999999

            if not ret:
                nf, ni, nd = len(pose_buckets["frontal"]), len(pose_buckets["izq"]), len(pose_buckets["der"])
                try:
                    frame_queue.put_nowait((None, {"frontal": nf, "izq": ni, "der": nd}, frame_idx, total_frames, True))
                except Exception:
                    pass
                break

            frame_dibujo = frame.copy()

            # Solo hacemos detección+embedding 1 de cada CADA_N_FRAMES frames
            if frame_idx % CADA_N_FRAMES == 0:
                faces = app.get(frame)
                if faces:
                    last_face = faces[0]
                    dibujar_cara_insightface(frame_dibujo, last_face)
                    # Estimar edad y género solo una vez, si el modelo lo proporciona
                    if user_meta["edad"] is None and hasattr(last_face, "age"):
                        try:
                            user_meta["edad"] = int(round(float(last_face.age)))
                        except Exception:
                            user_meta["edad"] = None
                    if user_meta["genero"] is None and hasattr(last_face, "gender"):
                        try:
                            g = int(round(float(last_face.gender)))
                            # Convención InsightFace: 0=female, 1=male
                            user_meta["genero"] = "mujer" if g == 0 else "hombre"
                        except Exception:
                            user_meta["genero"] = None
                    if hasattr(last_face, "embedding") and last_face.embedding is not None:
                        pose = clasificar_pose(last_face)
                        if len(pose_buckets[pose]) < MAX_FRAMES_POR_POSE:
                            pose_buckets[pose].append(last_face.embedding.copy())
                            encodings_recogidos.append(last_face.embedding.copy())
            else:
                # En frames intermedios solo dibujamos la última cara detectada (sin recalcular ArcFace)
                if last_face is not None:
                    dibujar_cara_insightface(frame_dibujo, last_face)
            nf, ni, nd = len(pose_buckets["frontal"]), len(pose_buckets["izq"]), len(pose_buckets["der"])
            try:
                frame_queue.put_nowait((frame_dibujo, {"frontal": nf, "izq": ni, "der": nd, "_total": total_frames}, frame_idx, total_frames, False))
            except Exception:
                pass

            # Si ya hemos alcanzado el máximo de frames por pose en las tres poses,
            # podemos terminar el procesado aunque el vídeo sea más largo.
            if (
                nf >= MAX_FRAMES_POR_POSE
                and ni >= MAX_FRAMES_POR_POSE
                and nd >= MAX_FRAMES_POR_POSE
            ):
                break

            frame_idx += 1

        try:
            frame_queue.put_nowait((None, {"frontal": nf, "izq": ni, "der": nd}, frame_idx, total_frames, True))
        except Exception:
            pass
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
                        progress_var.set(100)
                        lbl_progress.config(text=f"Frontal: {nf} | Izq: {ni} | Der: {nd} — Finalizado")
                        lbl_frame.config(text=f"Frame: {frame_idx} — Finalizado")
                        # Actualizar texto de sexo y edad estimada
                        sexo_txt = user_meta.get("genero") or "desconocido"
                        edad_val = user_meta.get("edad")
                        if edad_val is not None:
                            edad_txt = f"{int(edad_val)} años"
                        else:
                            edad_txt = "desconocida"
                        lbl_meta.config(text=f"Sexo: {sexo_txt.capitalize()}    Edad estimada: {edad_txt}")
                        guardar_parametrizacion(encodings_recogidos, id_persona, nombre, path_out, user_meta)
                    root.after(50, poll_queue)
                    return
                total_f = counts.get("_total", 1)
                pct = min(100, 100 * frame_idx / total_f) if total_f > 0 else 0
                progress_var.set(pct)
                lbl_progress.config(text=f"Frontal: {nf} | Izq: {ni} | Der: {nd}  ({frame_idx}/{total_f})")
                lbl_frame.config(text=f"Frame: {frame_idx}")
                photo_ref[0] = frame_bgr_a_photoimage(frame_dibujo)
                if photo_ref[0]:
                    lbl_video.config(image=photo_ref[0])
        except Empty:
            pass
        root.after(40, poll_queue)

    def guardar_parametrizacion(encodings_list, id_persona, nombre, path_out, user_meta):
        """Guarda en face_tracking/<id>_<nombre>.pkl el embedding medio (512-d) y metadatos para ArcFace."""
        if registro_guardado[0]:
            return
        if not encodings_list:
            return
        encodings_arr = np.array(encodings_list, dtype=np.float32)
        # Normalizar cada encoding para similitud coseno
        encodings_norm = []
        for i in range(len(encodings_arr)):
            e = encodings_arr[i].astype(np.float32)
            nrm = np.linalg.norm(e)
            if nrm > 1e-6:
                e = e / nrm
            encodings_norm.append(e)
        encoding_mean = np.array(encodings_norm, dtype=np.float32).mean(axis=0)
        nrm_mean = np.linalg.norm(encoding_mean)
        if nrm_mean > 1e-6:
            encoding_mean = encoding_mean / nrm_mean
        data = {
            "id": id_persona,
            "nombre": nombre,
            "encoding": encoding_mean,
            "encodings": encodings_norm,
            "num_frames": len(encodings_arr),
            "model": "arcface_insightface",
        }
        if user_meta is not None:
            if user_meta.get("edad") is not None:
                data["edad"] = int(user_meta["edad"])
            if user_meta.get("genero") is not None:
                data["genero"] = user_meta["genero"]
        with open(path_out, "wb") as f:
            pickle.dump(data, f)
        registro_guardado[0] = True
        print(f"Parametrización guardada en: {path_out}")
        # Registrar en JSON global para que test_detect_clothes.py pueda identificar usuarios
        registrar_usuario(
            id_persona,
            nombre,
            path_out,
            edad=user_meta.get("edad") if user_meta else None,
            genero=user_meta.get("genero") if user_meta else None,
        )

    def on_cerrar():
        # Al cerrar manualmente, si ya tenemos encodings pero aún no se ha guardado, guardar ahora.
        playing[0] = False
        if encodings_recogidos and not registro_guardado[0]:
            guardar_parametrizacion(encodings_recogidos, id_persona, nombre, path_out, user_meta)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_cerrar)
    btn_close.config(command=on_cerrar)

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
