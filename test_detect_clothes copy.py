#!/usr/bin/env python3
"""
Prueba de detección de ropa y complementos en vídeo.
- Detecta personas con YOLO.
- Para cada persona: complementos (mochila, bolso, paraguas, corbata, maleta),
  color de la parte superior e inferior.
- Usa tracking (BoT-SORT/ByteTrack) para IDs consistentes.
- Si hay usuarios registrados en face_tracking (ArcFace), intenta identificarlos
  comparando la cara frontal con sus embeddings y muestra nombre/ID en el panel.
"""

import argparse
import json
import warnings
import pickle
import sys
import threading
from pathlib import Path
from queue import Empty, Queue

import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO
import tkinter as tk

# Optimización CPU/GPU: limitar hilos en CPU y detectar CUDA
HAS_CUDA = False
DEVICE = "cpu"
try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
    DEVICE = "cuda" if HAS_CUDA else "cpu"
    torch.set_num_threads(min(4, torch.get_num_threads()))
except Exception:
    HAS_CUDA = False
    DEVICE = "cpu"

PATH_VIDEOS = "/home/debian/sharedVM/sergi_reconocimiento_facial/finalesv2"
# Modelos YOLO por defecto (puedes cambiarlos fácil para probar otros)
YOLO_MODEL = "yolo11x.pt"
YOLO_MODEL_POSE = "yolo11x-pose.pt"
BASE_DIR = Path(__file__).resolve().parent
ENGINES_DIR = BASE_DIR / "engines"

# Suprimir FutureWarning de InsightFace (skimage: tform.estimate deprecado)
warnings.filterwarnings("ignore", message=".*estimate.*deprecated.*", category=FutureWarning)

# ----------------------
# Identificación (ArcFace / InsightFace)
# ----------------------
_arcface_app = None


def get_arcface_app():
    """
    Carga InsightFace FaceAnalysis (ArcFace) una vez para todo el script.
    Usa el mismo modelo que face_tracking.py (buffalo_l).
    """
    global _arcface_app
    if _arcface_app is None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            return None
        root_dir = Path.home() / ".insightface"
        app = FaceAnalysis(name="buffalo_l", root=str(root_dir))
        # det_size mayor para detectar caras pequeñas (persona lejos en 4K)
        # ctx_id: 0→GPU si hay CUDA, -1→CPU
        ctx_id = 0 if HAS_CUDA else -1
        try:
            app.prepare(ctx_id=ctx_id, det_thresh=0.4, det_size=(640, 640))
        except Exception:
            # Si falla en GPU, reintentar en CPU
            app.prepare(ctx_id=-1, det_thresh=0.4, det_size=(640, 640))
        _arcface_app = app
    return _arcface_app


def load_registered_users():
    """
    Carga register_users.json generado por face_tracking.py.
    Devuelve lista de dicts con id, nombre, pkl_path, encoding (media) y encodings (lista).
    Usa encodings para comparación best-of-N (mejor que solo la media).
    """
    base_dir = Path(__file__).resolve().parent
    reg_path = base_dir / "register_users.json"
    if not reg_path.is_file():
        return []
    try:
        with open(reg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    users = []
    for u in data if isinstance(data, list) else []:
        pkl_path = Path(u.get("pkl_path", ""))
        if not pkl_path.is_file():
            continue
        try:
            with open(pkl_path, "rb") as f:
                contenido = pickle.load(f)
            enc_mean = np.asarray(contenido.get("encoding"), dtype=np.float32)
            if enc_mean.ndim != 1:
                continue
            nrm = np.linalg.norm(enc_mean)
            if nrm > 1e-6:
                enc_mean = enc_mean / nrm
            encodings_list = contenido.get("encodings", [])
            encodings_norm = []
            for e in encodings_list:
                e = np.asarray(e, dtype=np.float32)
                if e.ndim != 1:
                    continue
                n = np.linalg.norm(e)
                if n > 1e-6:
                    e = e / n
                encodings_norm.append(e)
            if not encodings_norm:
                encodings_norm = [enc_mean]
            users.append(
                {
                    "id": int(u.get("id")),
                    "nombre": contenido.get("nombre") or u.get("nombre", "desconocido"),
                    "pkl_path": str(pkl_path),
                    "encoding": enc_mean,
                    "encodings": encodings_norm,
                }
            )
        except Exception:
            continue
    return users


def identificar_cara_arcface(crop_bgr, usuarios_db, app_arcface, sim_threshold=0.42, debug=False, frame_idx=None):
    """
    Obtiene embedding ArcFace de crop_bgr y lo compara con la base de usuarios.
    Usa best-of-N: compara con todos los encodings guardados de cada usuario y toma el mejor score.
    Devuelve (identidad, has_face, edad, genero, embedding): identidad dict o None; embedding para re-ID.
    """
    if crop_bgr is None or crop_bgr.size == 0 or app_arcface is None:
        return None, False, None, None, None
    img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    faces = app_arcface.get(img)
    if not faces:
        return None, False, None, None, None
    face = faces[0]
    edad = int(face.age) if hasattr(face, "age") and face.age is not None else None
    genero = "hombre" if (hasattr(face, "gender") and face.gender == 0) else ("mujer" if (hasattr(face, "gender") and face.gender == 1) else None)
    if not hasattr(face, "embedding") or face.embedding is None:
        return None, True, edad, genero, None
    emb = np.asarray(face.embedding, dtype=np.float32)
    nrm = np.linalg.norm(emb)
    if nrm > 1e-6:
        emb = emb / nrm
    if not usuarios_db:
        return None, True, edad, genero, emb
    best_score = -1.0
    best_user = None
    scores_per_user = [] if debug else None
    for u in usuarios_db:
        encodings = u.get("encodings", [u["encoding"]])
        best_u = max(float(np.dot(vec, emb)) for vec in encodings)
        if debug:
            scores_per_user.append((u, best_u))
        if best_u > best_score:
            best_score = best_u
            best_user = u
    if debug and scores_per_user:
        frame_pref = f"Frame {frame_idx} | " if frame_idx is not None else ""
        log_parts = [f"{u['nombre']} ({Path(u['pkl_path']).name}): {s:.3f}" for u, s in scores_per_user]
        print(f"  [debug-face] {frame_pref}similaridad vs .pkl → {', '.join(log_parts)} | umbral={sim_threshold} | {'MATCH' if best_user and best_score >= sim_threshold else 'no match'}")
    if best_user is not None and best_score >= sim_threshold:
        return {"id": best_user["id"], "nombre": best_user["nombre"], "score": best_score}, True, edad, genero, emb
    return None, True, edad, genero, emb

def get_upper_crop(frame, xyxy, h_frame, w_frame, ratio=0.45):
    """Recorte de la parte superior de la bbox (zona cara/hombros) para comprobar cara frontal."""
    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_frame, x2), min(h_frame, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    h_crop = int((y2 - y1) * ratio)
    if h_crop < 1:
        return None
    return frame[y1 : y1 + h_crop, x1:x2].copy()


def get_face_crop_for_arcface(frame, xyxy, h_frame, w_frame, ratio=0.55, min_side=640):
    """
    Recorte de la zona cara desde el frame a máxima resolución.
    Para personas lejanas (p. ej. 5 m en 4K) el crop es pequeño; hace zoom (upscale)
    hasta min_side para que InsightFace detecte y ArcFace identifique correctamente.
    frame: frame completo a resolución original (4K, 1080p, etc.).
    """
    crop = get_upper_crop(frame, xyxy, h_frame, w_frame, ratio=ratio)
    if crop is None:
        return None
    h, w = crop.shape[:2]
    if h < min_side or w < min_side:
        scale = max(min_side / h, min_side / w)
        nw = max(min_side, int(w * scale))
        nh = max(min_side, int(h * scale))
        crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    return crop


# COCO pose: solo los índices que necesitamos (torso 4 pts, piernas 6 pts)
# 5=L_shoulder, 6=R_shoulder, 11=L_hip, 12=R_hip, 13=L_knee, 14=R_knee, 15=L_ankle, 16=R_ankle
IDX_TORSO = (5, 6, 11, 12)  # hombros + caderas
IDX_PIERNAS = (11, 12, 13, 14, 15, 16)  # caderas, rodillas, tobillos


def _get_pts_validos(xy, conf, indices, conf_min=0.25):
    """Devuelve lista de (x,y) para los índices indicados si conf >= conf_min."""
    pts = []
    for i in indices:
        if i < xy.shape[0] and i < len(conf) and conf[i] >= conf_min:
            pts.append((float(xy[i][0]), float(xy[i][1])))
    return pts


def get_torso_y_piernas_from_pose(crop_bgr, pose_model, conf_min=0.25):
    """
    Usa YOLO-pose (solo keypoints necesarios) para:
    a) Torso: bbox con 4 puntos (hombros + caderas) → color más común
    b) Piernas: bbox desde caderas hasta tobillos (cadera→rodilla→pie por cada lado)
    Devuelve (zona_torso, zona_piernas) como recortes BGR para captura y color.
    No dibuja esqueleto.
    """
    if crop_bgr is None or crop_bgr.size == 0 or pose_model is None:
        return None, None
    h_crop, w_crop = crop_bgr.shape[:2]
    if h_crop < 20 or w_crop < 20:
        return None, None
    try:
        results = pose_model(crop_bgr, verbose=False)
        if not results or len(results) == 0:
            return None, None
        r = results[0]
        kpts = r.keypoints
        if kpts is None:
            return None, None
        xy = kpts.xy[0].cpu().numpy() if len(kpts.xy) > 0 else None
        conf = kpts.conf[0].cpu().numpy() if len(kpts.conf) > 0 else None
        if xy is None or conf is None or xy.shape[0] < 17:
            return None, None
    except Exception:
        return None, None

    def _bbox_from_pts(pts, mw, mh):
        ys, xs = [p[1] for p in pts], [p[0] for p in pts]
        y1 = max(0, int(min(ys)) - mh)
        y2 = min(h_crop, int(max(ys)) + mh)
        x1 = max(0, int(min(xs)) - mw)
        x2 = min(w_crop, int(max(xs)) + mw)
        if y2 > y1 and x2 > x1:
            return crop_bgr[y1:y2, x1:x2].copy()
        return None

    margin = 0.08
    mw = max(2, int(w_crop * margin))
    mh = max(2, int(h_crop * margin))

    pts_torso = _get_pts_validos(xy, conf, IDX_TORSO, conf_min)
    zona_torso = _bbox_from_pts(pts_torso, mw, mh) if len(pts_torso) >= 3 else None

    pts_piernas = _get_pts_validos(xy, conf, IDX_PIERNAS, conf_min)
    zona_piernas = _bbox_from_pts(pts_piernas, mw, mh) if len(pts_piernas) >= 4 else None

    return zona_torso, zona_piernas


# Clases COCO: persona, perro y complementos vestimentarios
COCO_PERSON = 0
COCO_DOG = 16  # perro en COCO 80 clases
COCO_ACCESSORY_IDS = {
    24: "mochila",
    25: "paraguas",
    26: "bolso",
    27: "corbata",
    28: "maleta",
}

# Nombres de color en español (HSV aproximado: H en 0-180, S/V en 0-255)
# Orden importante: colores con tono definido (rojo, azul...) con V bajo ANTES que "gris oscuro"
# para que rojo oscuro/burdeos no se clasifique como gris.
COLOR_NAMES_HSV = [
    # (H_min, H_max, S_min, V_min, nombre)
    (0, 0, 0, 0, "negro"),
    (0, 180, 0, 250, "blanco"),
    # Rojo (incl. rojo oscuro/burdeos): H 0-10 o 170-180, S algo alta, V bajo permitido
    (0, 10, 40, 35, "rojo"),
    (170, 180, 40, 35, "rojo"),
    (10, 25, 100, 80, "naranja"),
    (25, 35, 100, 80, "amarillo"),
    (35, 85, 80, 80, "verde"),
    (85, 100, 80, 80, "verde lima"),
    (100, 130, 80, 80, "azul"),
    (130, 160, 80, 80, "violeta"),
    (160, 170, 80, 80, "magenta"),
    (0, 10, 50, 200, "rosa"),
    (10, 25, 80, 180, "rosa/naranja"),
    (0, 30, 30, 100, "marrón"),
    (20, 40, 30, 200, "beige"),
    # Grises después de los colores con tono
    (0, 180, 0, 50, "gris oscuro"),
    (0, 180, 1, 50, "gris"),
    (0, 180, 0, 200, "gris claro"),
]

# Mapeo nombre color -> RGB para rectángulos en el panel (Tkinter usa #rrggbb)
COLOR_NAME_TO_RGB = {
    "negro": "#1a1a1a",
    "blanco": "#ffffff",
    "gris oscuro": "#404040",
    "gris": "#808080",
    "gris claro": "#c0c0c0",
    "rojo": "#c03030",
    "naranja": "#ff8800",
    "amarillo": "#e0c040",
    "verde": "#208020",
    "verde lima": "#90ee90",
    "azul": "#3060c0",
    "azul turquesa": "#40e0d0",
    "violeta": "#8040a0",
    "magenta": "#c04080",
    "rosa": "#ffb6c1",
    "rosa/naranja": "#ffb088",
    "marrón": "#804020",
    "beige": "#f5f5dc",
    "oscuro": "#2a2a2a",
    "colorido": "#8080c0",
    "no detectable": "#505050",
}


def rgb_to_hsv(r, g, b):
    """Convierte RGB (0-255) a HSV (H 0-180, S/V 0-255) como en OpenCV."""
    bgr = np.uint8([[[b, g, r]]])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return tuple(map(int, hsv[0][0]))


def nombre_color_desde_rgb(r, g, b):
    """Devuelve el nombre en español del color más cercano para un RGB dado."""
    h, s, v = rgb_to_hsv(r, g, b)
    mejor = "desconocido"
    mejor_dist = float("inf")
    for hmn, hmx, smn, vmn, nombre in COLOR_NAMES_HSV:
        if hmn <= h <= hmx and s >= smn and v >= vmn:
            dist = abs(h - (hmn + hmx) / 2) + 0.01 * (smn - s if s < smn else 0)
            if dist < mejor_dist:
                mejor_dist = dist
                mejor = nombre
    if mejor == "desconocido":
        if v < 60:
            mejor = "oscuro"
        elif s < 30:
            mejor = "gris claro" if v > 180 else "gris"
        else:
            mejor = "colorido"
    return mejor


def rgb_to_hex(r, g, b):
    """Convierte R,G,B (0-255) a cadena hexadecimal #rrggbb."""
    r, g, b = max(0, min(255, int(r))), max(0, min(255, int(g))), max(0, min(255, int(b)))
    return f"#{r:02x}{g:02x}{b:02x}"


def color_dominante_region(imagen_bgr, max_muestras=2000, paso_cuantizacion=32, umbral_moda=0.25):
    """
    Color dominante: el más frecuente (moda) en hex, o media si no hay uno claro.
    Cuantiza RGB en bins; si un bin tiene > umbral_moda de píxeles, usa la media de ese bin;
    si no, usa la media de toda la región. Devuelve {"hex": "#rrggbb", "nombre": "rojo"}.
    """
    if imagen_bgr is None or imagen_bgr.size == 0:
        return {"hex": "#505050", "nombre": "no detectable"}
    pixels = imagen_bgr.reshape(-1, 3)
    n = len(pixels)
    if n > max_muestras:
        idx = np.random.choice(n, max_muestras, replace=False)
        pixels = pixels[idx]
        n = len(pixels)
    paso = paso_cuantizacion
    bins_r = (pixels[:, 2] // paso).astype(np.int32)
    bins_g = (pixels[:, 1] // paso).astype(np.int32)
    bins_b = (pixels[:, 0] // paso).astype(np.int32)
    n_niveles = 256 // paso + 1
    bin_flat = bins_r * n_niveles * n_niveles + bins_g * n_niveles + bins_b
    valores, counts = np.unique(bin_flat, return_counts=True)
    imax = np.argmax(counts)
    count_max = int(counts[imax])
    bin_ganador = valores[imax]
    if count_max >= n * umbral_moda:
        mascara = bin_flat == bin_ganador
        pix_bin = pixels[mascara]
        r = int(np.mean(pix_bin[:, 2]))
        g = int(np.mean(pix_bin[:, 1]))
        b = int(np.mean(pix_bin[:, 0]))
    else:
        r = int(np.mean(pixels[:, 2]))
        g = int(np.mean(pixels[:, 1]))
        b = int(np.mean(pixels[:, 0]))
    r, g, b = max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))
    hex_str = rgb_to_hex(r, g, b)
    nombre = nombre_color_desde_rgb(r, g, b)
    return {"hex": hex_str, "nombre": nombre}


def nombre_color_a_hex(nombre):
    """Devuelve color hex para el panel (#rrggbb)."""
    return COLOR_NAME_TO_RGB.get(nombre, "#505050")


def deduplicar_personas(person_boxes, iou_umbral=0.5):
    """
    NMS: elimina detecciones duplicadas de la misma persona (IoU alto).
    Mantiene la de mayor confianza cuando se solapan.
    """
    if len(person_boxes) <= 1:
        return person_boxes
    ordenadas = sorted(person_boxes, key=lambda x: -x[1])
    keep = []
    for cand in ordenadas:
        xyxy_c, conf_c, tid_c = cand
        solapado = False
        for guardada in keep:
            if iou_box(xyxy_c, guardada[0]) >= iou_umbral:
                solapado = True
                break
        if not solapado:
            keep.append(cand)
    return keep


def iou_box(box1, box2):
    """IoU entre dos cajas [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def centro_en_box(cx, cy, box):
    """True si el punto (cx, cy) está dentro de box [x1,y1,x2,y2] (expandida 20%)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    margin = 0.2
    x1e = x1 - margin * w
    y1e = y1 - margin * h
    x2e = x2 + margin * w
    y2e = y2 + margin * h
    return x1e <= cx <= x2e and y1e <= cy <= y2e


def asociar_complementos_a_personas(person_boxes, accessory_boxes):
    """
    person_boxes: lista de (xyxy, conf) o (xyxy, conf, track_id)
    accessory_boxes: lista de (xyxy, conf, class_id)
    Devuelve para cada persona una lista de (nombre, xyxy) para poder extraer color del complemento.
    """
    resultado = [[] for _ in person_boxes]
    for acc_xyxy, acc_conf, acc_cls in accessory_boxes:
        nombre = COCO_ACCESSORY_IDS.get(int(acc_cls), f"clase_{acc_cls}")
        acc_cx = (acc_xyxy[0] + acc_xyxy[2]) / 2
        acc_cy = (acc_xyxy[1] + acc_xyxy[3]) / 2
        mejor_iou = 0.0
        mejor_idx = -1
        for i, item in enumerate(person_boxes):
            p_xyxy = item[0] if len(item) >= 1 else item
            if centro_en_box(acc_cx, acc_cy, p_xyxy):
                iou = iou_box(p_xyxy, acc_xyxy)
                # centro_en_box ya es True aquí; actualizar si mejor IoU o sin match previo
                if iou > mejor_iou or mejor_iou == 0:
                    mejor_iou = iou
                    mejor_idx = i
        if mejor_idx >= 0:
            nombres_ya = [c[0] for c in resultado[mejor_idx]]
            if nombre not in nombres_ya:
                resultado[mejor_idx].append((nombre, acc_xyxy.copy()))
    return resultado


def procesar_frame(frame, model, imgsz=320):
    """
    Detecta personas y complementos, extrae color superior/inferior por persona.
    frame: BGR (OpenCV). imgsz: tamaño de entrada YOLO (320 más rápido en CPU).
    """
    results = model.predict(frame, imgsz=imgsz, verbose=False)[0]
    boxes = results.boxes
    if boxes is None:
        return [], []

    person_boxes = []
    accessory_boxes = []
    dog_boxes = []
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        xyxy = boxes.xyxy[i].cpu().numpy()
        conf = float(boxes.conf[i].item())
        if cls_id == COCO_PERSON:
            person_boxes.append((xyxy, conf))
        elif cls_id == COCO_DOG:
            dog_boxes.append((xyxy, conf))
        elif cls_id in COCO_ACCESSORY_IDS:
            accessory_boxes.append((xyxy, conf, cls_id))

    # Ordenar personas por posición vertical (arriba a abajo)
    person_boxes.sort(key=lambda x: (x[0][1] + x[0][3]) / 2)

    complementos_por_persona = asociar_complementos_a_personas(
        person_boxes, accessory_boxes
    )

    salidas = []
    for idx, ((x1, y1, x2, y2), conf) in enumerate(person_boxes):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # Recortes válidos
        h_frame, w_frame = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_frame, x2)
        y2 = min(h_frame, y2)
        if x2 <= x1 or y2 <= y1:
            salidas.append({
                "complementos": [c[0] for c in (complementos_por_persona[idx] if idx < len(complementos_por_persona) else [])],
                "color_superior": "no detectable",
                "color_inferior": "no detectable",
            })
            continue
        crop = frame[y1:y2, x1:x2]
        alto = crop.shape[0]
        mitad = alto // 2
        zona_superior = crop[:mitad, :]
        zona_inferior = crop[mitad:, :]

        color_sup = color_dominante_region(zona_superior)
        color_inf = color_dominante_region(zona_inferior)

        comp = complementos_por_persona[idx] if idx < len(complementos_por_persona) else []
        comp_nombres = [c[0] for c in comp]
        salidas.append({
            "complementos": comp_nombres,
            "color_superior": color_sup["nombre"],
            "color_inferior": color_inf["nombre"],
        })
    return person_boxes, salidas


def track_frame(frame, model, imgsz=320, tracker="botsort.yaml"):
    """
    Ejecuta YOLO con tracking. Por defecto BoT-SORT (mejor cuando la persona se gira/cambia de postura).
    Returns: (person_boxes_with_id, accessory_boxes, complementos_por_persona, dog_boxes).
    """
    results = model.track(
        frame, imgsz=imgsz, persist=True, verbose=False,
        tracker=tracker,
    )
    if not results:
        return [], [], [], []
    r = results[0]
    boxes = r.boxes
    if boxes is None:
        return [], [], [], []
    ids = boxes.id  # puede ser None si no hay tracking

    person_boxes = []
    accessory_boxes = []
    dog_boxes = []
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        xyxy = boxes.xyxy[i].cpu().numpy()
        conf = float(boxes.conf[i].item())
        tid = int(ids[i].item()) if ids is not None and i < len(ids) else None
        if cls_id == COCO_PERSON:
            person_boxes.append((xyxy, conf, tid))
        elif cls_id == COCO_DOG:
            dog_boxes.append((xyxy, conf))
        elif cls_id in COCO_ACCESSORY_IDS:
            accessory_boxes.append((xyxy, conf, cls_id))

    person_boxes = deduplicar_personas(person_boxes)
    person_boxes.sort(key=lambda x: (x[0][1] + x[0][3]) / 2)
    # Asociar complementos (por índice de persona)
    comp_por_persona = asociar_complementos_a_personas(
        [(b[0], b[1]) for b in person_boxes], accessory_boxes
    )
    return person_boxes, accessory_boxes, comp_por_persona, dog_boxes


def _zoom_region_for_color(region_bgr, min_side=160):
    """
    Hace un pequeño "zoom" (upscale) sobre la región antes de estimar el color
    para ganar estabilidad cuando la persona está lejos o la bbox es pequeña.
    No añade información nueva, pero mejora la cuantización de color.
    """
    if region_bgr is None or region_bgr.size == 0:
        return region_bgr
    h, w = region_bgr.shape[:2]
    if min(h, w) >= min_side:
        return region_bgr
    escala = float(min_side) / float(min(h, w))
    nw = max(min_side, int(w * escala))
    nh = max(min_side, int(h * escala))
    return cv2.resize(region_bgr, (nw, nh), interpolation=cv2.INTER_LANCZOS4)


def extract_person_info(frame, xyxy, comp_list, h_frame, w_frame, pose_model=None):
    """
    Extrae para una persona: thumbnail (zona cara/arriba), color superior, inferior, complementos con color.
    Si pose_model está definido, usa YOLO-pose para segmentar torso (hombros→caderas) y parte inferior
    (caderas→rodillas), evitando piel/pelo/fondo y mejorando la extracción de color.
    """
    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w_frame, x2)
    y2 = min(h_frame, y2)
    if x2 <= x1 or y2 <= y1:
        return {
            "thumbnail": None,
            "crop_persona": None,
            "color_superior": "no detectable",
            "color_inferior": "no detectable",
            "complementos": [],
            "region_superior": None,
            "region_inferior": None,
        }
    crop = frame[y1:y2, x1:x2]
    h_crop, w_crop = crop.shape[:2]
    # Crop de la persona (foto completa) para el panel
    crop_persona = cv2.resize(crop, (160, 200), interpolation=cv2.INTER_AREA)
    # Thumbnail: parte superior (~35 %) para simular zona cara/cuerpo
    h_thumb = max(1, int(h_crop * 0.35))
    thumb = frame[y1 : y1 + h_thumb, x1:x2].copy()
    thumb = cv2.resize(thumb, (96, 112), interpolation=cv2.INTER_AREA)

    # Color superior e inferior: YOLO-pose (solo keypoints torso y piernas).
    # Si no hay puntos válidos, NO inventar el color: marcar como desconocido.
    zona_torso, zona_piernas = None, None
    if pose_model is not None:
        zona_torso, zona_piernas = get_torso_y_piernas_from_pose(crop, pose_model)

    if zona_torso is not None and zona_torso.size > 0:
        zona_torso_zoom = _zoom_region_for_color(zona_torso, min_side=160)
        color_sup = color_dominante_region(zona_torso_zoom)
        region_sup = zona_torso_zoom.copy()
    else:
        color_sup = {"hex": "#505050", "nombre": "desconocido"}
        region_sup = None

    if zona_piernas is not None and zona_piernas.size > 0:
        zona_piernas_zoom = _zoom_region_for_color(zona_piernas, min_side=160)
        color_inf = color_dominante_region(zona_piernas_zoom)
        region_inf = zona_piernas_zoom.copy()
    else:
        color_inf = {"hex": "#505050", "nombre": "desconocido"}
        region_inf = None

    complementos = []
    for nombre, acc_xyxy in comp_list:
        ax1, ay1, ax2, ay2 = int(acc_xyxy[0]), int(acc_xyxy[1]), int(acc_xyxy[2]), int(acc_xyxy[3])
        ax1, ay1 = max(0, ax1), max(0, ay1)
        ax2, ay2 = min(w_frame, ax2), min(h_frame, ay2)
        if ax2 > ax1 and ay2 > ay1:
            acc_crop = frame[ay1:ay2, ax1:ax2]
            acc_crop_zoom = _zoom_region_for_color(acc_crop, min_side=120)
            c = color_dominante_region(acc_crop_zoom)
            complementos.append({"nombre": nombre, "color": c["nombre"], "color_hex": c["hex"]})
        else:
            complementos.append({"nombre": nombre, "color": "no detectable", "color_hex": "#505050"})

    return {
        "thumbnail": thumb,
        "crop_persona": crop_persona,
        "color_superior": color_sup["nombre"],
        "color_superior_hex": color_sup["hex"],
        "color_inferior": color_inf["nombre"],
        "color_inferior_hex": color_inf["hex"],
        "complementos": complementos,
        "region_superior": region_sup,
        "region_inferior": region_inf,
    }


def dibujar_cajas(frame, person_boxes, analisis):
    """Dibuja cajas y textos de detección sobre el frame (modifica frame in-place)."""
    for i, ((x1, y1, x2, y2), _) in enumerate(person_boxes):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if i < len(analisis):
            a = analisis[i]
            comp_str = ", ".join(a["complementos"]) if a["complementos"] else "ninguno"
            texto = f"P{i+1} | Sup: {a['color_superior']} | Inf: {a['color_inferior']} | Comp: {comp_str}"
            cv2.putText(
                frame, texto,
                (x1, y1 - 10 if y1 > 25 else y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (0, 255, 0), 1, cv2.LINE_AA,
            )


# Colores BGR para bounding boxes
COLOR_PERSONA_NO_ID = (255, 0, 0)    # azul (por defecto)
COLOR_PERSONA_ID = (0, 255, 0)       # verde (identificado)
COLOR_PERRO = (255, 0, 255)          # morado

def dibujar_cajas_track(frame, person_boxes_with_id, track_identity=None, dog_boxes=None):
    """
    Dibuja cajas: personas azul (por defecto) o verde (identificado), perros morado.
    Si identificado: muestra nombre en vez de ID. Bbox y texto más grandes.
    """
    track_identity = track_identity or {}
    dog_boxes = dog_boxes or []
    thickness = 4
    font_scale = 1.0
    font_thickness = 2

    for (x1, y1, x2, y2), conf, tid in person_boxes_with_id:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        ident = track_identity.get(tid) if tid is not None else None
        if ident and ident.get("nombre"):
            color = COLOR_PERSONA_ID
            label = ident["nombre"]
        else:
            color = COLOR_PERSONA_NO_ID
            label = str(tid) if tid is not None else "?"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        txt_pos = (x1, y1 - 8 if y1 > 30 else y1 + 28)
        cv2.putText(
            frame, label,
            txt_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, color, font_thickness, cv2.LINE_AA,
        )

    for (x1, y1, x2, y2), conf in dog_boxes:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PERRO, thickness)
        cv2.putText(
            frame, "Perro",
            (x1, y1 - 8 if y1 > 30 else y1 + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, COLOR_PERRO, font_thickness, cv2.LINE_AA,
        )


def frame_bgr_a_photoimage(frame_bgr, max_ancho=960, max_alto=720):
    """Convierte un frame BGR (OpenCV) a ImageTk.PhotoImage para Tkinter, redimensionando si hace falta."""
    h, w = frame_bgr.shape[:2]
    if w > max_ancho or h > max_alto:
        escala = min(max_ancho / w, max_alto / h)
        nw, nh = int(w * escala), int(h * escala)
        frame_bgr = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    return ImageTk.PhotoImage(image=img)


def thumbnail_a_photoimage(thumb_bgr, ancho=96, alto=112):
    """Convierte thumbnail BGR (ya redimensionado o no) a PhotoImage para la tarjeta."""
    if thumb_bgr is None or thumb_bgr.size == 0:
        return None
    h, w = thumb_bgr.shape[:2]
    if w != ancho or h != alto:
        thumb_bgr = cv2.resize(thumb_bgr, (ancho, alto), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(image=Image.fromarray(rgb))


def crop_a_photoimage(crop_bgr, ancho=160, alto=200):
    """Convierte crop BGR a PhotoImage para el panel."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    h, w = crop_bgr.shape[:2]
    if w != ancho or h != alto:
        crop_bgr = cv2.resize(crop_bgr, (ancho, alto), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(image=Image.fromarray(rgb))


def crop_cara_a_photoimage(crop_bgr, tam=120):
    """Convierte crop de cara BGR a PhotoImage cuadrado para el panel."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    h, w = crop_bgr.shape[:2]
    if w != tam or h != tam:
        crop_bgr = cv2.resize(crop_bgr, (tam, tam), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(image=Image.fromarray(rgb))


def _llenar_contenido_tarjeta(card, track_id, info, photo_refs_list):
    """Llena el contenido de una tarjeta (reutilizable para crear y actualizar)."""
    card.configure(bg="#3d3d3d")

    # --------- Fila 0-1: cara + ID / Nombre ----------
    face_col = tk.Frame(card, bg="#3d3d3d")
    face_col.grid(row=0, column=0, rowspan=3, sticky="n", padx=(0, 10), pady=(0, 4))

    # Crop de la cara (solo rostro)
    crop_cara = info.get("crop_cara")
    img_face = None
    if crop_cara is not None:
        img_face = crop_cara_a_photoimage(crop_cara)
    else:
        crop_persona = info.get("crop_persona") or info.get("thumbnail")
        if crop_persona is not None:
            img_face = crop_a_photoimage(crop_persona, ancho=160, alto=200)
    if img_face is not None:
        photo_refs_list.append(img_face)
        tk.Label(face_col, image=img_face, bg="#3d3d3d").pack()
    else:
        tk.Label(face_col, text="[Sin imagen]", bg="#3d3d3d", fg="#888").pack()

    info_col = tk.Frame(card, bg="#3d3d3d")
    info_col.grid(row=0, column=1, sticky="w", pady=(0, 4))

    # ID: track_id o user_id si identificado
    user_id = info.get("user_id")
    user_name = info.get("user_nombre", "desconocido")
    id_txt = f"ID: {user_id}" if user_id is not None else f"Track ID: {track_id}"
    tk.Label(info_col, text=id_txt, font=("Segoe UI", 10, "bold"), fg="white", bg="#3d3d3d").pack(anchor="w")

    # Nombre (verde si está identificado)
    identificado = info.get("identificado", False)
    nombre_color = "#40c040" if identificado else "#ddd"
    tk.Label(info_col, text=f"Nombre: {user_name}", font=("Segoe UI", 10), fg=nombre_color, bg="#3d3d3d").pack(anchor="w")

    # Separador 1
    sep1 = tk.Frame(card, bg="#555555", height=1)
    sep1.grid(row=1, column=1, sticky="ew", pady=(6, 6))

    # --------- Bloque ropa superior ----------
    sup_row = tk.Frame(card, bg="#3d3d3d")
    sup_row.grid(row=2, column=1, sticky="w")

    # Mini foto de región superior
    foto_sup = info.get("region_superior")
    if foto_sup is not None:
        ph_sup = thumbnail_a_photoimage(foto_sup, ancho=80, alto=60)
        if ph_sup is not None:
            photo_refs_list.append(ph_sup)
            tk.Label(sup_row, image=ph_sup, bg="#3d3d3d").pack(side="left", padx=(0, 8))

    sup_text = tk.Frame(sup_row, bg="#3d3d3d")
    sup_text.pack(side="left")
    tk.Label(sup_text, text="Ropa superior", font=("Segoe UI", 9, "bold"), fg="#ccc", bg="#3d3d3d").pack(anchor="w")
    hex_sup = info.get("color_superior_hex") or nombre_color_a_hex(info.get("color_superior", "no detectable"))
    txt_sup = info.get("color_superior", "—")
    fila_sup = tk.Frame(sup_text, bg="#3d3d3d")
    fila_sup.pack(anchor="w")
    tk.Label(fila_sup, text="Color:", font=("Segoe UI", 9), fg="#ccc", bg="#3d3d3d").pack(side="left")
    tk.Label(fila_sup, text="", width=2, bg=hex_sup, relief="solid", borderwidth=1).pack(side="left", padx=4)
    tk.Label(fila_sup, text=f"{txt_sup} ({hex_sup})", font=("Segoe UI", 9), fg="white", bg="#3d3d3d").pack(side="left")

    # Separador 2
    sep2 = tk.Frame(card, bg="#555555", height=1)
    sep2.grid(row=3, column=1, sticky="ew", pady=(6, 6))

    # --------- Bloque ropa inferior ----------
    inf_row = tk.Frame(card, bg="#3d3d3d")
    inf_row.grid(row=4, column=1, sticky="w")

    foto_inf = info.get("region_inferior")
    if foto_inf is not None:
        ph_inf = thumbnail_a_photoimage(foto_inf, ancho=80, alto=60)
        if ph_inf is not None:
            photo_refs_list.append(ph_inf)
            tk.Label(inf_row, image=ph_inf, bg="#3d3d3d").pack(side="left", padx=(0, 8))

    inf_text = tk.Frame(inf_row, bg="#3d3d3d")
    inf_text.pack(side="left")
    tk.Label(inf_text, text="Ropa inferior", font=("Segoe UI", 9, "bold"), fg="#ccc", bg="#3d3d3d").pack(anchor="w")
    hex_inf = info.get("color_inferior_hex") or nombre_color_a_hex(info.get("color_inferior", "no detectable"))
    txt_inf = info.get("color_inferior", "—")
    fila_inf = tk.Frame(inf_text, bg="#3d3d3d")
    fila_inf.pack(anchor="w")
    tk.Label(fila_inf, text="Color:", font=("Segoe UI", 9), fg="#ccc", bg="#3d3d3d").pack(side="left")
    tk.Label(fila_inf, text="", width=2, bg=hex_inf, relief="solid", borderwidth=1).pack(side="left", padx=4)
    tk.Label(fila_inf, text=f"{txt_inf} ({hex_inf})", font=("Segoe UI", 9), fg="white", bg="#3d3d3d").pack(side="left")


def crear_tarjeta_persona(parent, track_id, info, photo_refs_list):
    """
    Crea una tarjeta independiente en el panel: foto, id, nombre, ropa superior/inferior.
    Devuelve el wrapper (para poder actualizarlo luego).
    """
    identificado = info.get("identificado", False)
    borde_color = "#00c040" if identificado else "#0070ff"
    wrapper = tk.Frame(
        parent,
        bg="#2b2b2b",
        pady=0,
        highlightbackground=borde_color,
        highlightcolor=borde_color,
        highlightthickness=2,
        bd=0,
    )
    # Colocar las tarjetas en fila, con espacio horizontal entre ellas
    wrapper.pack(side="left", padx=8, pady=10)
    card = tk.Frame(wrapper, bg="#3d3d3d", padx=12, pady=12)
    card.pack(fill="x")
    _llenar_contenido_tarjeta(card, track_id, info, photo_refs_list)
    return wrapper


def actualizar_tarjeta_persona(wrapper, track_id, info, photo_refs_list):
    """Reemplaza el contenido de una tarjeta existente (para mejor match en frame posterior)."""
    identificado = info.get("identificado", False)
    borde_color = "#00c040" if identificado else "#0070ff"
    wrapper.configure(highlightbackground=borde_color, highlightcolor=borde_color)
    for child in list(wrapper.winfo_children()):
        child.destroy()
    card = tk.Frame(wrapper, bg="#3d3d3d", padx=12, pady=12)
    card.pack(fill="x")
    _llenar_contenido_tarjeta(card, track_id, info, photo_refs_list)


def run_visor_tkinter(cap, model, fps, writer, args):
    """Vídeo con tracking (ByteTrack), IDs estables y panel lateral con tarjetas por persona."""
    path_video = Path(args.video)
    cada_n = 1  # procesar todos los frames, no saltar ninguno
    max_frames = getattr(args, "max_frames", None)
    imgsz = getattr(args, "imgsz", 320)
    playing = [True]
    frame_queue = Queue(maxsize=2)
    person_queue = Queue()  # ("new", track_id, info_dict)
    frames_procesados = [0]
    known_track_ids = set()
    # Mapa track_id -> identidad de usuario (id, nombre, score)
    track_identity = {}
    # Último frame en que intentamos reconocer por track_id (para espaciar reintentos)
    track_last_recog_frame = {}
    RECOG_RETRY_CADA_FRAMES = 5  # reintentar cada 5 frames (ahorrar CPU, ~6 intentos/seg)
    REID_THRESHOLD = 0.42  # similitud mínima para re-identificar misma persona con otro track_id
    # Cache para re-ID: user_id -> (embedding, identidad) cuando el tracker asigna nuevo ID
    embedding_by_user = {}
    identity_by_user = {}
    usuarios_registrados = getattr(args, "_usuarios_registrados", [])
    arcface_app = None
    debug_face = getattr(args, "debug_face", False)
    pose_model = getattr(args, "_pose_model", None)

    root = tk.Tk()
    root.title("Detección ropa — " + path_video.name)
    # Ajustar ventana a un tamaño fijo pero adaptado a la pantalla, para que siempre se vea el botón Cerrar
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    win_w = min(1600, int(screen_w * 0.9))
    win_h = min(900, int(screen_h * 0.9))
    root.geometry(f"{win_w}x{win_h}")
    root.minsize(win_w, win_h)
    root.resizable(False, False)
    root.configure(bg="#2b2b2b")
    # Tamaño máximo del área de vídeo dentro de la ventana
    video_max_w = int(win_w * 0.95)
    video_max_h = int(win_h * 0.6)

    # Contenedor principal: vídeo arriba, usuarios abajo
    main = tk.Frame(root, bg="#2b2b2b")
    main.pack(fill="both", expand=True, padx=4, pady=4)
    main.grid_rowconfigure(0, weight=3)  # vídeo
    main.grid_rowconfigure(1, weight=2)  # tarjetas
    main.grid_columnconfigure(0, weight=1)

    # ——— Parte superior: vídeo ———
    top = tk.Frame(main, bg="#2b2b2b")
    top.grid(row=0, column=0, sticky="nsew")
    lbl_frame = tk.Label(top, text="Frame: 0", font=("Segoe UI", 10), fg="white", bg="#2b2b2b")
    lbl_frame.pack(pady=4)
    lbl_video = tk.Label(top, bg="#1e1e1e")
    lbl_video.pack(fill="both", expand=True, padx=4, pady=2)
    photo_ref = [None]

    # ——— Parte inferior: panel con usuarios ———
    bottom = tk.Frame(main, bg="#2b2b2b")
    bottom.grid(row=1, column=0, sticky="nsew")
    tk.Label(bottom, text="Personas detectadas", font=("Segoe UI", 12, "bold"), fg="white", bg="#2b2b2b").pack(pady=(0, 8))

    # Contenedor scroll: canvas + scrollbar horizontal
    scroll_container = tk.Frame(bottom, bg="#2b2b2b")
    scroll_container.pack(fill="both", expand=True)
    canvas = tk.Canvas(scroll_container, bg="#2b2b2b", highlightthickness=0)
    scrollbar = tk.Scrollbar(scroll_container, orient="horizontal", command=canvas.xview)

    scrollable = tk.Frame(canvas, bg="#2b2b2b")
    scrollable_id = canvas.create_window((0, 0), window=scrollable, anchor="nw")

    def _on_frame_configure(event):
        # Ajustar región desplazable al contenido (en horizontal)
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _on_mousewheel(event):
        delta = 0
        if getattr(event, "num", None) == 5:
            delta = 1
        elif getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "delta", 0) != 0:
            # Ratón típico en Windows/Linux
            delta = -1 * int(event.delta / 120)
        if delta != 0:
            canvas.xview_scroll(delta, "units")

    scrollable.bind("<Configure>", _on_frame_configure)
    canvas.bind_all("<MouseWheel>", _on_mousewheel)
    canvas.bind_all("<Button-4>", _on_mousewheel)
    canvas.bind_all("<Button-5>", _on_mousewheel)
    canvas.configure(xscrollcommand=scrollbar.set)
    canvas.pack(side="top", fill="both", expand=True)
    scrollbar.pack(side="bottom", fill="x")
    card_photo_refs = []  # referencias a PhotoImage de las tarjetas
    card_by_user = {}    # user_id -> wrapper (identificados: una tarjeta por usuario)
    card_by_track = {}   # track_id -> wrapper (no identificados)

    def worker():
        nonlocal arcface_app
        frame_idx = 0
        last_boxes_with_id = []
        last_dog_boxes = []
        h_frame, w_frame = None, None
        while playing[0]:
            ret, frame = cap.read()
            if not ret:
                try:
                    frame_queue.put_nowait((None, -1))
                except Exception:
                    pass
                break
            if max_frames is not None and frame_idx >= max_frames:
                try:
                    frame_queue.put_nowait((None, frame_idx))
                except Exception:
                    pass
                break
            if h_frame is None:
                h_frame, w_frame = frame.shape[0], frame.shape[1]

            if frame_idx % cada_n == 0:
                person_boxes, accessory_boxes, comp_por_persona, dog_boxes = track_frame(
                    frame, model, imgsz=imgsz, tracker=getattr(args, "tracker", "botsort.yaml")
                )
                last_boxes_with_id = person_boxes
                last_dog_boxes = dog_boxes
                for i, (xyxy, conf, tid) in enumerate(person_boxes):
                    if tid is None:
                        continue

                    crop_arcface = get_face_crop_for_arcface(frame, xyxy, h_frame, w_frame)
                    if crop_arcface is None:
                        continue

                    identidad_actual = track_identity.get(tid)
                    last_recog = track_last_recog_frame.get(tid, -999)
                    es_nuevo = tid not in known_track_ids
                    hubo_mejor_match = False

                    # Si ya está identificado: no seguir ejecutando ArcFace
                    if identidad_actual and identidad_actual.get("nombre"):
                        debe_intentar = False
                    else:
                        debe_intentar = es_nuevo or (frame_idx - last_recog >= RECOG_RETRY_CADA_FRAMES)
                    if debe_intentar:
                        track_last_recog_frame[tid] = frame_idx
                        if arcface_app is None:
                            arcface_app = get_arcface_app()
                        identidad, has_face, edad_face, genero_face, emb = identificar_cara_arcface(
                            crop_arcface, usuarios_registrados or [], arcface_app,
                            debug=debug_face, frame_idx=frame_idx,
                        )
                        if not has_face:
                            if es_nuevo:
                                continue
                        else:
                            if identidad is not None:
                                score_nuevo = identidad.get("score", 0)
                                identidad["edad"] = edad_face
                                identidad["genero"] = genero_face
                                score_actual = (identidad_actual.get("score") or 0) if (identidad_actual and identidad_actual.get("nombre")) else -1
                                if score_nuevo > score_actual:
                                    track_identity[tid] = identidad
                                    identidad_actual = identidad
                                    hubo_mejor_match = True
                                    # Cache para re-ID cuando el tracker asigne otro ID a la misma persona
                                    uid = identidad.get("id")
                                    if uid is not None and emb is not None:
                                        embedding_by_user[uid] = emb.copy()
                                        identity_by_user[uid] = identidad.copy()
                            else:
                                # Sin match contra DB: intentar re-ID contra personas ya vistas en sesión
                                if emb is not None and embedding_by_user and (identidad_actual is None or not identidad_actual.get("nombre")):
                                    best_uid, best_sim = None, -1.0
                                    for uid, cached_emb in embedding_by_user.items():
                                        sim = float(np.dot(emb, cached_emb))
                                        if sim >= REID_THRESHOLD and sim > best_sim:
                                            best_sim = sim
                                            best_uid = uid
                                    if best_uid is not None:
                                        cached_id = identity_by_user.get(best_uid)
                                        if cached_id is not None:
                                            identidad = cached_id.copy()
                                            identidad["edad"] = edad_face
                                            identidad["genero"] = genero_face
                                            identidad["score"] = best_sim
                                            track_identity[tid] = identidad
                                            identidad_actual = identidad
                                            hubo_mejor_match = True
                                            embedding_by_user[best_uid] = emb.copy()
                                            identity_by_user[best_uid] = identidad.copy()
                                if identidad_actual is None or not identidad_actual.get("nombre"):
                                    track_identity[tid] = {"edad": edad_face, "genero": genero_face}
                                    identidad_actual = track_identity[tid]

                    if es_nuevo:
                        known_track_ids.add(tid)

                    identidad = track_identity.get(tid)
                    comp_list = comp_por_persona[i] if i < len(comp_por_persona) else []
                    info = extract_person_info(
                        frame, xyxy, comp_list, h_frame, w_frame, pose_model=pose_model
                    )

                    if identidad is not None and identidad.get("nombre"):
                        info["identificado"] = True
                        info["user_id"] = identidad.get("id")
                        info["user_nombre"] = identidad.get("nombre", "desconocido")
                        info["user_score"] = identidad.get("score")
                        info["edad"] = identidad.get("edad")
                        info["genero"] = identidad.get("genero")
                        info["pct_acierto"] = int(round((identidad.get("score") or 0) * 100))
                    else:
                        info["identificado"] = False
                        info["user_nombre"] = "desconocido"
                        info["edad"] = identidad.get("edad") if identidad else None
                        info["genero"] = identidad.get("genero") if identidad else None
                    info["crop_cara"] = crop_arcface.copy()

                    try:
                        if es_nuevo:
                            person_queue.put_nowait(("new", tid, info))
                        elif hubo_mejor_match:
                            person_queue.put_nowait(("update", tid, info))
                    except Exception:
                        pass

            frame_dibujo = frame.copy()
            dibujar_cajas_track(frame_dibujo, last_boxes_with_id, track_identity, last_dog_boxes)
            if writer is not None:
                writer.write(frame_dibujo)
            try:
                frame_queue.put_nowait((frame_dibujo, frame_idx))
            except Exception:
                pass
            frames_procesados[0] = frame_idx
            frame_idx += 1

    def poll_queue():
        if not playing[0]:
            return
        try:
            while True:
                item = frame_queue.get_nowait()
                frame_dibujo, idx = item
                if frame_dibujo is None:
                    lbl_frame.config(text=f"Frame: {idx} — Fin del vídeo" if idx >= 0 else "Fin del vídeo")
                    root.after(40, poll_queue)
                    return
                photo_ref[0] = frame_bgr_a_photoimage(
                    frame_dibujo,
                    max_ancho=video_max_w,
                    max_alto=video_max_h,
                )
                lbl_video.config(image=photo_ref[0])
                lbl_frame.config(text=f"Frame: {idx}")
        except Empty:
            pass
        try:
            while True:
                msg = person_queue.get_nowait()
                if msg[0] == "new":
                    _, track_id, info = msg
                    user_id = info.get("user_id") if info.get("identificado") else None
                    if user_id is not None and user_id in card_by_user:
                        # Mismo usuario ya mostrado (otro track_id): actualizar tarjeta existente, no crear duplicado
                        actualizar_tarjeta_persona(card_by_user[user_id], track_id, info, card_photo_refs)
                    elif user_id is not None:
                        wrapper = crear_tarjeta_persona(scrollable, track_id, info, card_photo_refs)
                        card_by_user[user_id] = wrapper
                    else:
                        wrapper = crear_tarjeta_persona(scrollable, track_id, info, card_photo_refs)
                        card_by_track[track_id] = wrapper
                    canvas.configure(scrollregion=canvas.bbox("all"))
                elif msg[0] == "update":
                    _, track_id, info = msg
                    user_id = info.get("user_id") if info.get("identificado") else None
                    if user_id is not None and user_id in card_by_user:
                        actualizar_tarjeta_persona(card_by_user[user_id], track_id, info, card_photo_refs)
                        if track_id in card_by_track:
                            card_by_track[track_id].destroy()
                            del card_by_track[track_id]
                    elif user_id is not None and track_id in card_by_track:
                        actualizar_tarjeta_persona(card_by_track[track_id], track_id, info, card_photo_refs)
                        card_by_user[user_id] = card_by_track.pop(track_id)
                    elif track_id in card_by_track:
                        actualizar_tarjeta_persona(card_by_track[track_id], track_id, info, card_photo_refs)
                    canvas.configure(scrollregion=canvas.bbox("all"))
        except Empty:
            pass
        root.after(40, poll_queue)

    def on_cerrar():
        playing[0] = False
        cap.release()  # El worker recibirá ret=False y saldrá limpio del loop
        th.join(timeout=2.0)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_cerrar)
    tk.Button(root, text="Cerrar", command=on_cerrar, font=("Segoe UI", 10), cursor="hand2").pack(pady=6)

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    root.after(50, poll_queue)
    root.mainloop()

    return frames_procesados[0]


def main():
    parser = argparse.ArgumentParser(
        description="Detección de ropa y complementos en vídeo (YOLO)."
    )
    parser.add_argument(
        "video",
        type=str,
        help="Nombre del vídeo (se concatena con PATH_VIDEOS) o ruta absoluta al fichero",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=YOLO_MODEL,
        help=f"Modelo YOLO (default: {YOLO_MODEL})",
    )
    parser.add_argument(
        "--mostrar",
        action="store_true",
        help="Mostrar ventana con el vídeo y las detecciones",
    )
    parser.add_argument(
        "--salida",
        type=str,
        default=None,
        help="Ruta de vídeo de salida con anotaciones (opcional)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Máximo de frames a procesar (útil para pruebas)",
    )
    parser.add_argument(
        "--cada-n-frames",
        type=int,
        default=12,
        help="Procesar detección cada N frames (default: 12, aprox. 1 por segundo a 12fps)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Tamaño de entrada YOLO en píxeles (320 más rápido en CPU, 640 más preciso; default: 320)",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="botsort.yaml",
        choices=["botsort.yaml", "bytetrack.yaml"],
        help="Tracker: botsort (mejor si la persona se gira) o bytetrack (default: botsort.yaml)",
    )
    parser.add_argument(
        "--pose-model",
        type=str,
        default=YOLO_MODEL_POSE,
        help=f"Modelo YOLO-pose para segmentar torso/inferior (default: {YOLO_MODEL_POSE})",
    )
    parser.add_argument(
        "--no-pose",
        action="store_true",
        help="Desactivar pose (usar mitad superior/inferior para color, sin segmentación)",
    )
    parser.add_argument(
        "--debug-face",
        action="store_true",
        help="Imprimir scores de similitud al intentar identificar caras (diagnóstico)",
    )
    args = parser.parse_args()

    # Si el argumento es una ruta absoluta, usarla tal cual; si no, concatenar con PATH_VIDEOS
    video_arg = args.video
    path_video = Path(video_arg)
    if not path_video.is_absolute():
        path_video = Path(PATH_VIDEOS) / video_arg
    if not path_video.exists():
        print(f"Error: no existe el fichero '{args.video}'", file=sys.stderr)
        sys.exit(1)

    # --- Carga del modelo principal YOLO: primero intentamos engine, luego .pt (GPU/CPU) ---
    ENGINES_DIR.mkdir(parents=True, exist_ok=True)
    model_stem = Path(args.model).stem
    engine_path = ENGINES_DIR / f"{model_stem}.engine"

    if engine_path.exists():
        print(f"Cargando modelo YOLO desde engine: '{engine_path}'")
        model = YOLO(str(engine_path))
    else:
        print(f"Cargando modelo YOLO desde pesos '{args.model}' en dispositivo '{DEVICE}'...")
        model = YOLO(args.model)
        try:
            model.to(DEVICE)
        except Exception:
            pass

    # --- Carga del modelo de pose (si procede): engine > .pt (GPU/CPU) ---
    args._pose_model = None
    if args.mostrar and not args.no_pose:
        pose_stem = Path(args.pose_model).stem
        pose_engine_path = ENGINES_DIR / f"{pose_stem}.engine"
        if pose_engine_path.exists():
            print(f"Cargando modelo YOLO-pose desde engine: '{pose_engine_path}'")
            args._pose_model = YOLO(str(pose_engine_path))
        else:
            print(f"Cargando modelo YOLO-pose desde pesos '{args.pose_model}' en dispositivo '{DEVICE}'...")
            args._pose_model = YOLO(args.pose_model)
            try:
                args._pose_model.to(DEVICE)
            except Exception:
                pass
    # Base de usuarios registrados (para identificación con ArcFace)
    args._usuarios_registrados = load_registered_users()
    cap = cv2.VideoCapture(str(path_video))
    if not cap.isOpened():
        print(f"Error: no se pudo abrir el vídeo '{args.video}'", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if args.salida:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.salida, fourcc, fps, (w, h))

    frame_idx = 0
    try:
        if args.mostrar:
            frame_idx = run_visor_tkinter(cap, model, fps, writer, args)
        else:
            cada_n = args.cada_n_frames
            last_boxes, last_analisis = [], []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if args.max_frames is not None and frame_idx >= args.max_frames:
                    break
                if frame_idx % cada_n == 0:
                    person_boxes, analisis = procesar_frame(frame, model, imgsz=args.imgsz)
                    last_boxes, last_analisis = person_boxes, analisis
                    if analisis:
                        print(f"\n--- Frame {frame_idx} ---")
                        for i, a in enumerate(analisis):
                            comp = ", ".join(a["complementos"]) if a["complementos"] else "ninguno"
                            print(
                                f"  Persona {i+1}: "
                                f"complementos=[{comp}] | "
                                f"parte superior (color)={a['color_superior']} | "
                                f"parte inferior (color)={a['color_inferior']}"
                            )
                dibujar_cajas(frame, last_boxes, last_analisis)
                if writer is not None:
                    writer.write(frame)
                frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    print(f"\nProcesados {frame_idx} frames.")


if __name__ == "__main__":
    main()
