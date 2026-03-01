#!/usr/bin/env python3
"""
Prueba de detección de ropa y complementos en vídeo.
- Detecta personas con YOLO.
- Para cada persona: complementos (mochila, bolso, paraguas, corbata, maleta),
  color de la parte superior e inferior.
- El tipo concreto de prenda (camiseta/abrigo, pantalón/falda) requeriría
  un modelo de moda; aquí se reporta solo la zona y el color.
"""

import argparse
import sys
import threading
from pathlib import Path
from queue import Empty, Queue

import cv2
import numpy as np
from PIL import Image, ImageTk
from sklearn.cluster import KMeans
from ultralytics import YOLO
import tkinter as tk

# Optimización CPU: limitar hilos de PyTorch (evita sobrecarga, suele mejorar throughput)
try:
    import torch
    torch.set_num_threads(min(4, torch.get_num_threads()))
except Exception:
    pass

# Cascade para detección de cara frontal (OpenCV). Carga perezosa.
_face_cascade = None

def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(path)
    return _face_cascade


def has_frontal_face(crop_bgr, min_size_ratio=0.2):
    """
    True si en el recorte (zona superior del cuerpo/cara) se detecta al menos una cara frontal.
    crop_bgr: recorte BGR (p. ej. parte superior de la bbox de la persona).
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return False
    cascade = _get_face_cascade()
    if cascade.empty():
        return False
    gris = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gris.shape[:2]
    min_side = int(min(w, h) * min_size_ratio)
    min_side = max(20, min_side)
    caras = cascade.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=4, minSize=(min_side, min_side))
    return len(caras) > 0


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


# COCO pose keypoints: 5=L_shoulder, 6=R_shoulder, 11=L_hip, 12=R_hip, 13=L_knee, 14=R_knee
KPT_SHOULDERS = (5, 6)
KPT_HIPS = (11, 12)
KPT_KNEES = (13, 14)


def get_torso_regions_from_pose(crop_bgr, pose_model, conf_min=0.25):
    """
    Usa YOLO-pose para definir torso (hombros→caderas) y parte inferior (caderas→rodillas).
    Devuelve (zona_torso_sup, zona_torso_inf) como recortes BGR, o (None, None) si falla.
    Excluye cara/pelo/brazos y enfoca la ropa.
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
        # xy: (n_persons, 17, 2), conf: (n_persons, 17)
        xy = kpts.xy[0].cpu().numpy() if len(kpts.xy) > 0 else None
        conf = kpts.conf[0].cpu().numpy() if len(kpts.conf) > 0 else None
        if xy is None or conf is None or xy.shape[0] < 13:
            return None, None
    except Exception:
        return None, None

    def get_points(indices):
        pts = []
        for i in indices:
            if i < len(xy) and i < len(conf) and conf[i] >= conf_min:
                pts.append((float(xy[i][0]), float(xy[i][1])))
        return pts

    shoulders = get_points(KPT_SHOULDERS)
    hips = get_points(KPT_HIPS)
    knees = get_points(KPT_KNEES)
    if not shoulders or not hips:
        return None, None

    margin = 0.05
    mw = int(w_crop * margin)
    mh = int(h_crop * margin)

    # Torso: hombros -> caderas (ropa superior)
    y_top = max(0, int(min(p[1] for p in shoulders)) - mh)
    y_bot = min(h_crop - 1, int(max(p[1] for p in hips)) + mh)
    all_x = [p[0] for p in shoulders + hips]
    x_left = max(0, int(min(all_x)) - mw)
    x_right = min(w_crop - 1, int(max(all_x)) + mw)
    if y_bot > y_top and x_right > x_left:
        zona_torso = crop_bgr[y_top:y_bot, x_left:x_right].copy()
    else:
        zona_torso = None

    # Parte inferior: caderas -> rodillas (ropa inferior)
    zona_inf = None
    if hips and knees:
        y_top_inf = max(0, int(min(p[1] for p in hips)) - mh)
        y_bot_inf = min(h_crop - 1, int(max(p[1] for p in knees)) + mh)
        all_x_inf = [p[0] for p in hips + knees]
        x_left_inf = max(0, int(min(all_x_inf)) - mw)
        x_right_inf = min(w_crop - 1, int(max(all_x_inf)) + mw)
        if y_bot_inf > y_top_inf and x_right_inf > x_left_inf:
            zona_inf = crop_bgr[y_top_inf:y_bot_inf, x_left_inf:x_right_inf].copy()

    return zona_torso, zona_inf


# Clases COCO: persona y complementos vestimentarios
COCO_PERSON = 0
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
                if iou > mejor_iou or (iou == 0 and centro_en_box(acc_cx, acc_cy, p_xyxy)):
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
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        xyxy = boxes.xyxy[i].cpu().numpy()
        conf = float(boxes.conf[i].item())
        if cls_id == COCO_PERSON:
            person_boxes.append((xyxy, conf))
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
    ByteTrack solo usa posición/movimiento y puede asignar otro ID al mismo al cambiar la apariencia.
    Returns: (person_boxes_with_id, accessory_boxes, complementos_por_persona).
    """
    results = model.track(
        frame, imgsz=imgsz, persist=True, verbose=False,
        tracker=tracker,
    )
    if not results:
        return [], [], []
    r = results[0]
    boxes = r.boxes
    if boxes is None:
        return [], [], []
    ids = boxes.id  # puede ser None si no hay tracking

    person_boxes = []
    accessory_boxes = []
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        xyxy = boxes.xyxy[i].cpu().numpy()
        conf = float(boxes.conf[i].item())
        tid = int(ids[i].item()) if ids is not None and i < len(ids) else None
        if cls_id == COCO_PERSON:
            person_boxes.append((xyxy, conf, tid))
        elif cls_id in COCO_ACCESSORY_IDS:
            accessory_boxes.append((xyxy, conf, cls_id))

    person_boxes.sort(key=lambda x: (x[0][1] + x[0][3]) / 2)
    # Asociar complementos (por índice de persona)
    comp_por_persona = asociar_complementos_a_personas(
        [(b[0], b[1]) for b in person_boxes], accessory_boxes
    )
    return person_boxes, accessory_boxes, comp_por_persona


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
            "color_superior": "no detectable",
            "color_inferior": "no detectable",
            "complementos": [],
            "region_superior": None,
            "region_inferior": None,
        }
    crop = frame[y1:y2, x1:x2]
    h_crop, w_crop = crop.shape[:2]
    # Thumbnail: parte superior (~35 %) para simular zona cara/cuerpo
    h_thumb = max(1, int(h_crop * 0.35))
    thumb = frame[y1 : y1 + h_thumb, x1:x2].copy()
    thumb = cv2.resize(thumb, (96, 112), interpolation=cv2.INTER_AREA)

    # Color superior e inferior: usar pose si está disponible (segmentación más precisa)
    zona_torso, zona_inf_pose = None, None
    if pose_model is not None:
        zona_torso, zona_inf_pose = get_torso_regions_from_pose(crop, pose_model)

    if zona_torso is not None and zona_torso.size > 0:
        color_sup = color_dominante_region(zona_torso)
        region_sup = zona_torso.copy()
    else:
        mitad = h_crop // 2
        zona_sup = crop[:mitad, :]
        color_sup = color_dominante_region(zona_sup)
        region_sup = zona_sup.copy()

    if zona_inf_pose is not None and zona_inf_pose.size > 0:
        color_inf = color_dominante_region(zona_inf_pose)
        region_inf = zona_inf_pose.copy()
    else:
        mitad = h_crop // 2
        zona_inf = crop[mitad:, :]
        color_inf = color_dominante_region(zona_inf)
        region_inf = zona_inf.copy()

    complementos = []
    for nombre, acc_xyxy in comp_list:
        ax1, ay1, ax2, ay2 = int(acc_xyxy[0]), int(acc_xyxy[1]), int(acc_xyxy[2]), int(acc_xyxy[3])
        ax1, ay1 = max(0, ax1), max(0, ay1)
        ax2, ay2 = min(w_frame, ax2), min(h_frame, ay2)
        if ax2 > ax1 and ay2 > ay1:
            acc_crop = frame[ay1:ay2, ax1:ax2]
            c = color_dominante_region(acc_crop)
            complementos.append({"nombre": nombre, "color": c["nombre"], "color_hex": c["hex"]})
        else:
            complementos.append({"nombre": nombre, "color": "no detectable", "color_hex": "#505050"})

    return {
        "thumbnail": thumb,
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


def dibujar_cajas_track(frame, person_boxes_with_id):
    """Dibuja cajas y track_id. person_boxes_with_id = [(xyxy, conf, track_id), ...]."""
    for (x1, y1, x2, y2), conf, tid in person_boxes_with_id:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = str(tid) if tid is not None else "?"
        cv2.putText(
            frame, f"ID {label}",
            (x1, y1 - 6 if y1 > 20 else y1 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 255, 0), 1, cv2.LINE_AA,
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


def crear_tarjeta_persona(parent, track_id, info, photo_refs_list):
    """
    Crea una tarjeta en el panel lateral: thumbnail, ID, ropa sup/inf con color, complementos, Identificado: No.
    photo_refs_list: lista donde guardar referencias a PhotoImage para que no se pierdan.
    """
    card = tk.Frame(parent, bg="#3d3d3d", padx=8, pady=8, relief="ridge", borderwidth=1)
    card.pack(fill="x", padx=4, pady=4)

    # Thumbnail (zona cara/arriba)
    if info.get("thumbnail") is not None:
        ph = thumbnail_a_photoimage(info["thumbnail"])
        if ph is not None:
            photo_refs_list.append(ph)
            tk.Label(card, image=ph, bg="#3d3d3d").pack(pady=(0, 4))
    else:
        tk.Label(card, text="[Sin imagen]", bg="#3d3d3d", fg="#888").pack(pady=(0, 4))

    # ID
    tk.Label(card, text=f"ID: {track_id}", font=("Segoe UI", 11, "bold"), fg="white", bg="#3d3d3d").pack(anchor="w")

    # Ropa superior: [cuadro color] texto (nombre + hex)
    row_sup = tk.Frame(card, bg="#3d3d3d")
    row_sup.pack(fill="x", pady=2)
    tk.Label(row_sup, text="Ropa superior:", font=("Segoe UI", 9), fg="#ccc", bg="#3d3d3d").pack(side="left")
    hex_sup = info.get("color_superior_hex") or nombre_color_a_hex(info.get("color_superior", "no detectable"))
    tk.Label(row_sup, text="", width=2, bg=hex_sup, relief="solid", borderwidth=1).pack(side="left", padx=4)
    txt_sup = info.get("color_superior", "—")
    if info.get("color_superior_hex"):
        txt_sup += f" ({info['color_superior_hex']})"
    tk.Label(row_sup, text=txt_sup, font=("Segoe UI", 9), fg="white", bg="#3d3d3d").pack(side="left")

    # Segmentación parte superior (imagen de la región usada)
    if info.get("region_superior") is not None:
        try:
            ph_sup = thumbnail_a_photoimage(info["region_superior"], ancho=100, alto=70)
            if ph_sup is not None:
                photo_refs_list.append(ph_sup)
                tk.Label(card, text="Segmentación superior (región usada):", font=("Segoe UI", 8), fg="#888", bg="#3d3d3d").pack(anchor="w")
                tk.Label(card, image=ph_sup, bg="#3d3d3d", relief="solid", borderwidth=1).pack(anchor="w", pady=(0, 4))
        except Exception:
            pass

    # Ropa inferior
    row_inf = tk.Frame(card, bg="#3d3d3d")
    row_inf.pack(fill="x", pady=2)
    tk.Label(row_inf, text="Ropa inferior:", font=("Segoe UI", 9), fg="#ccc", bg="#3d3d3d").pack(side="left")
    hex_inf = info.get("color_inferior_hex") or nombre_color_a_hex(info.get("color_inferior", "no detectable"))
    tk.Label(row_inf, text="", width=2, bg=hex_inf, relief="solid", borderwidth=1).pack(side="left", padx=4)
    txt_inf = info.get("color_inferior", "—")
    if info.get("color_inferior_hex"):
        txt_inf += f" ({info['color_inferior_hex']})"
    tk.Label(row_inf, text=txt_inf, font=("Segoe UI", 9), fg="white", bg="#3d3d3d").pack(side="left")

    # Segmentación parte inferior (imagen de la región usada)
    if info.get("region_inferior") is not None:
        try:
            ph_inf = thumbnail_a_photoimage(info["region_inferior"], ancho=100, alto=70)
            if ph_inf is not None:
                photo_refs_list.append(ph_inf)
                tk.Label(card, text="Segmentación inferior (región usada):", font=("Segoe UI", 8), fg="#888", bg="#3d3d3d").pack(anchor="w")
                tk.Label(card, image=ph_inf, bg="#3d3d3d", relief="solid", borderwidth=1).pack(anchor="w", pady=(0, 4))
        except Exception:
            pass

    # Complementos: tipo + [cuadro color] color
    comps = info.get("complementos", [])
    if comps:
        tk.Label(card, text="Complementos:", font=("Segoe UI", 9), fg="#ccc", bg="#3d3d3d").pack(anchor="w")
        for c in comps:
            fcomp = tk.Frame(card, bg="#3d3d3d")
            fcomp.pack(fill="x", padx=(12, 0))
            tk.Label(fcomp, text=c.get("nombre", ""), font=("Segoe UI", 9), fg="white", bg="#3d3d3d").pack(side="left")
            hex_c = c.get("color_hex") or nombre_color_a_hex(c.get("color", "no detectable"))
            tk.Label(fcomp, text="", width=2, bg=hex_c, relief="solid", borderwidth=1).pack(side="left", padx=4)
            tk.Label(fcomp, text=c.get("color", "—"), font=("Segoe UI", 9), fg="#aaa", bg="#3d3d3d").pack(side="left")
    else:
        tk.Label(card, text="Complementos: ninguno", font=("Segoe UI", 9), fg="#888", bg="#3d3d3d").pack(anchor="w")

    # Identificado: No
    tk.Label(card, text="Identificado: No", font=("Segoe UI", 9), fg="#f0a030", bg="#3d3d3d").pack(anchor="w", pady=(4, 0))
    return card


def run_visor_tkinter(cap, model, fps, writer, args):
    """Vídeo con tracking (ByteTrack), IDs estables y panel lateral con tarjetas por persona."""
    path_video = Path(args.video)
    cada_n = getattr(args, "cada_n_frames", 12)
    max_frames = getattr(args, "max_frames", None)
    imgsz = getattr(args, "imgsz", 320)
    playing = [True]
    frame_queue = Queue(maxsize=2)
    person_queue = Queue()  # ("new", track_id, info_dict)
    frames_procesados = [0]
    known_track_ids = set()

    root = tk.Tk()
    root.title("Detección ropa — " + path_video.name)
    root.geometry("1280x750")
    root.configure(bg="#2b2b2b")

    # Contenedor principal: izquierda (vídeo) + derecha (panel)
    main = tk.Frame(root, bg="#2b2b2b")
    main.pack(fill="both", expand=True, padx=4, pady=4)

    # ——— Izquierda: vídeo ———
    left = tk.Frame(main, bg="#2b2b2b")
    left.pack(side="left", fill="both", expand=True)
    lbl_frame = tk.Label(left, text="Frame: 0", font=("Segoe UI", 10), fg="white", bg="#2b2b2b")
    lbl_frame.pack(pady=4)
    lbl_video = tk.Label(left, bg="#1e1e1e")
    lbl_video.pack(padx=4, pady=2)
    photo_ref = [None]

    # ——— Derecha: panel con scroll ———
    right = tk.Frame(main, bg="#2b2b2b", width=320)
    right.pack(side="right", fill="y", padx=(8, 0))
    tk.Label(right, text="Personas detectadas", font=("Segoe UI", 11, "bold"), fg="white", bg="#2b2b2b").pack(pady=(0, 4))

    canvas = tk.Canvas(right, bg="#2b2b2b", highlightthickness=0)
    scrollbar = tk.Scrollbar(right, orient="vertical", command=canvas.yview)
    scrollable = tk.Frame(canvas, bg="#2b2b2b")
    scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    def _on_mousewheel(event):
        if getattr(event, "num", None) == 5:
            canvas.yview_scroll(1, "units")
        elif getattr(event, "num", None) == 4:
            canvas.yview_scroll(-1, "units")
        elif getattr(event, "delta", 0) != 0:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    canvas.bind_all("<MouseWheel>", _on_mousewheel)
    canvas.bind_all("<Button-4>", _on_mousewheel)
    canvas.bind_all("<Button-5>", _on_mousewheel)
    canvas.create_window((0, 0), window=scrollable, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    card_photo_refs = []  # referencias a PhotoImage de las tarjetas

    def worker():
        frame_idx = 0
        last_boxes_with_id = []
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
                person_boxes, accessory_boxes, comp_por_persona = track_frame(
                    frame, model, imgsz=imgsz, tracker=getattr(args, "tracker", "botsort.yaml")
                )
                last_boxes_with_id = person_boxes
                for i, (xyxy, conf, tid) in enumerate(person_boxes):
                    if tid is not None and tid not in known_track_ids:
                        crop_cara = get_upper_crop(frame, xyxy, h_frame, w_frame)
                        if crop_cara is not None and has_frontal_face(crop_cara):
                            known_track_ids.add(tid)
                            comp_list = comp_por_persona[i] if i < len(comp_por_persona) else []
                            pose_model = getattr(args, "_pose_model", None)
                            info = extract_person_info(
                                frame, xyxy, comp_list, h_frame, w_frame, pose_model=pose_model
                            )
                            try:
                                person_queue.put_nowait(("new", tid, info))
                            except Exception:
                                pass

            frame_dibujo = frame.copy()
            dibujar_cajas_track(frame_dibujo, last_boxes_with_id)
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
                photo_ref[0] = frame_bgr_a_photoimage(frame_dibujo)
                lbl_video.config(image=photo_ref[0])
                lbl_frame.config(text=f"Frame: {idx} (cada {cada_n} frames)")
        except Empty:
            pass
        try:
            while True:
                msg = person_queue.get_nowait()
                if msg[0] == "new":
                    _, track_id, info = msg
                    crear_tarjeta_persona(scrollable, track_id, info, card_photo_refs)
                    canvas.configure(scrollregion=canvas.bbox("all"))
        except Empty:
            pass
        root.after(40, poll_queue)

    def on_cerrar():
        playing[0] = False
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
        help="Ruta al fichero de vídeo",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Modelo YOLO (default: yolov8n.pt)",
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
        default="yolov8n-pose.pt",
        help="Modelo YOLO-pose para segmentar torso/inferior (default: yolov8n-pose.pt)",
    )
    parser.add_argument(
        "--no-pose",
        action="store_true",
        help="Desactivar pose (usar mitad superior/inferior para color, sin segmentación)",
    )
    args = parser.parse_args()

    path_video = Path(args.video)
    if not path_video.exists():
        print(f"Error: no existe el fichero '{args.video}'", file=sys.stderr)
        sys.exit(1)

    print("Cargando modelo YOLO...")
    model = YOLO(args.model)
    args._pose_model = None
    if args.mostrar and not args.no_pose:
        print("Cargando modelo YOLO-pose para segmentación de torso/color...")
        args._pose_model = YOLO(args.pose_model)
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
