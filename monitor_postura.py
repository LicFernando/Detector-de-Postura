import os
import sys
import time
import math
import threading
from datetime import datetime
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- 1. RUTAS RELATIVAS Y PARÁMETROS DEL SISTEMA ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "pose_landmarker_lite.task"
MUESTRAS_NORMAL_DIR = BASE_DIR / "muestras" / "normal"
MUESTRAS_ENCORVADO_DIR = BASE_DIR / "muestras" / "encorvado"
REPORTES_DIR = BASE_DIR / "reportes"

IDX_NARIZ = 0
IDX_OREJA_IZQ = 7
IDX_OREJA_DER = 8
IDX_HOMBRO_IZQ = 11
IDX_HOMBRO_DER = 12

FRAMES_CONFIRMACION = 15
SEGUNDOS_SOSTENIDOS_ALERTA = 4.0
COOLDOWN_ALERTA_SEG = 25.0
VENTANA_SPAM_SEG = 300.0
MAX_ALERTAS_EN_VENTANA = 3
DURACION_POPUP_MS = 4500


# --- 2. EXTRACCIÓN GEOMÉTRICA Y CALIBRACIÓN CON IMÁGENES DE MUESTRA ---
def calcular_metricas_postura(landmarks):
    try:
        nariz = landmarks[IDX_NARIZ]
        oreja_izq = landmarks[IDX_OREJA_IZQ]
        oreja_der = landmarks[IDX_OREJA_DER]
        hombro_izq = landmarks[IDX_HOMBRO_IZQ]
        hombro_der = landmarks[IDX_HOMBRO_DER]
    except (IndexError, TypeError):
        return None

    if hombro_izq.visibility < 0.5 or hombro_der.visibility < 0.5:
        return None

    hombro_x = (hombro_izq.x + hombro_der.x) / 2.0
    hombro_y = (hombro_izq.y + hombro_der.y) / 2.0

    if oreja_izq.visibility >= 0.5 and oreja_der.visibility >= 0.5:
        cabeza_x = (oreja_izq.x + oreja_der.x) / 2.0
        cabeza_y = (oreja_izq.y + oreja_der.y) / 2.0
    else:
        cabeza_x, cabeza_y = nariz.x, nariz.y

    # Normalización por ancho de hombros para compensar la distancia a la cámara
    ancho_hombros = math.hypot(hombro_izq.x - hombro_der.x, hombro_izq.y - hombro_der.y)
    if ancho_hombros < 1e-4:
        return None

    dx = cabeza_x - hombro_x
    dy = hombro_y - cabeza_y

    # Ángulo cervical respecto a la vertical y caída vertical normalizada
    angulo_grados = math.degrees(math.atan2(abs(dx), max(dy, 1e-4)))
    caida_vertical = (hombro_y - nariz.y) / ancho_hombros

    puntos = {
        "nariz": (nariz.x, nariz.y),
        "oreja_izq": (oreja_izq.x, oreja_izq.y),
        "oreja_der": (oreja_der.x, oreja_der.y),
        "hombro_izq": (hombro_izq.x, hombro_izq.y),
        "hombro_der": (hombro_der.x, hombro_der.y),
        "centro_hombros": (hombro_x, hombro_y),
        "centro_cabeza": (cabeza_x, cabeza_y),
    }
    return angulo_grados, caida_vertical, puntos


def procesar_imagenes_referencia():
    os.makedirs(MUESTRAS_NORMAL_DIR, exist_ok=True)
    os.makedirs(MUESTRAS_ENCORVADO_DIR, exist_ok=True)

    # Márgenes y línea base predeterminada calibrados para cámaras de laptop
    margen_leve_ang = 8.0
    margen_critico_ang = 16.0
    margen_leve_caida = 0.15
    margen_critico_caida = 0.28
    base_ang_default = 12.0
    base_caida_default = 0.52

    if not MODEL_PATH.exists():
        return {
            "delta_ang_leve": margen_leve_ang,
            "delta_ang_critico": margen_critico_ang,
            "delta_caida_leve": margen_leve_caida,
            "delta_caida_critico": margen_critico_caida,
            "base_ang_default": base_ang_default,
            "base_caida_default": base_caida_default,
        }

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
    )

    extensiones = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def extraer_de_carpeta(carpeta, landmarker):
        angulos, caidas = [], []
        for archivo in carpeta.iterdir():
            if archivo.suffix.lower() in extensiones:
                img_bgr = cv2.imread(str(archivo))
                if img_bgr is None:
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                res = landmarker.detect(mp_img)
                if res.pose_landmarks:
                    metricas = calcular_metricas_postura(res.pose_landmarks[0])
                    if metricas:
                        angulos.append(metricas[0])
                        caidas.append(metricas[1])
        return angulos, caidas

    # Procesa imágenes locales para ajustar los márgenes de encorvamiento
    with PoseLandmarker.create_from_options(options) as landmarker:
        ang_norm, caida_norm = extraer_de_carpeta(MUESTRAS_NORMAL_DIR, landmarker)
        ang_enc, caida_enc = extraer_de_carpeta(MUESTRAS_ENCORVADO_DIR, landmarker)

    base_ang = float(np.mean(ang_norm)) if ang_norm else base_ang_default
    base_caida = float(np.mean(caida_norm)) if caida_norm else base_caida_default

    if ang_norm and ang_enc:
        diff_ang = max(6.0, float(np.mean(ang_enc) - np.mean(ang_norm)))
        margen_leve_ang = round(diff_ang * 0.55, 2)
        margen_critico_ang = round(diff_ang, 2)

    if caida_norm and caida_enc:
        diff_caida = max(0.08, float(np.mean(caida_norm) - np.mean(caida_enc)))
        margen_leve_caida = round(diff_caida * 0.55, 3)
        margen_critico_caida = round(diff_caida, 3)

    print(f"[CALIBRACIÓN BASE] Muestras -> Normal: {len(ang_norm)}, Encorvado: {len(ang_enc)}")

    return {
        "delta_ang_leve": margen_leve_ang,
        "delta_ang_critico": margen_critico_ang,
        "delta_caida_leve": margen_leve_caida,
        "delta_caida_critico": margen_critico_caida,
        "base_ang_default": base_ang,
        "base_caida_default": base_caida,
    }


# --- 3. SISTEMA DE ALERTAS SILENCIOSAS Y FILTRO ANTI-SPAM ---
class GestorAlertasSilenciosas:
    def __init__(self):
        self.ultima_alerta_ts = 0.0
        self.historial_alertas = deque()
        self.popup_activo = False
        self.alertas_lanzadas = 0
        self.alertas_omitidas_spam = 0
        self._lock = threading.Lock()

    def intentar_lanzar_alerta(self, nivel_postura, desviacion_ang):
        ahora = time.time()

        with self._lock:
            while self.historial_alertas and (ahora - self.historial_alertas[0] > VENTANA_SPAM_SEG):
                self.historial_alertas.popleft()

            # Filtro anti-spam: evita superposición, respeta cooldown y límite por ventana
            if self.popup_activo:
                return False
            if (ahora - self.ultima_alerta_ts) < COOLDOWN_ALERTA_SEG:
                self.alertas_omitidas_spam += 1
                return False
            if len(self.historial_alertas) >= MAX_ALERTAS_EN_VENTANA:
                self.alertas_omitidas_spam += 1
                return False

            self.ultima_alerta_ts = ahora
            self.historial_alertas.append(ahora)
            self.alertas_lanzadas += 1
            self.popup_activo = True

        # Ejecución en hilo secundario para no congelar el video de OpenCV
        hilo = threading.Thread(
            target=self._mostrar_ventana_tk,
            args=(nivel_postura, desviacion_ang),
            daemon=True,
        )
        hilo.start()
        return True

    def _mostrar_ventana_tk(self, nivel_postura, desviacion_ang):
        try:
            root = tk.Tk()
            root.overrideredirect(True)
            root.attributes("-topmost", True)
            root.configure(bg="#1E1E2E")

            ancho_win, alto_win = 360, 105
            pos_x = root.winfo_screenwidth() - ancho_win - 24
            pos_y = root.winfo_screenheight() - alto_win - 64
            root.geometry(f"{ancho_win}x{alto_win}+{pos_x}+{pos_y}")

            color_acento = "#F38BA8" if "Crítico" in nivel_postura else "#FAB387"

            borde = tk.Frame(root, bg=color_acento, padx=2, pady=2)
            borde.pack(fill="both", expand=True)

            contenedor = tk.Frame(borde, bg="#1E1E2E", padx=14, pady=10)
            contenedor.pack(fill="both", expand=True)

            lbl_titulo = tk.Label(
                contenedor,
                text=f"Aviso de Postura ({nivel_postura})",
                font=("Segoe UI", 10, "bold"),
                fg=color_acento,
                bg="#1E1E2E",
                anchor="w",
            )
            lbl_titulo.pack(fill="x")

            lbl_cuerpo = tk.Label(
                contenedor,
                text=f"Desviación detectada: +{desviacion_ang:.1f}°.\nEndereza tu espalda y alinea tus hombros.",
                font=("Segoe UI", 9),
                fg="#CDD6F4",
                bg="#1E1E2E",
                justify="left",
                anchor="w",
            )
            lbl_cuerpo.pack(fill="x", pady=(4, 0))

            lbl_cerrar = tk.Label(
                contenedor,
                text="Clic para cerrar",
                font=("Segoe UI", 7),
                fg="#7F849C",
                bg="#1E1E2E",
                anchor="e",
            )
            lbl_cerrar.pack(fill="x")

            def cerrar(_event=None):
                try:
                    root.destroy()
                except tk.TclError:
                    pass

            for widget in (root, borde, contenedor, lbl_titulo, lbl_cuerpo, lbl_cerrar):
                widget.bind("<Button-1>", cerrar)

            root.after(DURACION_POPUP_MS, cerrar)
            root.mainloop()
        except Exception:
            pass
        finally:
            with self._lock:
                self.popup_activo = False


# --- 4. HISTORIAL ESTADÍSTICO Y EXPORTACIÓN DE REPORTE Y GRÁFICA ---
class HistorialPostura:
    def __init__(self):
        self.inicio_sesion = datetime.now()
        self.eventos = []
        self.estado_actual = "Normal"
        self.inicio_estado_ts = time.time()
        self.tiempo_por_estado = {
            "Normal": 0.0,
            "Encorvamiento Leve": 0.0,
            "Encorvamiento Crítico": 0.0,
        }
        self.conteo_encorvamientos = 0

    def registrar_transicion(self, nuevo_estado, angulo, desviacion):
        ahora = time.time()
        duracion_previa = ahora - self.inicio_estado_ts

        if self.estado_actual in self.tiempo_por_estado:
            self.tiempo_por_estado[self.estado_actual] += duracion_previa

        if nuevo_estado != self.estado_actual:
            if "Encorvamiento" in nuevo_estado and self.estado_actual == "Normal":
                self.conteo_encorvamientos += 1

            self.eventos.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "estado_anterior": self.estado_actual,
                "duracion_anterior_seg": round(duracion_previa, 2),
                "nuevo_estado": nuevo_estado,
                "angulo_actual": round(angulo, 1),
                "desviacion_deg": round(desviacion, 1),
                "frecuencia_encorvamientos": self.conteo_encorvamientos,
            })
            self.estado_actual = nuevo_estado
            self.inicio_estado_ts = ahora

    def finalizar_y_exportar(self, gestor_alertas, calibracion_usuario_activa):
        ahora = time.time()
        duracion_final = ahora - self.inicio_estado_ts
        if self.estado_actual in self.tiempo_por_estado:
            self.tiempo_por_estado[self.estado_actual] += duracion_final

        fin_sesion = datetime.now()
        duracion_total = sum(self.tiempo_por_estado.values())
        marca_tiempo_archivo = fin_sesion.strftime("%Y%m%d_%H%M%S")

        os.makedirs(REPORTES_DIR, exist_ok=True)
        ruta_reporte = REPORTES_DIR / f"reporte_postura_{marca_tiempo_archivo}.txt"
        ruta_grafica = REPORTES_DIR / f"grafica_postura_{marca_tiempo_archivo}.png"

        lineas = [
            "====================================================================",
            "           REPORTE ESTADÍSTICO DE SESIÓN - MONITOR DE POSTURA       ",
            "====================================================================",
            f"Inicio de sesión          : {self.inicio_sesion.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Fin de sesión             : {fin_sesion.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Duración total monitoreada: {duracion_total:.1f} segundos",
            f"Calibración personalizada : {'Sí (Línea base de usuario)' if calibracion_usuario_activa else 'No (Valores por defecto)'}",
            "--------------------------------------------------------------------",
            "RESUMEN DE TIEMPOS POR ESTADO:",
        ]

        for estado, segs in self.tiempo_por_estado.items():
            pct = (segs / duracion_total * 100.0) if duracion_total > 0 else 0.0
            lineas.append(f"  - {estado:<24}: {segs:>6.1f} s ({pct:>5.1f}%)")

        lineas.extend([
            "--------------------------------------------------------------------",
            "ESTADÍSTICAS DE EVENTOS Y ALERTAS:",
            f"  - Episodios de encorvamiento iniciados : {self.conteo_encorvamientos}",
            f"  - Total de cambios de estado           : {len(self.eventos)}",
            f"  - Alertas silenciosas mostradas        : {gestor_alertas.alertas_lanzadas}",
            f"  - Alertas omitidas por filtro anti-spam: {gestor_alertas.alertas_omitidas_spam}",
            "--------------------------------------------------------------------",
            "HISTORIAL CRONOLÓGICO DE CAMBIOS DE POSTURA:",
        ])

        if not self.eventos:
            lineas.append("  (Sin cambios de postura registrados durante la sesión)")
        else:
            for ev in self.eventos:
                lineas.append(
                    f"  [{ev['timestamp']}] {ev['estado_anterior']} ({ev['duracion_anterior_seg']}s) "
                    f"-> {ev['nuevo_estado']} | Ángulo: {ev['angulo_actual']}° "
                    f"(+{ev['desviacion_deg']}°) | Acumulado: {ev['frecuencia_encorvamientos']}"
                )

        lineas.append("====================================================================")
        contenido = "\n".join(lineas)

        with open(ruta_reporte, "w", encoding="utf-8") as f:
            f.write(contenido)

        self._exportar_grafica(ruta_grafica, duracion_total, gestor_alertas)

        print("\n" + contenido)
        print(f"\n[INFO] Reporte de texto guardado en : {ruta_reporte}")
        print(f"[INFO] Gráfica estadística guardada en: {ruta_grafica}")

    def _exportar_grafica(self, ruta_grafica, duracion_total, gestor_alertas):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor("#F8F9FA")

        estados = ["Normal", "Encorvamiento\nLeve", "Encorvamiento\nCrítico"]
        tiempos = [
            self.tiempo_por_estado["Normal"],
            self.tiempo_por_estado["Encorvamiento Leve"],
            self.tiempo_por_estado["Encorvamiento Crítico"],
        ]
        colores = ["#2ECC71", "#F39C12", "#E74C3C"]

        barras = ax1.bar(estados, tiempos, color=colores, width=0.55, edgecolor="#2C3E50")
        ax1.set_title("Distribución de Tiempo por Postura", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Tiempo (segundos)")
        ax1.grid(axis="y", linestyle="--", alpha=0.4)
        ax1.set_ylim(0, max(tiempos + [1.0]) * 1.22)

        for barra, segs in zip(barras, tiempos):
            pct = (segs / duracion_total * 100.0) if duracion_total > 0 else 0.0
            ax1.text(
                barra.get_x() + barra.get_width() / 2.0,
                barra.get_height() + (max(tiempos + [1.0]) * 0.03),
                f"{segs:.1f}s\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        if self.eventos:
            horas = [ev["timestamp"].split(" ")[1] for ev in self.eventos]
            angulos = [ev["angulo_actual"] for ev in self.eventos]
            ax2.plot(horas, angulos, marker="o", color="#3498DB", linewidth=2, markersize=6)
            ax2.set_title("Ángulo Cervical en Cambios de Postura", fontsize=11, fontweight="bold")
            ax2.set_ylabel("Ángulo (°)")
            ax2.set_xlabel("Hora del evento")
            ax2.tick_params(axis="x", rotation=35)
            ax2.grid(True, linestyle="--", alpha=0.4)
        else:
            ax2.text(
                0.5,
                0.5,
                "Sin cambios de estado\n(Postura estable toda la sesión)",
                ha="center",
                va="center",
                fontsize=10,
                color="#555555",
            )
            ax2.set_title("Evolución de Cambios de Postura", fontsize=11, fontweight="bold")
            ax2.axis("off")

        fig.suptitle(
            f"Resumen de Sesión | Episodios encorvados: {self.conteo_encorvamientos} | "
            f"Alertas: {gestor_alertas.alertas_lanzadas} (Omitidas anti-spam: {gestor_alertas.alertas_omitidas_spam})",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(ruta_grafica, dpi=150)
        plt.close(fig)


# --- 5. CAPTURA DE VIDEO, VISUALIZACIÓN Y BUCLE PRINCIPAL ---
def inicializar_camara():
    for indice in (0, 1):
        cap = cv2.VideoCapture(indice)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"[CÁMARA] Conectada correctamente en índice {indice}.")
                return cap
            cap.release()
    return None


def dibujar_hud_y_esqueleto(
    frame,
    puntos,
    estado,
    angulo_actual,
    desv_angulo,
    desv_caida,
    calibrando,
    progreso_calibracion,
    calibrado_usuario,
):
    alto, ancho, _ = frame.shape
    escala = max(0.45, min(ancho, alto) / 720.0)
    grosor_linea = max(1, int(2 * escala))
    radio_punto = max(3, int(5 * escala))

    colores_estado = {
        "Normal": (80, 220, 100),
        "Encorvamiento Leve": (0, 190, 255),
        "Encorvamiento Crítico": (60, 60, 245),
    }
    color_activo = colores_estado.get(estado, (200, 200, 200))

    if puntos:
        def a_px(coord_norm):
            return int(coord_norm[0] * ancho), int(coord_norm[1] * alto)

        p_h_izq = a_px(puntos["hombro_izq"])
        p_h_der = a_px(puntos["hombro_der"])
        p_c_hombros = a_px(puntos["centro_hombros"])
        p_c_cabeza = a_px(puntos["centro_cabeza"])
        p_nariz = a_px(puntos["nariz"])
        p_o_izq = a_px(puntos["oreja_izq"])
        p_o_der = a_px(puntos["oreja_der"])

        cv2.line(frame, p_h_izq, p_h_der, (255, 200, 80), grosor_linea)
        cv2.line(frame, p_c_hombros, p_c_cabeza, color_activo, grosor_linea + 1)
        cv2.line(frame, p_o_izq, p_o_der, (200, 200, 200), max(1, grosor_linea - 1))

        for pt in (p_h_izq, p_h_der, p_nariz, p_o_izq, p_o_der):
            cv2.circle(frame, pt, radio_punto, (255, 255, 255), -1)
            cv2.circle(frame, pt, radio_punto + 1, color_activo, 1)

    pad = int(14 * escala)
    panel_w = int(430 * escala)
    panel_h = int(125 * escala)
    overlay = frame.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + panel_w, pad + panel_h), (24, 24, 36), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (pad, pad), (pad + panel_w, pad + panel_h), color_activo, grosor_linea)

    fuente = cv2.FONT_HERSHEY_SIMPLEX
    y_texto = pad + int(28 * escala)
    salto_y = int(28 * escala)

    cv2.putText(
        frame,
        f"Estado: {estado}",
        (pad + int(12 * escala), y_texto),
        fuente,
        0.65 * escala,
        color_activo,
        max(1, int(2 * escala)),
    )
    y_texto += salto_y
    cv2.putText(
        frame,
        f"Ang: {angulo_actual:.1f} deg ({desv_angulo:+.1f}) | Caida: {desv_caida:+.2f}",
        (pad + int(12 * escala), y_texto),
        fuente,
        0.52 * escala,
        (240, 240, 240),
        max(1, int(1.5 * escala)),
    )
    y_texto += salto_y
    etiqueta_base = "Personalizada (Usuario)" if calibrado_usuario else "Por defecto (Presiona C)"
    cv2.putText(
        frame,
        f"Base: {etiqueta_base}",
        (pad + int(12 * escala), y_texto),
        fuente,
        0.50 * escala,
        (180, 210, 255),
        max(1, int(1.5 * escala)),
    )

    if calibrando:
        bar_w = int(340 * escala)
        bar_h = int(26 * escala)
        bx = (ancho - bar_w) // 2
        by = alto - int(75 * escala)
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (30, 30, 30), -1)
        cv2.rectangle(
            frame,
            (bx, by),
            (bx + int(bar_w * progreso_calibracion), by + bar_h),
            (80, 220, 100),
            -1,
        )
        cv2.putText(
            frame,
            "Calibrando postura natural... Manten la posicion",
            (bx - int(25 * escala), by - int(10 * escala)),
            fuente,
            0.55 * escala,
            (80, 220, 100),
            max(1, int(2 * escala)),
        )

    cv2.putText(
        frame,
        "[C] Calibrar postura base   |   [Q / ESC] Salir y generar reporte",
        (pad, alto - pad),
        fuente,
        0.50 * escala,
        (220, 220, 220),
        max(1, int(1.5 * escala)),
    )


def main():
    if not MODEL_PATH.exists():
        print(f"[ERROR CRÍTICO] No se encontró el archivo del modelo: {MODEL_PATH}")
        sys.exit(1)

    ref = procesar_imagenes_referencia()
    base_angulo = ref["base_ang_default"]
    base_caida = ref["base_caida_default"]
    calibrado_usuario = False

    cap = inicializar_camara()
    if cap is None:
        print("[ERROR] No se detectó ninguna cámara web disponible (índices 0 y 1).")
        sys.exit(1)

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    gestor_alertas = GestorAlertasSilenciosas()
    historial = HistorialPostura()

    buffer_angulos = deque(maxlen=6)
    buffer_caidas = deque(maxlen=6)
    estado_confirmado = "Normal"
    estado_candidato = "Normal"
    contador_candidato = 0
    inicio_encorvamiento_sostenido = None

    calibrando = False
    muestras_calibracion_ang = []
    muestras_calibracion_caida = []
    TOTAL_MUESTRAS_CALIBRACION = 45

    timestamp_anterior_ms = 0
    print("[INICIO] Monitor de Postura activo. Presiona 'C' para calibrar tu postura natural.")

    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int(time.time() * 1000)
            if timestamp_ms <= timestamp_anterior_ms:
                timestamp_ms = timestamp_anterior_ms + 1
            timestamp_anterior_ms = timestamp_ms

            resultado = landmarker.detect_for_video(mp_image, timestamp_ms)

            puntos_dibujo = None
            ang_suavizado = base_angulo
            desv_ang = 0.0
            desv_caida = 0.0

            if resultado.pose_landmarks:
                metricas = calcular_metricas_postura(resultado.pose_landmarks[0])
                if metricas:
                    ang_raw, caida_raw, puntos_dibujo = metricas

                    # Calibración rápida en vivo (Tecla C) para registrar línea base personal
                    if calibrando:
                        muestras_calibracion_ang.append(ang_raw)
                        muestras_calibracion_caida.append(caida_raw)
                        if len(muestras_calibracion_ang) >= TOTAL_MUESTRAS_CALIBRACION:
                            base_angulo = float(np.median(muestras_calibracion_ang))
                            base_caida = float(np.median(muestras_calibracion_caida))
                            calibrando = False
                            calibrado_usuario = True
                            buffer_angulos.clear()
                            buffer_caidas.clear()
                            print(
                                f"[CALIBRACIÓN USUARIO] Ángulo base: {base_angulo:.1f}° | "
                                f"Caída base: {base_caida:.2f}"
                            )

                    # Suavizado móvil y cálculo de desviación respecto a la postura base
                    buffer_angulos.append(ang_raw)
                    buffer_caidas.append(caida_raw)
                    ang_suavizado = float(np.mean(buffer_angulos))
                    caida_suavizada = float(np.mean(buffer_caidas))

                    desv_ang = ang_suavizado - base_angulo
                    desv_caida = base_caida - caida_suavizada

                    if desv_ang >= ref["delta_ang_critico"] or desv_caida >= ref["delta_caida_critico"]:
                        estado_instantaneo = "Encorvamiento Crítico"
                    elif desv_ang >= ref["delta_ang_leve"] or desv_caida >= ref["delta_caida_leve"]:
                        estado_instantaneo = "Encorvamiento Leve"
                    else:
                        estado_instantaneo = "Normal"

                    # Validación por frames consecutivos para filtrar movimientos rápidos
                    if estado_instantaneo == estado_candidato:
                        contador_candidato += 1
                    else:
                        estado_candidato = estado_instantaneo
                        contador_candidato = 1

                    if contador_candidato >= FRAMES_CONFIRMACION and estado_candidato != estado_confirmado:
                        estado_confirmado = estado_candidato
                        historial.registrar_transicion(estado_confirmado, ang_suavizado, max(0.0, desv_ang))

                        if "Encorvamiento" in estado_confirmado:
                            inicio_encorvamiento_sostenido = time.time()
                        else:
                            inicio_encorvamiento_sostenido = None

                    if "Encorvamiento" in estado_confirmado and inicio_encorvamiento_sostenido is not None:
                        if (time.time() - inicio_encorvamiento_sostenido) >= SEGUNDOS_SOSTENIDOS_ALERTA:
                            gestor_alertas.intentar_lanzar_alerta(estado_confirmado, max(0.0, desv_ang))

            progreso_cal = (
                len(muestras_calibracion_ang) / TOTAL_MUESTRAS_CALIBRACION if calibrando else 0.0
            )

            dibujar_hud_y_esqueleto(
                frame=frame,
                puntos=puntos_dibujo,
                estado=estado_confirmado,
                angulo_actual=ang_suavizado,
                desv_angulo=desv_ang,
                desv_caida=desv_caida,
                calibrando=calibrando,
                progreso_calibracion=progreso_cal,
                calibrado_usuario=calibrado_usuario,
            )

            cv2.imshow("Monitor Inteligente de Postura", frame)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("q"), ord("Q"), 27):
                break
            elif tecla in (ord("c"), ord("C")) and not calibrando:
                calibrando = True
                muestras_calibracion_ang.clear()
                muestras_calibracion_caida.clear()
                print("[CALIBRACIÓN] Mantén tu postura natural cómoda frente al monitor...")

    cap.release()
    cv2.destroyAllWindows()
    historial.finalizar_y_exportar(gestor_alertas, calibrado_usuario)


if __name__ == "__main__":
    main()