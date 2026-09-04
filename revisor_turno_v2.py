"""
revisor_turno_v2.py
===================
Herramienta de revisión por lotes para un turno completo de ensayos.

FLUJO:
  1. Seleccionar carpeta del piñón (PIMA / ARBOL_SEC_14 / ARBOL_SEC_15)
  2. El programa escanea todos los CSV, los procesa automáticamente y
     los pre-clasifica como BUENO / SOSPECHOSO según curtosis
  3. El operador revisa los SOSPECHOSOS uno a uno y decide:
       ✔ BUENO  |  ✖ MALO  |  — IGNORAR
  4. Al terminar, exporta dataset.csv con features + etiqueta
     listo para entrenar el modelo ML

ESTRUCTURA DE CARPETAS ESPERADA:
  ARBOL_SEC_14/
      engrane_YYYYMMDD_HHMMSS_XXXX.csv
      ...
  PIMA/
      engrane_...csv
  ARBOL_SEC_15/
      engrane_...csv

SALIDA:
  ARBOL_SEC_14/dataset_YYYYMMDD.csv
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt, savgol_filter
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import glob
import threading
import datetime

# ==========================
# CONFIGURACIÓN
# ==========================
FS  = 48000
RPM = 1135.0
F_ROT = RPM / 60.0       # 18.917 Hz
T_ROT = 1.0 / F_ROT
M_ROT = int(T_ROT * FS)   # muestras por giro (~2537)      # ~52.9 ms

PINONES = {
    "PIMA":         {"dientes": 26},
    "ARBOL_SEC_14": {"dientes": 14},
    "ARBOL_SEC_15": {"dientes": 15},
}
for k, v in PINONES.items():
    v["gmf"]      = v["dientes"] * RPM / 60
    v["T_diente"] = 1.0 / v["gmf"]
    v["M_diente"] = int(v["T_diente"] * FS)

OPCIONES = [
    ("PIMA",         f"PIMA  (26d — GMF {PINONES['PIMA']['gmf']:.1f} Hz)"),
    ("ARBOL_SEC_14", f"ÁRBOL SEC. 14d — GMF {PINONES['ARBOL_SEC_14']['gmf']:.1f} Hz"),
    ("ARBOL_SEC_15", f"ÁRBOL SEC. 15d — GMF {PINONES['ARBOL_SEC_15']['gmf']:.1f} Hz"),
]
IDX_A_CLAVE = {i: clave for i, (clave, _) in enumerate(OPCIONES)}

MARGEN_BUSQUEDA   = 0.15
FACTOR_UMBRAL_SEP = 2.0
VENTANA_ENV_LENTA = 0.015
TRAMO_SEG         = 0.5
TRAMO_MUESTRAS    = int(TRAMO_SEG * FS)
TOL_T             = 0.10
TOL_SB            = F_ROT * 0.6

# ── Análisis por ventanas deslizantes ──
# En lugar de un único tramo central de 0.5s, se analiza toda la zona
# limpia con ventanas de VENTANA_SEG segundos cada PASO_SEG segundos.
# De cada criterio se extraen: máximo, media y percentil 90.
# Esto captura defectos que se manifiestan fuera del centro de la señal.
VENTANA_SEG        = 0.1     # duración de cada ventana (s)
VENTANA_MUESTRAS   = int(VENTANA_SEG * FS)   # 4800 muestras
PASO_SEG           = 0.05    # paso entre ventanas (50% solapamiento)
PASO_MUESTRAS      = int(PASO_SEG * FS)      # 2400 muestras
MIN_VENTANAS       = 3       # mínimo de ventanas para análisis válido

# Umbral de curtosis para pre-clasificar como SOSPECHOSO
# Se actualiza dinámicamente con los datos del turno
KURT_UMBRAL_INICIAL = 5.0   # valor inicial conservador

# Umbrales adicionales para pre-clasificación sospechoso (OR entre criterios)
SB_UMBRAL_INICIAL     = 8.0   # sb_ratio_max sobre este valor → sospechoso
CEPSTRUM_UMBRAL_COEF  = 1.4   # cepstrum_max > media_turno × coef → sospechoso

# ==========================
# VALIDACIÓN DE GRABACIÓN
# Filtro duro + score 0–100
# ==========================
# ── Filtro duro (descarte automático) ──
VAL_RMS_EMP_RATIO      = 0.45   # RMS_emp / RMS_ret mínimo → si es muy bajo, empuje sin engrane
                                 # (subido de 0.25: ruido de fondo puede alcanzar 25% fácilmente)
VAL_RMS_EMP_CV_MIN     = 0.08   # coeficiente de variación mínimo del RMS en empuje
                                 # CV < umbral → señal plana = sin engrane (complementa ratio)
VAL_KURT_GOLPE         = 12.0   # Kurt_max en ventana de 50ms → golpe externo
VAL_CLIP_PCT           = 0.5    # % de muestras saturadas (|x|>=0.995) → clipping
VAL_DUR_RET_MIN        = 0.30   # duración mínima retroceso estabilizado (s)
                                 # Justificación física: T_ROT=52.9ms → 0.30s = 5.7 giros
                                 # 4 ventanas de 0.10s → FFT estadísticamente robusto
                                 # 0.40s era demasiado estricto (10ms rechazaban datos válidos)
VAL_DUR_TOTAL_MIN      = 1.80   # duración total mínima del archivo (s)
                                 # si dur_total >= esto Y sep > 55% del archivo,
                                 # la segmentación tardía no descarta el dato
VAL_RMS_RET_ABS        = 3e-4   # RMS retroceso mínimo absoluto (señal muy débil)

# ── Score de confianza ──
# Cada criterio aporta puntos; score total 0–100
# score ≥ 70 → VALIDO | 40–69 → DUDOSO | < 40 → INVALIDO
VAL_SCORE_VALIDO  = 70
VAL_SCORE_DUDOSO  = 40

# Colores badge validación
C_VALIDO  = "#22c55e"    # verde
C_DUDOSO  = "#f59e0b"    # ámbar
C_INVALIDO= "#ef4444"    # rojo

# ==========================
# PALETA
# ==========================
C_BG       = "#0e0f11"
C_SURFACE  = "#161820"
C_SURFACE2 = "#1e2028"
C_BORDER   = "#2a2d38"
C_BORDER2  = "#353848"
C_TEXT     = "#e2e4ed"
C_TEXT_SUB = "#7a7f96"
C_TEXT_DIM = "#4a4f64"
C_ACENTO   = "#4f8ef7"
C_BUENO    = "#22c55e"
C_BUENO_BG = "#052e16"
C_MALO     = "#ef4444"
C_MALO_BG  = "#2d0a0a"
C_SOSP     = "#f59e0b"
C_SOSP_BG  = "#2d1e05"
C_IGNORAR  = "#6b7280"
C_MONO     = "Consolas"

LOGO_IMG = None


# ==========================
# HELPERS UI
# ==========================
def _dk(h):
    h = h.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return f"#{max(r-20,0):02x}{max(g-20,0):02x}{max(b-20,0):02x}"


def hacer_boton(parent, texto, comando, bg=C_SURFACE2, fg=C_TEXT,
                ancho=16, alto=1, fs=9, bold=False, state="normal"):
    peso = "bold" if bold else "normal"
    btn  = tk.Button(parent, text=texto, command=comando,
                     bg=bg, fg=fg, activebackground=_dk(bg), activeforeground=fg,
                     relief="flat", bd=0, font=(C_MONO, fs, peso),
                     width=ancho, height=alto, cursor="hand2", state=state)
    btn.bind("<Enter>", lambda e: btn.config(bg=_dk(bg)) if btn["state"]!="disabled" else None)
    btn.bind("<Leave>", lambda e: btn.config(bg=bg)      if btn["state"]!="disabled" else None)
    return btn


def lbl(parent, texto, fg=C_TEXT_SUB, fs=9, bold=False, bg=None):
    peso = "bold" if bold else "normal"
    return tk.Label(parent, text=texto, fg=fg,
                    bg=bg or C_SURFACE, font=(C_MONO, fs, peso))


def sep_h(parent, bg=C_SURFACE, pady=2):
    tk.Frame(parent, bg=C_BORDER, height=1).pack(fill="x", pady=pady)


def _cargar_logo():
    global LOGO_IMG
    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo_empresa.png")
    if not os.path.exists(ruta):
        return
    try:
        img = Image.open(ruta).convert("RGBA")
        img.thumbnail((110, 40), Image.LANCZOS)
        LOGO_IMG = ImageTk.PhotoImage(img)
    except Exception:
        pass


# ==========================
# NÚCLEO: ANÁLISIS (reutilizado del explorador)
# ==========================
def bp_filter(sig, lo, hi, fs=FS, order=4):
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype='band')
    return filtfilt(b, a, sig)


def detectar_separador(señal):
    """
    Detecta el punto de separación empuje→retroceso — máquina DEMM.
    Método RMS por giro: primer giro > 4× nivel_ref del empuje estable.
    Fallback progresivo: 3×, 2.5×, máximo absoluto.
    """
    n     = len(señal)
    s     = señal.astype(np.float64)
    M_ROT = int(T_ROT * FS)

    v_rap = max(1, int(VENTANA_ENV_LENTA * FS))
    env   = np.sqrt(np.convolve(s**2, np.ones(v_rap)/v_rap, mode='same'))

    i0 = int(n * 0.15)
    i1 = int(n * 0.85)
    zona    = s[i0:i1]
    n_giros = len(zona) // M_ROT

    if n_giros < 4:
        v_gru   = max(1, int(0.20 * FS))
        env_gru = np.sqrt(np.convolve(s**2, np.ones(v_gru)/v_gru, mode='same'))
        return int(np.argmax(env_gru[i0:i1])) + i0, env

    rms_giros = np.array([
        float(np.sqrt(np.mean(zona[i*M_ROT:(i+1)*M_ROT]**2)))
        for i in range(n_giros)
    ])

    n_ref     = max(2, n_giros // 3)
    nivel_ref = float(np.median(rms_giros[:n_ref]))
    if nivel_ref < 1e-9:
        nivel_ref = float(np.median(rms_giros)) + 1e-9

    for factor in [4.0, 3.0, 2.5]:
        for i, rms in enumerate(rms_giros):
            if rms > nivel_ref * factor:
                return i0 + i * M_ROT, env

    return i0 + int(np.argmax(rms_giros)) * M_ROT, env


def detectar_freno(señal_zona):
    """
    Detecta el final útil del retroceso — máquina DEMM.

    FÍSICA:
    - Tras el separador: 2-3 giros de arranque (peaks altos diente-diente)
    - Luego: engrane estabilizado (~4-10 giros)
    - Finalmente: freno final (OPCIONAL) — RMS comparable al separador
    El freno NUNCA ocurre antes de ~7 giros desde el separador (~370ms).
    Usar mínimo 350ms como guardia para no confundir arranque con freno.

    MÉTODO:
    - Nivel ref = giros 3-6 del retroceso (zona de arranque post-sep,
      usamos su mediana como referencia del nivel ALTO del arranque).
      Esperar a que el RMS baje a < 1.5× ese nivel (engrane estable)
      y LUEGO buscar un nuevo pico > 2.5× nivel estable.
    - Guardia temporal: no buscar en los primeros 350ms.
    - Fallback 92%: operador extrajo pieza antes del freno final.
    """
    n     = len(señal_zona)
    if n < int(FS * 0.15):
        return n

    s     = señal_zona.astype(np.float64)
    M_ROT = int(T_ROT * FS)
    n_giros = n // M_ROT
    if n_giros < 6:
        return int(n * 0.92)

    rms_giros = np.array([
        float(np.sqrt(np.mean(s[i*M_ROT:(i+1)*M_ROT]**2)))
        for i in range(n_giros)
    ])

    # Guardia temporal: ignorar los primeros 350ms (~6.6 giros)
    # Esto excluye arranque Y comienzo del engrane estable
    MIN_GIROS_GUARDIA = max(7, int(0.35 * FS / M_ROT))

    # Nivel estable = mediana de giros DESPUÉS de la guardia temporal
    # (zona de engrane estabilizado, antes del freno)
    i_est_0 = MIN_GIROS_GUARDIA
    i_est_1 = min(i_est_0 + 5, n_giros - 1)
    if i_est_1 > i_est_0 and i_est_0 < n_giros:
        nivel_ref = float(np.median(rms_giros[i_est_0:i_est_1]))
    else:
        # Zona muy corta: no hay suficiente retroceso para freno
        return int(n * 0.92)

    if nivel_ref < 1e-9:
        return int(n * 0.92)

    # Buscar DESPUÉS de la guardia el primer giro sobre 2.5× nivel estable
    for i in range(MIN_GIROS_GUARDIA, n_giros):
        if rms_giros[i] > nivel_ref * 2.5:
            return max(0, i * M_ROT - M_ROT // 2)

    return int(n * 0.92)   # freno no ocurrió


def detectar_estabilizacion_retroceso(zona_ret):
    """
    Detecta donde termina la transicion al inicio del retroceso.

    Estrategia:
      1. Saltar siempre un minimo de GIROS_MINIMOS giros desde el inicio
         (los peaks de arranque del AS duran tipicamente 4-5 giros)
      2. Desde ahi, buscar 3 giros consecutivos estables usando
         el percentil 20 de la segunda mitad como referencia
      3. Fallback: saltar GIROS_MINIMOS si no converge
    """
    s       = zona_ret.astype(np.float64)
    n       = len(s)
    M_ROT   = int(T_ROT * FS)
    n_giros = n // M_ROT
    if n_giros < 4:
        return 0

    rms_giros = np.array([
        float(np.sqrt(np.mean(s[i*M_ROT:(i+1)*M_ROT]**2)))
        for i in range(n_giros)
    ])

    # Nivel de referencia: percentil 20 de la segunda mitad
    segunda_mitad = rms_giros[n_giros // 2:]
    nivel_ref = float(np.percentile(segunda_mitad, 20))
    if nivel_ref < 1e-9:
        nivel_ref = float(np.median(rms_giros))
    if nivel_ref < 1e-9:
        return 0

    umbral_min   = nivel_ref * 0.35
    umbral_max   = nivel_ref * 1.80
    CONFIRM      = 3
    GIROS_MINIMOS = 5   # saltar siempre los primeros 5 giros (~265ms)
                        # cubre los peaks de arranque del AS

    inicio_busqueda = min(GIROS_MINIMOS, n_giros - CONFIRM - 1)

    for i in range(inicio_busqueda, n_giros - CONFIRM + 1):
        ventana = rms_giros[i:i+CONFIRM]
        if not all(umbral_min <= r <= umbral_max for r in ventana):
            continue
        if np.max(ventana) / (np.min(ventana) + 1e-12) > 1.8:
            continue
        return max(0, i * M_ROT - M_ROT // 2)

    # Fallback: retornar despues de GIROS_MINIMOS
    return min(GIROS_MINIMOS * M_ROT, n // 3)
def tramo_central(zona):
    nz = len(zona)
    if nz <= TRAMO_MUESTRAS:
        return zona.copy()
    centro = nz // 2
    mitad  = TRAMO_MUESTRAS // 2
    return zona[centro - mitad : centro + mitad]


def clasificar_impulsos(tramo, gmf):
    n   = len(tramo)
    sig = tramo.astype(np.float64)
    v_env = max(1, int(FS * 0.002))
    env   = np.convolve(np.abs(sig), np.ones(v_env)/v_env, mode='same')
    umbral   = np.mean(env) + 3.0 * np.std(env)
    dist_m   = int(T_ROT * FS * 0.3)
    peaks, _ = find_peaks(env, height=umbral, distance=dist_m)
    if len(peaks) == 0:
        return np.array([]), np.array([]), tramo.copy(), {"n_defecto":0,"n_ruido":0,"hay_sidebands":False}

    # Buscar periodicidad a N×T_ROT para N = 1, 2, 3, 4
    #
    # FÍSICA: el umbral mean+3std es selectivo — en señales con muchos golpes
    # solo detecta los más extremos. Estos pueden estar espaciados a 2, 3 o 4×T_ROT.
    # Todos confirman el mismo defecto cíclico (el mismo diente cada N vueltas
    # aparece como el peak más alto de su grupo).
    # Tolerancia ±15% (antes 10%) para absorber variación de velocidad.
    TOL_PER = 0.15
    bandas = [(k * T_ROT * FS * (1 - TOL_PER),
               k * T_ROT * FS * (1 + TOL_PER))
              for k in range(1, 5)]   # 1×, 2×, 3×, 4× T_ROT
    periodico = np.zeros(len(peaks), dtype=bool)
    for i, pk in enumerate(peaks):
        for j, pk2 in enumerate(peaks):
            if i == j:
                continue
            dt = abs(int(pk) - int(pk2))
            if any(lo <= dt <= hi for lo, hi in bandas):
                periodico[i] = True
                break

    ventana_h = np.hanning(n)
    fft_mag   = np.abs(np.fft.rfft(sig * ventana_h)) * 2 / n
    freqs     = np.fft.rfftfreq(n, d=1.0/FS)

    def e_banda(fc, bw):
        mask = (freqs >= fc-bw) & (freqs <= fc+bw)
        return float(np.sum(fft_mag[mask]**2))

    e_sb = (e_banda(gmf+F_ROT,TOL_SB) + e_banda(gmf-F_ROT,TOL_SB) +
            e_banda(gmf+2*F_ROT,TOL_SB) + e_banda(gmf-2*F_ROT,TOL_SB)) / 4
    e_ref = e_banda(gmf*1.8, TOL_SB)
    hay_sb = e_sb > e_ref * 1.5

    # CRITERIO DE CLASIFICACIÓN:
    # Un impulso es DEFECTO DE DIENTE si es periódico a T_ROT.
    # La periodicidad sola es condición suficiente — el mismo diente
    # golpeando en cada vuelta ES el defecto que buscamos detectar.
    # Los sidebands son una evidencia adicional pero NO un requisito:
    # defectos puntuales o tempranos pueden no tener sidebands aún.
    #
    # Un impulso NO periódico = ruido externo o evento aislado.
    defecto_mask = periodico          # periódico = defecto de diente
    ruido_mask   = ~periodico         # no periódico = golpe externo/ruido
    peaks_def = peaks[defecto_mask]
    peaks_rui = peaks[ruido_mask]

    # Solo suprimir del tramo los impulsos NO periódicos (ruido externo)
    # Los impulsos de defecto se conservan — son la señal de interés
    tramo_limpio = sig.copy()
    ventana_sust = int(T_ROT * FS * 0.4)
    for pk in peaks_rui:
        a = max(0, int(pk) - ventana_sust)
        b = min(n, int(pk) + ventana_sust)
        if a > 0 and b < n:
            tramo_limpio[a:b] = np.linspace(tramo_limpio[a], tramo_limpio[b], b-a)
        else:
            tramo_limpio[a:b] = np.mean(sig)

    return peaks_def, peaks_rui, tramo_limpio.astype(np.float32), {
        "n_defecto":     int(np.sum(defecto_mask)),
        "n_ruido":       int(np.sum(ruido_mask)),
        "hay_sidebands": hay_sb,
        "peaks_all":     peaks,
        "env":           env,
    }


def _nvh_una_ventana(seg, gmf):
    """
    Calcula todos los criterios NVH sobre un segmento corto (≥ VENTANA_MUESTRAS).
    Función interna usada por calcular_features_ventanas().
    """
    sig = seg.astype(np.float64)
    n   = len(sig)

    def curtosis(s):
        mu  = np.mean(s)
        num = np.mean((s - mu) ** 4)
        den = (np.mean((s - mu) ** 2)) ** 2
        return float(num / (den + 1e-12))

    rms      = float(np.sqrt(np.mean(sig ** 2)))
    kurt     = curtosis(sig)

    ventana_h = np.hanning(n)
    fft_mag   = np.abs(np.fft.rfft(sig * ventana_h)) * 2 / n
    freqs     = np.fft.rfftfreq(n, d=1.0/FS)
    fft_db    = 20 * np.log10(fft_mag + 1e-12)

    # Savgol: ajustar window_length al tamaño real del segmento (debe ser impar y < n)
    wl = min(24, len(fft_db) - 1)
    if wl % 2 == 0: wl -= 1
    po = min(8, wl - 1)
    fft_suav = savgol_filter(fft_db, window_length=max(wl,3), polyorder=max(po,1))

    fondo_espectro = float(np.sqrt(np.mean((fft_db - fft_suav) ** 2)))

    ordenes   = freqs / F_ROT
    orden_gmf = gmf / F_ROT
    idx_gmf   = int(np.argmin(np.abs(ordenes - orden_gmf)))
    nivel_gmf = float(fft_db[idx_gmf])

    diff  = fft_db - fft_suav
    above = diff > 0
    dx    = float(ordenes[1] - ordenes[0]) if len(ordenes) > 1 else 1.0
    densidad_ruido      = float(np.sum(diff[above]) * dx)
    n_frec_sobre_umbral = int(np.sum(above))
    nivel_max_espectral = float(np.max(diff))

    amp_esp    = np.abs(np.fft.rfft(sig)) / n
    wl2 = min(25, len(amp_esp) - 1)
    if wl2 % 2 == 0: wl2 -= 1
    umbral_amp = savgol_filter(amp_esp, window_length=max(wl2,3), polyorder=3)
    amplitud_max_espectral = float(np.max(amp_esp - umbral_amp))

    spectrum_sq  = np.abs(np.fft.fft(sig)) ** 2
    log_spec     = np.log(spectrum_sq + 1e-10)
    cepstrum     = np.abs(np.fft.ifft(log_spec)) ** 2
    cepstrum_max = float(np.max(cepstrum))

    def e_banda(fc, bw):
        mask = (freqs >= fc - bw) & (freqs <= fc + bw)
        return float(np.sum(fft_mag[mask] ** 2))
    e_sb     = (e_banda(gmf + F_ROT, TOL_SB) + e_banda(gmf - F_ROT, TOL_SB)) / 2
    e_ref    = e_banda(gmf * 1.8, TOL_SB) + 1e-20
    sb_ratio = float(e_sb / e_ref)

    return {
        "rms": rms, "kurt": kurt,
        "fondo_espectro": fondo_espectro, "nivel_gmf": nivel_gmf,
        "densidad_ruido": densidad_ruido,
        "n_frec_sobre_umbral": n_frec_sobre_umbral,
        "nivel_max_espectral": nivel_max_espectral,
        "amplitud_max_espectral": amplitud_max_espectral,
        "cepstrum_max": cepstrum_max,
        "sb_ratio": sb_ratio,
        "fft_db": fft_db, "fft_suav": fft_suav,
        "ordenes": ordenes, "orden_gmf": orden_gmf,
    }


def _agregar_ventanas(lista_vals):
    """
    Agrega una lista de valores escalares en: max, mean, p90.
    Retorna dict con sufijos _max, _mean, _p90.
    """
    arr = np.array(lista_vals, dtype=np.float64)
    return {
        "max":  float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "p90":  float(np.percentile(arr, 90)),
    }


def calcular_features(tramo, tramo_limpio, gmf):
    """
    Análisis por ventanas deslizantes sobre toda la zona limpia.

    En lugar de un único tramo central de 0.5s, divide la zona completa
    en ventanas de VENTANA_SEG s con solapamiento del 50% y calcula los
    criterios NVH en cada una. Extrae máximo, media y percentil 90 de
    cada criterio — esto captura defectos en cualquier parte de la zona,
    no solo en el centro.

    Criterios NVH (doc BMIR-H0176-2025-0001):
      - Curtosis (Eq. 4)
      - Nivel global / RMS (Eq. 12)
      - Fondo espectro (Eq. 6)  w=24, m=8
      - Nivel de orden GMF (Eq. 13)
      - Cepstrum (Eq. 8)
      - Densidad de ruido (Eq. 1)
      - Nº frec. sobre umbral (Eq. 3)
      - Amplitud máx. espectral (Eq. 9)
      - Nivel máx. espectral (Eq. 11)
      - Sidebands ratio
    """
    sig  = tramo.astype(np.float64)
    sigl = tramo_limpio.astype(np.float64)
    n    = len(sig)

    # ── Ventanas deslizantes sobre señal original y limpia ──────────────────
    # Clave: analizar TODA la zona, no solo el centro
    resultados_orig  = []
    resultados_limp  = []

    for inicio in range(0, n - VENTANA_MUESTRAS + 1, PASO_MUESTRAS):
        fin = inicio + VENTANA_MUESTRAS
        seg_o = sig[inicio:fin]
        seg_l = sigl[inicio:fin]
        resultados_orig.append(_nvh_una_ventana(seg_o, gmf))
        resultados_limp.append(_nvh_una_ventana(seg_l, gmf))

    # Si la señal es muy corta, usar toda ella como única ventana
    if len(resultados_orig) < MIN_VENTANAS:
        resultados_orig = [_nvh_una_ventana(sig,  gmf)]
        resultados_limp = [_nvh_una_ventana(sigl, gmf)]

    # ── Agregación: max, mean, p90 por criterio ─────────────────────────────
    criterios_escalares = [
        "rms", "kurt", "fondo_espectro", "nivel_gmf",
        "densidad_ruido", "n_frec_sobre_umbral",
        "nivel_max_espectral", "amplitud_max_espectral",
        "cepstrum_max", "sb_ratio",
    ]

    def agregar(resultados, clave):
        vals = [r[clave] for r in resultados]
        return _agregar_ventanas(vals)

    # Ventana representativa para gráfico: la de mayor curtosis
    idx_rep_o = int(np.argmax([r["kurt"] for r in resultados_orig]))
    idx_rep_l = int(np.argmax([r["kurt"] for r in resultados_limp]))
    rep_o     = resultados_orig[idx_rep_o]
    rep_l     = resultados_limp[idx_rep_l]

    # ── Construir resultado ──────────────────────────────────────────────────
    out = {}
    for c in criterios_escalares:
        ag_o = agregar(resultados_orig, c)
        ag_l = agregar(resultados_limp, c)
        out[f"{c}_max"]       = ag_o["max"]
        out[f"{c}_mean"]      = ag_o["mean"]
        out[f"{c}_p90"]       = ag_o["p90"]
        out[f"{c}_limp_max"]  = ag_l["max"]
        out[f"{c}_limp_mean"] = ag_l["mean"]
        out[f"{c}_limp_p90"]  = ag_l["p90"]

    # Alias de compatibilidad con la UI existente
    out["rms"]       = out["rms_mean"]
    out["rms_limp"]  = out["rms_limp_mean"]
    out["kurt"]      = out["kurt_mean"]
    out["kurt_limp"] = out["kurt_limp_mean"]
    # Para pre-clasificación usamos el MAX (peor ventana detectada)
    out["kurt_limp_worst"]     = out["kurt_limp_max"]
    out["cepstrum_max"]        = out["cepstrum_max_max"]
    out["sb_ratio"]            = out["sb_ratio_max"]
    out["fondo_espectro"]      = out["fondo_espectro_mean"]
    out["nivel_gmf"]           = out["nivel_gmf_mean"]
    out["densidad_ruido"]      = out["densidad_ruido_mean"]
    out["n_frec_sobre_umbral"] = out["n_frec_sobre_umbral_mean"]
    out["nivel_max_espectral"] = out["nivel_max_espectral_max"]
    out["amplitud_max_espectral"] = out["amplitud_max_espectral_max"]

    # Para graficar: ventana representativa
    out["fft_db"]   = rep_o["fft_db"]
    out["fft_suav"] = rep_o["fft_suav"]
    out["ordenes"]  = rep_o["ordenes"]
    out["orden_gmf"]= rep_o["orden_gmf"]
    out["n"]        = n
    out["n_ventanas"] = len(resultados_orig)

    return out


def analizar_archivo(ruta, gmf, M_diente):
    """
    Procesa un CSV completo y retorna el resultado con todas las features.
    Retorna None si hay error.
    """
    try:
        df    = pd.read_csv(ruta)
        señal = df["senal"].values.astype(np.float32)
        n     = len(señal)

        # Detectar zonas
        idx_sep, env_lenta = detectar_separador(señal)
        margen_sep   = int(FS * 0.02)
        i_emp_inicio = int(n * 0.05)
        i_emp_fin    = max(0, idx_sep - margen_sep)
        i_ret_inicio = min(n, idx_sep + margen_sep)
        i_ret_fin_max= int(n * 0.95)
        zona_ret_prel = señal[i_ret_inicio:i_ret_fin_max]
        idx_freno_l   = detectar_freno(zona_ret_prel)
        i_ret_fin     = i_ret_inicio + idx_freno_l

        zona_emp = señal[i_emp_inicio:i_emp_fin]
        zona_ret_bruta = señal[i_ret_inicio:i_ret_fin]
        idx_estab      = detectar_estabilizacion_retroceso(zona_ret_bruta)
        zona_ret       = zona_ret_bruta[idx_estab:]
        i_ret_estab    = i_ret_inicio + idx_estab
        te = tramo_central(zona_emp)
        tr = tramo_central(zona_ret)

        # Clasificar impulsos — tramo central para features NVH
        _, _, te_limp, inf_e = clasificar_impulsos(te, gmf)
        _, _, tr_limp, inf_r = clasificar_impulsos(tr, gmf)
        # Clasificar impulsos sobre zona COMPLETA de empuje para detección
        # de periodicidad robusta (zona_emp tiene ~17 giros vs ~9.5 del tramo)
        _, _, _, inf_e_full = clasificar_impulsos(zona_emp, gmf)

        # Features
        fe = calcular_features(te, te_limp, gmf)
        fr = calcular_features(tr, tr_limp, gmf)

        # Pre-clasificación automática — usa el PEOR valor entre todas las ventanas
        kurt_max  = max(fe["kurt_limp_worst"], fr["kurt_limp_worst"])
        sb_max    = max(fe["sb_ratio"],        fr["sb_ratio"])
        ceps_max  = max(fe["cepstrum_max"],    fr["cepstrum_max"])

        resultado = {
            "ruta":        ruta,
            "nombre":      os.path.basename(ruta),
            "señal":       señal,
            "env_lenta":   env_lenta,
            "t_sep":       idx_sep / FS,
            "t_freno":     i_ret_fin / FS,
            "i_emp_inicio":i_emp_inicio,
            "i_emp_fin":   i_emp_fin,
            "i_ret_inicio":i_ret_inicio,
            "i_ret_estab": i_ret_estab,
            "i_ret_fin":   i_ret_fin,
            "tramo_emp":   te,
            "tramo_ret":   tr,
            "fe":          fe,
            "fr":          fr,
            "inf_e":       inf_e,
            "inf_r":       inf_r,
            "inf_e_full":  inf_e_full,
            "kurt_max":    kurt_max,
            "sb_max":      sb_max,
            "ceps_max":    ceps_max,
            "gmf":         gmf,
            "M_diente":    M_diente,
            "n_dientes":   round(gmf / F_ROT),
            "etiqueta":    None,   # se asigna por el operador
            "hora":        _hora_desde_nombre(ruta),
        }
        # Validación de calidad de la grabación
        resultado["validacion"] = validar_grabacion(resultado)
        return resultado
    except Exception as ex:
        return {"error": str(ex), "nombre": os.path.basename(ruta)}


def _hora_desde_nombre(ruta):
    """Extrae la hora del nombre de archivo engrane_YYYYMMDD_HHMMSS_XXXX.csv"""
    try:
        partes = os.path.basename(ruta).replace(".csv","").split("_")
        hora   = partes[2]
        return f"{hora[:2]}:{hora[2:4]}:{hora[4:6]}"
    except Exception:
        return "--:--:--"


# ==========================
# VALIDACIÓN DE GRABACIÓN
# ==========================
def validar_grabacion(res):
    """
    Analiza si una grabación es técnicamente válida para incluirse
    en el dataset de entrenamiento de calidad.

    Retorna dict con:
      "valida"     : bool — True si pasa el filtro duro
      "score"      : int  — 0-100 score de confianza
      "nivel"      : str  — "VÁLIDO" / "DUDOSO" / "INVÁLIDO"
      "color"      : str  — color UI
      "razones"    : list[str] — por qué fue rechazada o penalizada
      "bonos"      : list[str] — qué criterios positivos suman
    """
    if "error" in res:
        return {"valida": False, "score": 0, "nivel": "INVÁLIDO",
                "color": C_INVALIDO,
                "razones": [f"Error de lectura: {res['error']}"],
                "bonos": []}

    fe     = res["fe"]
    fr     = res["fr"]
    señal  = res["señal"]
    razones = []
    bonos   = []
    score   = 0

    # ─────────────────────────────────────────────────────────────
    # CAPA 1 — FILTROS DUROS (cualquiera descarta la grabación)
    # ─────────────────────────────────────────────────────────────

    # 1a. RMS empuje vs retroceso — detecta grabaciones sin engrane real
    #
    # CRITERIO 1: RMS global empuje vs retroceso muy bajo + señal plana
    rms_emp = fe.get("rms_mean", 0.0)
    rms_ret = fr.get("rms_mean", 1e-9)
    ratio_rms_flancos = rms_emp / (rms_ret + 1e-12)
    rms_emp_max = fe.get("rms_max", rms_emp)
    cv_rms_emp  = (rms_emp_max - rms_emp) / (rms_emp + 1e-12)

    sin_engrane = (ratio_rms_flancos < 0.20) and (cv_rms_emp < VAL_RMS_EMP_CV_MIN)
    if sin_engrane:
        razones.append(
            f"✖ Sin engrane en empuje: RMS_emp/RMS_ret={ratio_rms_flancos:.2f} "
            f"y CV_rms_emp={cv_rms_emp:.3f} — señal plana")
        return {"valida": False, "score": 0, "nivel": "INVÁLIDO",
                "color": C_INVALIDO, "razones": razones, "bonos": []}

    # CRITERIO 2: Captura temprana — inicio del empuje sin engrane
    #
    # FÍSICA: el programa automático a veces inicia la captura antes de que
    # el piñón contacte. El resultado es un empuje con los primeros giros
    # planos (baja amplitud) y solo la parte posterior con engrane real.
    #
    # MÉTODO: análisis giro a giro del empuje.
    # 1. Calcular RMS por giro sobre toda la zona de empuje
    # 2. Nivel de engrane estable = mediana del último 2/3 de giros
    # 3. Contar giros iniciales con RMS < 30% del nivel estable
    # 4. Si ≥ 20% del total de giros son planos → captura temprana
    #
    # Ventaja sobre ratio ini/fin: robusto ante zonas de transición mixtas
    # y funciona igual para PIMA (baja amplitud) y ÁRBOL SEC (alta amplitud).
    # Soportar ambas convenciones de nombre de clave
    i_emp_i = res.get("i_emp_i", res.get("i_emp_inicio", 0))
    i_emp_f = res.get("i_emp_f", res.get("i_emp_fin",   len(señal)//2))
    zona_emp_raw = señal[i_emp_i : i_emp_f]
    n_giros_emp  = len(zona_emp_raw) // M_ROT

    if n_giros_emp >= 4 and rms_ret > VAL_RMS_RET_ABS * 2:
        rms_giros_emp = np.array([
            float(np.sqrt(np.mean(zona_emp_raw[i*M_ROT:(i+1)*M_ROT].astype(np.float64)**2)))
            for i in range(n_giros_emp)
        ])
        # Mediana del empuje — robusta ante peaks aislados de golpes o separador
        mediana_emp   = float(np.median(rms_giros_emp))
        ratio_emp_ret = mediana_emp / (rms_ret + 1e-12)
        # Si la mediana del empuje es < 30% del retroceso → empuje sin engrane
        # Calibración:
        #   Sin engrane:       ratio ≈ 0.25 → INVÁLIDO
        #   PIMA normal:       ratio ≈ 0.80 → OK
        #   PIMA con defecto:  ratio ≈ 0.40 → OK
        #   ÁRBOL normal:      ratio ≈ 1.00 → OK
        #   ÁRBOL con defecto: ratio ≈ 0.33 → OK
        if ratio_emp_ret < 0.30:
            razones.append(
                f"✖ Empuje sin engrane: mediana_emp/RMS_ret={ratio_emp_ret:.2f} "
                f"(mín 0.30) — captura antes del contacto del piñón "
                f"(med_emp={mediana_emp:.5f}, RMS_ret={rms_ret:.5f})")
            return {"valida": False, "score": 0, "nivel": "INVÁLIDO",
                    "color": C_INVALIDO, "razones": razones, "bonos": []}

    # 1b. Señal retroceso demasiado débil — sensor desconectado o sin contacto
    if rms_ret < VAL_RMS_RET_ABS:
        razones.append(
            f"✖ Señal retroceso muy débil: RMS={rms_ret:.6f} "
            f"(mín {VAL_RMS_RET_ABS:.1e})")
        return {"valida": False, "score": 0, "nivel": "INVÁLIDO",
                "color": C_INVALIDO, "razones": razones, "bonos": []}

    # 1c. Saturación — clipping en señal bruta
    n_clip    = int(np.sum(np.abs(señal) >= 0.995))
    pct_clip  = n_clip / len(señal) * 100
    if pct_clip > VAL_CLIP_PCT:
        razones.append(
            f"✖ Saturación detectada: {pct_clip:.1f}% muestras clipeadas "
            f"(máx {VAL_CLIP_PCT}%)")
        return {"valida": False, "score": 0, "nivel": "INVÁLIDO",
                "color": C_INVALIDO, "razones": razones, "bonos": []}

    # 1d. Duración retroceso + posición del separador
    #
    # FÍSICA: En una captura correcta, el separador cae entre el 25%-60%
    # del archivo (empuje y retroceso tienen duración similar).
    # Si sep > 65% del archivo → la captura fue fuera de tiempo:
    # el programa automático tomó la señal sin engrane real, o el engrane
    # había terminado cuando se inició la captura. → INVÁLIDO.
    # Si sep > 55% pero ≤ 65% → posible error de segmentación → DUDOSO.
    i_estab   = res.get("i_ret_estab", res["i_ret_inicio"])
    dur_ret   = (res["i_ret_fin"] - i_estab) / FS
    dur_total = len(señal) / FS
    t_sep     = res.get("t_sep", dur_total * 0.5)
    frac_sep  = t_sep / (dur_total + 1e-9)

    # Captura fuera de tiempo: sep después del 65% → inválido directamente
    if frac_sep > 0.65:
        razones.append(
            f"✖ Captura fuera de tiempo: sep={t_sep:.2f}s ({frac_sep*100:.0f}% del archivo) "
            f"— sin engrane real, programa automático tomó la señal incorrectamente")
        return {"valida": False, "score": 0, "nivel": "INVÁLIDO",
                "color": C_INVALIDO, "razones": razones, "bonos": []}

    if dur_ret < VAL_DUR_RET_MIN:
        seg_tardia = frac_sep > 0.55 and dur_total >= VAL_DUR_TOTAL_MIN
        if seg_tardia:
            razones.append(
                f"~ Separador tardío: sep={t_sep:.2f}s ({frac_sep*100:.0f}% del archivo), "
                f"retroceso útil={dur_ret:.2f}s — segmentación posiblemente errónea")
        else:
            razones.append(
                f"✖ Retroceso útil muy corto: {dur_ret:.3f}s "
                f"(mín {VAL_DUR_RET_MIN}s, archivo={dur_total:.2f}s)")
            return {"valida": False, "score": 0, "nivel": "INVÁLIDO",
                    "color": C_INVALIDO, "razones": razones, "bonos": []}

    # 1e. Golpe externo — curtosis impulsiva en ventana de 50ms
    #
    # FÍSICA: Un golpe de diente dañado produce curtosis alta PERO es
    # periódico a T_ROT (mismo diente golpea cada vuelta). Es el dato
    # más valioso del ensayo — NO debe descartarse como golpe externo.
    # Un golpe externo real es aislado (no periódico): impacto accidental,
    # manipulación brusca, objeto extraño.
    #
    # CRITERIO: solo marcar como golpe externo si la curtosis es muy alta
    # Y los impulsos detectados son mayoritariamente NO periódicos.
    # Si hay impulsos periódicos (n_defecto > 0) o más defecto que ruido,
    # es un diente dañado → permitir y clasificar normalmente.
    # 1e. Golpe externo — solo si curtosis alta Y sin ningún peak periódico
    #
    # FÍSICA DEL PROYECTO: El objetivo es detectar piñones con dientes dañados.
    # Un diente dañado produce peaks de alta curtosis a T_ROT — exactamente lo
    # que buscamos. Si hay CUALQUIER peak periódico en la zona de empuje,
    # no es un golpe externo sino el defecto que queremos medir.
    #
    # Golpe externo real = curtosis alta + CERO peaks periódicos.
    # Diente dañado      = curtosis alta + AL MENOS UN peak periódico.
    #
    # Se usa inf_e_full (zona completa de empuje, ~17 giros) para mayor
    # robustez que el tramo central de 0.5s.
    kurt_golpe = fe.get("kurt_limp_worst", res.get("kurt_max", 0.0))
    inf_emp    = res.get("inf_e_full", res.get("inf_e", {}))
    n_def_emp  = inf_emp.get("n_defecto", 0)   # peaks periódicos a T_ROT
    n_rui_emp  = inf_emp.get("n_ruido",   0)   # peaks no periódicos

    if kurt_golpe > VAL_KURT_GOLPE:
        if n_def_emp > 0:
            # Hay peaks periódicos → es defecto de diente, no golpe externo
            # Registrar como nota informativa pero NO descartar
            razones.append(
                f"~ Curtosis alta ({kurt_golpe:.1f}) con {n_def_emp} impulso(s) "
                f"periódico(s) a T_ROT → defecto de diente detectado en empuje")
        else:
            # Sin periodicidad → golpe externo real
            razones.append(
                f"✖ Golpe externo en empuje: Kurt_max={kurt_golpe:.1f} "
                f"(máx {VAL_KURT_GOLPE}) — sin impulsos periódicos a T_ROT "
                f"(def={n_def_emp}, ruido={n_rui_emp})")
            return {"valida": False, "score": 0, "nivel": "INVÁLIDO",
                    "color": C_INVALIDO, "razones": razones, "bonos": []}

    # ─────────────────────────────────────────────────────────────
    # CAPA 2 — SCORE DE CONFIANZA (0–100)
    # ─────────────────────────────────────────────────────────────
    # Si llega aquí, la grabación pasó los filtros duros.
    # Cada criterio suma puntos según qué tan "limpia" es la señal.

    # 2a. Estabilidad RMS en retroceso (CV bajo = señal estable) — máx 20 pts
    rms_vals_ret = [fr.get(f"rms_{s}", rms_ret)
                    for s in ("mean","p90","max")]
    cv_rms_ret = (max(rms_vals_ret) - min(rms_vals_ret)) / (np.mean(rms_vals_ret) + 1e-12)
    if cv_rms_ret < 0.15:
        score += 20
        bonos.append(f"✔ RMS retroceso muy estable (CV={cv_rms_ret:.2f})")
    elif cv_rms_ret < 0.35:
        score += 12
        bonos.append(f"~ RMS retroceso estable (CV={cv_rms_ret:.2f})")
    else:
        score += 4
        razones.append(f"~ RMS retroceso inestable (CV={cv_rms_ret:.2f})")

    # 2b. GMF visible en retroceso — máx 20 pts
    nivel_gmf_ret = fr.get("nivel_gmf_mean", -100.0)
    nivel_gmf_emp = fe.get("nivel_gmf_mean", -100.0)
    if nivel_gmf_ret > -60:
        score += 20
        bonos.append(f"✔ GMF visible en retroceso ({nivel_gmf_ret:.1f} dB)")
    elif nivel_gmf_ret > -75:
        score += 12
        bonos.append(f"~ GMF débil en retroceso ({nivel_gmf_ret:.1f} dB)")
    else:
        score += 3
        razones.append(f"~ GMF no visible en retroceso ({nivel_gmf_ret:.1f} dB)")

    # 2c. Curtosis retroceso en rango normal (3–7) — máx 20 pts
    kurt_ret_mean = fr.get("kurt_limp_mean", 3.0)
    if 3.0 <= kurt_ret_mean <= 5.5:
        score += 20
        bonos.append(f"✔ Curtosis retroceso normal ({kurt_ret_mean:.2f})")
    elif kurt_ret_mean <= 7.5:
        score += 12
        bonos.append(f"~ Curtosis retroceso elevada ({kurt_ret_mean:.2f})")
    else:
        score += 4
        razones.append(f"~ Curtosis retroceso alta ({kurt_ret_mean:.2f})")

    # 2d. Separación empuje/retroceso coherente — máx 20 pts
    # El retroceso debe tener RMS similar o mayor al empuje (es la zona de análisis principal)
    if 0.5 <= ratio_rms_flancos <= 3.0:
        score += 20
        bonos.append(f"✔ Proporción emp/ret coherente ({ratio_rms_flancos:.2f})")
    elif ratio_rms_flancos <= 5.0:
        score += 10
        bonos.append(f"~ Proporción emp/ret aceptable ({ratio_rms_flancos:.2f})")
    else:
        score += 3
        razones.append(f"~ Proporción emp/ret anómala ({ratio_rms_flancos:.2f})")

    # 2e. Suficientes ventanas de análisis en retroceso — máx 15 pts
    n_vent_ret = fr.get("n_ventanas", 1)
    if n_vent_ret >= 8:
        score += 15
        bonos.append(f"✔ Muchas ventanas de análisis ({n_vent_ret})")
    elif n_vent_ret >= 4:
        score += 9
        bonos.append(f"~ Ventanas de análisis suficientes ({n_vent_ret})")
    else:
        score += 3
        razones.append(f"~ Pocas ventanas de análisis ({n_vent_ret})")

    # 2f. Calidad de segmentación (separador en posición esperada) — máx 5 pts
    #     Separador en 20–55% del archivo → segmentación confiable
    frac_sep_score = res.get("t_sep", 0) / (len(señal) / FS + 1e-9)
    if 0.20 <= frac_sep_score <= 0.55:
        score += 5
        bonos.append(f"✔ Separador en posición normal ({frac_sep_score*100:.0f}% del archivo)")
    elif frac_sep_score <= 0.65:
        score += 2
        razones.append(f"~ Separador algo tardío ({frac_sep_score*100:.0f}% del archivo)")
    else:
        score += 0
        razones.append(f"~ Separador muy tardío ({frac_sep_score*100:.0f}%) — retroceso reducido")

    # ── Determinar nivel ──
    score = int(min(score, 100))
    if score >= VAL_SCORE_VALIDO:
        nivel = "VÁLIDO";   color = C_VALIDO
    elif score >= VAL_SCORE_DUDOSO:
        nivel = "DUDOSO";   color = C_DUDOSO
    else:
        nivel = "INVÁLIDO"; color = C_INVALIDO

    return {
        "valida":  score >= VAL_SCORE_DUDOSO,   # dudosos se guardan pero marcados
        "score":   score,
        "nivel":   nivel,
        "color":   color,
        "razones": razones,
        "bonos":   bonos,
    }


# ==========================
# ESTADO GLOBAL
# ==========================
ensayos      = []        # lista de dicts resultado de analizar_archivo
idx_actual   = [0]       # índice del ensayo en pantalla
kurt_umbral  = [KURT_UMBRAL_INICIAL]
clave_pinon  = ["ARBOL_SEC_14"]


# Umbrales dinámicos — se actualizan al cargar el turno
sb_umbral    = [SB_UMBRAL_INICIAL]
ceps_umbral  = [999.0]   # se calcula como media_turno × CEPSTRUM_UMBRAL_COEF

def es_sospechoso(res):
    """
    Un ensayo es sospechoso si CUALQUIERA de los 3 criterios supera su umbral:
      - Curtosis máxima (peor ventana)  → detecta nicks / impactos
      - SB ratio máximo                 → detecta micropitting / modulación
      - Cepstrum máximo                 → detecta armonías / defecto periódico
    """
    if "error" in res:
        return True
    return (res["kurt_max"]  >= kurt_umbral[0]  or
            res["sb_max"]    >= sb_umbral[0]     or
            res["ceps_max"]  >= ceps_umbral[0])


# ==========================
# UI — CONSTRUCCIÓN
# ==========================
root = tk.Tk()
root.title("Revisor de Turno — DEMM")
root.geometry("1300x820")
root.minsize(1100, 700)
root.configure(bg=C_BG)
root.resizable(True, True)

_cargar_logo()


# ── HEADER ──
frame_header = tk.Frame(root, bg=C_SURFACE, height=52)
frame_header.pack(fill="x")
frame_header.pack_propagate(False)
tk.Frame(frame_header, bg=C_ACENTO, width=3).pack(side="left", fill="y")
if LOGO_IMG:
    tk.Label(frame_header, image=LOGO_IMG, bg=C_SURFACE).pack(side="left", padx=12, pady=6)
else:
    tk.Label(frame_header, text="HORSE", bg=C_SURFACE, fg=C_ACENTO,
             font=(C_MONO, 13, "bold")).pack(side="left", padx=14)
tk.Frame(frame_header, bg=C_BORDER2, width=1).pack(side="left", fill="y", pady=8)
fh = tk.Frame(frame_header, bg=C_SURFACE); fh.pack(side="left", padx=12)
tk.Label(fh, text="REVISOR DE TURNO  —  DEMM", bg=C_SURFACE, fg=C_TEXT,
         font=(C_MONO, 11, "bold")).pack(anchor="w")
lbl_turno = tk.Label(fh, text="Sin turno cargado", bg=C_SURFACE, fg=C_TEXT_SUB,
                     font=(C_MONO, 9)); lbl_turno.pack(anchor="w")

# Progreso en header
fh2 = tk.Frame(frame_header, bg=C_SURFACE); fh2.pack(side="right", padx=18)
lbl_prog = tk.Label(fh2, text="0 / 0", bg=C_SURFACE, fg=C_TEXT_SUB,
                    font=(C_MONO, 9)); lbl_prog.pack(anchor="e")
frame_pbar = tk.Frame(fh2, bg=C_SURFACE); frame_pbar.pack(anchor="e")
canvas_pbar = tk.Canvas(frame_pbar, width=160, height=4, bg=C_BORDER,
                        highlightthickness=0); canvas_pbar.pack()
rect_pbar   = canvas_pbar.create_rectangle(0, 0, 0, 4, fill=C_ACENTO, outline="")

tk.Frame(root, bg=C_BORDER, height=1).pack(fill="x")


# ── MAIN LAYOUT ──
frame_main = tk.Frame(root, bg=C_BG); frame_main.pack(fill="both", expand=True)

# ── PANEL IZQUIERDO ──
frame_lista = tk.Frame(frame_main, bg=C_SURFACE, width=260)
frame_lista.pack(side="left", fill="y")
frame_lista.pack_propagate(False)
tk.Frame(frame_lista, bg=C_BG, width=1).pack(side="right", fill="y")

# Cabecera lista
fl_hdr = tk.Frame(frame_lista, bg=C_SURFACE, padx=12, pady=8)
fl_hdr.pack(fill="x")
tk.Label(fl_hdr, text="ENSAYOS DEL TURNO", bg=C_SURFACE, fg=C_TEXT_SUB,
         font=(C_MONO, 8, "bold")).pack(anchor="w", pady=(0,6))

# Filtros
frame_filtros = tk.Frame(fl_hdr, bg=C_SURFACE); frame_filtros.pack(anchor="w")
var_filtro = tk.StringVar(value="todos")

def btn_filtro(parent, texto, valor):
    def cmd():
        var_filtro.set(valor)
        refrescar_lista()
        for b in btns_filtro:
            b.config(bg=C_SURFACE2 if b._valor != valor else C_ACENTO,
                     fg=C_TEXT_SUB if b._valor != valor else "white")
    b = tk.Button(parent, text=texto, command=cmd, bg=C_SURFACE2, fg=C_TEXT_SUB,
                  font=(C_MONO, 8), relief="flat", bd=0, padx=6, pady=2, cursor="hand2")
    b._valor = valor
    return b

btns_filtro = []
for txt, val in [("Todos","todos"),("⚠ Sospec.","sospechosos"),("Pendientes","pendientes"),("✔ Revisados","revisados"),("✖ Inválidos","invalidos")]:
    b = btn_filtro(frame_filtros, txt, val)
    b.pack(side="left", padx=2)
    btns_filtro.append(b)
btns_filtro[0].config(bg=C_ACENTO, fg="white")

sep_h(frame_lista)

# Lista scrollable
frame_lista_scroll = tk.Frame(frame_lista, bg=C_SURFACE)
frame_lista_scroll.pack(fill="both", expand=True)
scrollbar = tk.Scrollbar(frame_lista_scroll, bg=C_SURFACE2, troughcolor=C_SURFACE,
                          width=6, relief="flat")
scrollbar.pack(side="right", fill="y")
listbox = tk.Listbox(frame_lista_scroll, bg=C_SURFACE, fg=C_TEXT_SUB,
                     font=(C_MONO, 8), relief="flat", bd=0,
                     selectbackground=C_ACENTO, selectforeground="white",
                     activestyle="none", highlightthickness=0,
                     yscrollcommand=scrollbar.set)
listbox.pack(fill="both", expand=True)
scrollbar.config(command=listbox.yview)

# Footer lista — contadores
sep_h(frame_lista)
frame_contadores = tk.Frame(frame_lista, bg=C_SURFACE, pady=8)
frame_contadores.pack(fill="x")
lbl_cnt_total = tk.Label(frame_contadores, text="0\nTOTAL", bg=C_SURFACE, fg=C_TEXT_SUB,
                          font=(C_MONO, 9), justify="center"); lbl_cnt_total.pack(side="left", expand=True)
lbl_cnt_buenos= tk.Label(frame_contadores, text="0\nBUENOS", bg=C_SURFACE, fg=C_BUENO,
                          font=(C_MONO, 9), justify="center"); lbl_cnt_buenos.pack(side="left", expand=True)
lbl_cnt_malos = tk.Label(frame_contadores, text="0\nMALOS", bg=C_SURFACE, fg=C_MALO,
                          font=(C_MONO, 9), justify="center"); lbl_cnt_malos.pack(side="left", expand=True)
lbl_cnt_sosp  = tk.Label(frame_contadores, text="0\nSOSPECH.", bg=C_SURFACE, fg=C_SOSP,
                          font=(C_MONO, 9), justify="center"); lbl_cnt_sosp.pack(side="left", expand=True)
lbl_cnt_inv   = tk.Label(frame_contadores, text="0\nINVÁL.", bg=C_SURFACE, fg=C_INVALIDO,
                          font=(C_MONO, 9), justify="center"); lbl_cnt_inv.pack(side="left", expand=True)

sep_h(frame_lista)

# Botón exportar
hacer_boton(frame_lista, "💾  EXPORTAR DATASET", lambda: exportar_dataset(),
            bg=C_ACENTO, fg="white", ancho=24, fs=9, bold=True
            ).pack(pady=8, padx=10, fill="x")

# Umbral curtosis ajustable
frame_umbral = tk.Frame(frame_lista, bg=C_SURFACE, padx=12)
frame_umbral.pack(fill="x", pady=(0,8))
tk.Label(frame_umbral, text=f"Umbral curtosis sospechoso:", bg=C_SURFACE,
         fg=C_TEXT_SUB, font=(C_MONO, 8)).pack(anchor="w")
var_umbral = tk.DoubleVar(value=KURT_UMBRAL_INICIAL)
scale_umbral = tk.Scale(frame_umbral, from_=3.0, to=20.0, resolution=0.5,
                         orient="horizontal", variable=var_umbral,
                         bg=C_SURFACE, fg=C_TEXT_SUB, highlightthickness=0,
                         font=(C_MONO, 8), length=200,
                         command=lambda v: (kurt_umbral.__setitem__(0, float(v)), refrescar_lista()))
scale_umbral.pack(fill="x")


# ── PANEL CENTRAL ──
frame_central = tk.Frame(frame_main, bg=C_BG)
frame_central.pack(side="left", fill="both", expand=True)

# Sub-header central
frame_viz_hdr = tk.Frame(frame_central, bg=C_SURFACE, height=42)
frame_viz_hdr.pack(fill="x")
frame_viz_hdr.pack_propagate(False)
lbl_filename  = tk.Label(frame_viz_hdr, text="Sin ensayo seleccionado",
                          bg=C_SURFACE, fg=C_TEXT, font=(C_MONO, 10, "bold"))
lbl_filename.pack(side="left", padx=14)
lbl_meta      = tk.Label(frame_viz_hdr, text="", bg=C_SURFACE, fg=C_TEXT_SUB,
                          font=(C_MONO, 8)); lbl_meta.pack(side="left")
lbl_badge     = tk.Label(frame_viz_hdr, text="", bg=C_SURFACE, fg=C_SOSP,
                          font=(C_MONO, 9, "bold")); lbl_badge.pack(side="left", padx=8)

frame_nav = tk.Frame(frame_viz_hdr, bg=C_SURFACE); frame_nav.pack(side="right", padx=12)
hacer_boton(frame_nav, "← Ant.", lambda: navegar(-1), fs=8, ancho=8).pack(side="left", padx=2)
hacer_boton(frame_nav, "Sig. →", lambda: navegar(+1), fs=8, ancho=8).pack(side="left", padx=2)

tk.Frame(frame_central, bg=C_BORDER, height=1).pack(fill="x")

# Gráfico — 2 paneles
frame_graf = tk.Frame(frame_central, bg=C_BG)
frame_graf.pack(fill="both", expand=True, padx=8, pady=6)

fig, (ax_señal, ax_fft) = plt.subplots(2, 1, figsize=(8, 5.5), dpi=96, facecolor="#0e0f11")
for ax in (ax_señal, ax_fft):
    ax.set_facecolor("#161820")
    ax.tick_params(colors=C_TEXT_SUB, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(C_BORDER2)
    ax.grid(True, alpha=0.15, color=C_BORDER2)
fig.tight_layout(pad=1.2)

canvas_fig = FigureCanvasTkAgg(fig, master=frame_graf)
canvas_fig.get_tk_widget().pack(fill="both", expand=True)

# Barra de decisión
tk.Frame(frame_central, bg=C_BORDER, height=1).pack(fill="x")
frame_decision = tk.Frame(frame_central, bg=C_SURFACE, height=54)
frame_decision.pack(fill="x")
frame_decision.pack_propagate(False)

tk.Label(frame_decision, text="DECISIÓN:", bg=C_SURFACE, fg=C_TEXT_SUB,
         font=(C_MONO, 9)).pack(side="left", padx=(14,8))

btn_bueno   = hacer_boton(frame_decision, "✔  BUENO  [G]",   lambda: decidir("BUENO"),
                           bg=C_BUENO_BG, fg=C_BUENO, ancho=16, fs=10, bold=True)
btn_malo    = hacer_boton(frame_decision, "✖  MALO   [M]",   lambda: decidir("MALO"),
                           bg=C_MALO_BG,  fg=C_MALO,  ancho=16, fs=10, bold=True)
btn_ignorar = hacer_boton(frame_decision, "—  IGNORAR  [I]", lambda: decidir("IGNORAR"),
                           bg=C_SURFACE2, fg=C_TEXT_SUB, ancho=16, fs=10)
for b in (btn_bueno, btn_malo, btn_ignorar):
    b.pack(side="left", padx=4, pady=8)

lbl_autoguardar = tk.Label(frame_decision,
    text="● Autoguardado en dataset.csv", bg=C_SURFACE, fg=C_TEXT_DIM,
    font=(C_MONO, 8)); lbl_autoguardar.pack(side="right", padx=16)


# ── PANEL DERECHO ──
frame_stats = tk.Frame(frame_main, bg=C_SURFACE, width=270)
frame_stats.pack(side="left", fill="y")
frame_stats.pack_propagate(False)
tk.Frame(frame_stats, bg=C_BG, width=1).pack(side="left", fill="y")

fr_st_inner = tk.Frame(frame_stats, bg=C_SURFACE, padx=12)
fr_st_inner.pack(fill="both", expand=True)

tk.Label(fr_st_inner, text="ESTADÍSTICOS  —  TRAMOS 0.5s",
         bg=C_SURFACE, fg=C_TEXT_SUB, font=(C_MONO, 8, "bold")
         ).pack(anchor="w", pady=(10,6))
sep_h(fr_st_inner)

# Canvas scrollable para estadísticos
stats_canvas   = tk.Canvas(fr_st_inner, bg=C_SURFACE, highlightthickness=0)
stats_scroll   = tk.Scrollbar(fr_st_inner, orient="vertical", command=stats_canvas.yview,
                               bg=C_SURFACE2, troughcolor=C_SURFACE, width=5, relief="flat")
stats_canvas.configure(yscrollcommand=stats_scroll.set)
stats_scroll.pack(side="right", fill="y")
stats_canvas.pack(side="left", fill="both", expand=True)
frame_stats_content = tk.Frame(stats_canvas, bg=C_SURFACE)
stats_canvas.create_window((0,0), window=frame_stats_content, anchor="nw")
frame_stats_content.bind("<Configure>",
    lambda e: stats_canvas.configure(scrollregion=stats_canvas.bbox("all")))


# ==========================
# LÓGICA DE LA LISTA
# ==========================
indices_visibles = []   # índices en `ensayos` según filtro activo


def refrescar_lista():
    filtro = var_filtro.get()
    indices_visibles.clear()
    listbox.delete(0, "end")

    for i, res in enumerate(ensayos):
        if "error" in res:
            if filtro in ("todos","sospechosos","pendientes"):
                indices_visibles.append(i)
                listbox.insert("end", f"  ✖ {res['nombre'][:28]}")
                listbox.itemconfig("end", fg="#ef4444")
            continue

        sosp    = es_sospechoso(res)
        et      = res["etiqueta"]
        val     = res.get("validacion", {})
        v_nivel = val.get("nivel", "VÁLIDO")
        invalido= v_nivel == "INVÁLIDO"
        dudoso  = v_nivel == "DUDOSO"

        mostrar = (
            filtro == "todos" or
            (filtro == "sospechosos" and sosp and et is None) or
            (filtro == "pendientes"  and et is None) or
            (filtro == "revisados"   and et is not None) or
            (filtro == "invalidos"   and invalido)
        )
        if not mostrar:
            continue

        indices_visibles.append(i)
        kurt_str = f"{res['kurt_max']:.2f}"
        hora     = res["hora"]

        if et == "BUENO":
            sym = "✔"; color = C_BUENO
        elif et == "MALO":
            sym = "✖"; color = C_MALO
        elif et == "IGNORAR":
            sym = "—"; color = C_TEXT_DIM
        elif invalido:
            sym = "✖"; color = C_INVALIDO
        elif dudoso:
            sym = "~"; color = C_DUDOSO
        elif sosp:
            sym = "⚠"; color = C_SOSP
        else:
            sym = "·"; color = C_TEXT_SUB

        nombre_corto = res["nombre"].replace("engrane_","").replace(".csv","")[:22]
        listbox.insert("end", f"  {sym} {nombre_corto}  {kurt_str}")
        listbox.itemconfig("end", fg=color)

    # Actualizar contadores
    total    = len([r for r in ensayos if "error" not in r])
    buenos   = len([r for r in ensayos if r.get("etiqueta") == "BUENO"])
    malos    = len([r for r in ensayos if r.get("etiqueta") == "MALO"])
    sosps    = len([r for r in ensayos if "error" not in r and
                    es_sospechoso(r) and r.get("etiqueta") is None])
    invalidos= len([r for r in ensayos if "error" not in r and
                    r.get("validacion", {}).get("nivel") == "INVÁLIDO"])
    lbl_cnt_total.config(text=f"{total}\nTOTAL")
    lbl_cnt_buenos.config(text=f"{buenos}\nBUENOS")
    lbl_cnt_malos.config(text=f"{malos}\nMALOS")
    lbl_cnt_sosp.config(text=f"{sosps}\nSOSPECH.")
    lbl_cnt_inv.config(text=f"{invalidos}\nINVÁL.")

    # Progreso
    revisados = buenos + malos + len([r for r in ensayos if r.get("etiqueta")=="IGNORAR"])
    pct       = revisados / total if total > 0 else 0
    canvas_pbar.coords(rect_pbar, 0, 0, int(160 * pct), 4)
    lbl_prog.config(text=f"{revisados} / {total}")


def on_lista_select(event=None):
    sel = listbox.curselection()
    if not sel or not indices_visibles:
        return
    i_vis = sel[0]
    if i_vis >= len(indices_visibles):
        return
    idx_actual[0] = indices_visibles[i_vis]
    mostrar_ensayo(idx_actual[0])


listbox.bind("<<ListboxSelect>>", on_lista_select)


# ==========================
# VISUALIZACIÓN
# ==========================
def mostrar_ensayo(idx):
    if idx < 0 or idx >= len(ensayos):
        return
    res = ensayos[idx]

    if "error" in res:
        lbl_filename.config(text=res["nombre"], fg=C_MALO)
        lbl_meta.config(text=f"Error: {res['error']}")
        lbl_badge.config(text="✖ ERROR")
        return

    señal    = res["señal"]
    gmf      = res["gmf"]
    fe       = res["fe"]
    fr       = res["fr"]
    t_sep    = res["t_sep"]
    t_freno  = res["t_freno"]
    nombre   = res["nombre"]
    hora     = res["hora"]
    sosp     = es_sospechoso(res)
    et       = res["etiqueta"]

    # Badge
    if et == "BUENO":
        lbl_badge.config(text="✔ BUENO", fg=C_BUENO)
    elif et == "MALO":
        lbl_badge.config(text="✖ MALO", fg=C_MALO)
    elif et == "IGNORAR":
        lbl_badge.config(text="— IGNORAR", fg=C_IGNORAR)
    elif sosp:
        lbl_badge.config(text="⚠ SOSPECHOSO", fg=C_SOSP)
    else:
        lbl_badge.config(text="· PENDIENTE", fg=C_TEXT_DIM)

    lbl_filename.config(text=nombre.replace(".csv",""), fg=C_TEXT)
    val  = res.get("validacion", {})
    val_nivel = val.get("nivel", "—")
    val_score = val.get("score", 0)
    val_color = val.get("color", C_TEXT_DIM)
    lbl_meta.config(
        text=f"{PINONES[clave_pinon[0]]['dientes']}d  ·  {hora}  ·  "
             f"{len(señal)/FS:.2f}s  ·  Kurt_max={res['kurt_max']:.2f}  ·  "
             f"Grab: {val_nivel} ({val_score}/100)",
        fg=val_color)

    t_full = np.arange(len(señal)) / FS
    n      = len(señal)

    # ── Panel 1: Señal con zonas ──
    ax_señal.clear()
    ax_señal.set_facecolor("#161820")
    ax_señal.tick_params(colors=C_TEXT_SUB, labelsize=7)
    for sp in ax_señal.spines.values(): sp.set_color(C_BORDER2)
    ax_señal.grid(True, alpha=0.15, color=C_BORDER2)

    ax_señal.plot(t_full, señal, color=C_TEXT_DIM, linewidth=0.35, zorder=1)
    ax_señal.axvspan(res["i_emp_inicio"]/FS, res["i_emp_fin"]/FS,
                     alpha=0.08, color="#ef4444")
    ax_señal.axvspan(res["i_ret_inicio"]/FS, res["i_ret_fin"]/FS,
                     alpha=0.08, color="#4f8ef7")
    # ── Zonas de fondo ──
    # Empuje completa (rojo tenue)
    ax_señal.axvspan(res["i_emp_inicio"]/FS, res["i_emp_fin"]/FS,
                     alpha=0.07, color="#ef4444")
    # Retroceso completa (azul muy tenue)
    ax_señal.axvspan(res["i_ret_inicio"]/FS, res["i_ret_fin"]/FS,
                     alpha=0.05, color="#4f8ef7")

    # ── Zona de transición excluida (gris oscuro sobre la zona de retroceso) ──
    _i_estab = res.get("i_ret_estab", res["i_ret_inicio"])
    if _i_estab > res["i_ret_inicio"] + int(FS * 0.02):
        ax_señal.axvspan(res["i_ret_inicio"]/FS, _i_estab/FS,
                         alpha=0.45, color="#0a0b0e",
                         label="Trans. excluida")

    # ── Zona de retroceso estabilizada (azul más intenso) ──
    ax_señal.axvspan(_i_estab/FS, res["i_ret_fin"]/FS,
                     alpha=0.13, color="#4f8ef7")

    # ── Tramo de análisis empuje: centro de zona empuje ──
    t_te = res["i_emp_inicio"]/FS + (res["i_emp_fin"] - res["i_emp_inicio"])/(2*FS) - TRAMO_SEG/2

    # ── Tramo de análisis retroceso: centro de zona YA estabilizada ──
    _largo_ret_estab = res["i_ret_fin"] - _i_estab
    t_tr = _i_estab/FS + _largo_ret_estab/(2*FS) - TRAMO_SEG/2
    t_tr = max(t_tr, _i_estab/FS)   # nunca antes del inicio estable

    ax_señal.axvspan(t_te, t_te + TRAMO_SEG, alpha=0.30, color="#ef4444",
                     label="Análisis emp")
    ax_señal.axvspan(t_tr, t_tr + TRAMO_SEG, alpha=0.30, color="#4f8ef7",
                     label="Análisis ret")

    # ── Líneas de referencia ──
    ax_señal.axvline(t_sep,   color=C_SOSP,   linewidth=1.2, linestyle="--",
                     label=f"Sep {t_sep:.2f}s")
    _n_s       = len(res["señal"])
    _i_ret_fin = res["i_ret_fin"]
    _freno_real= _i_ret_fin < int(_n_s * 0.95) - int(FS * 0.05)
    _freno_lbl = f"Freno {t_freno:.2f}s" if _freno_real else f"Fin ret {t_freno:.2f}s"
    _freno_col = "#a855f7" if _freno_real else "#6b7280"
    ax_señal.axvline(t_freno, color=_freno_col, linewidth=1.2, linestyle="--",
                     label=_freno_lbl)
    if _i_estab > res["i_ret_inicio"] + int(FS * 0.02):
        ax_señal.axvline(_i_estab/FS, color="#34d399", linewidth=1.5,
                         linestyle=":", label=f"Estab {_i_estab/FS:.2f}s")

    _recorte   = (_i_estab - res["i_ret_inicio"]) / FS
    _estab_str = f"   estab: {_i_estab/FS:.3f}s (+{_recorte:.3f}s)" if _recorte > 0.02 else ""
    ax_señal.set_title(
        f"Señal completa  |  sep: {t_sep:.3f}s   freno: {t_freno:.3f}s{_estab_str}",
        fontsize=8, color=C_TEXT_SUB)
    ax_señal.set_ylabel("Amplitud", fontsize=7, color=C_TEXT_SUB)
    ax_señal.legend(fontsize=6, loc="upper right",
                    facecolor=C_SURFACE2, edgecolor=C_BORDER2, labelcolor=C_TEXT_SUB)

    # ── Panel 2: Espectro en órdenes ──
    ax_fft.clear()
    ax_fft.set_facecolor("#161820")
    ax_fft.tick_params(colors=C_TEXT_SUB, labelsize=7)
    for sp in ax_fft.spines.values(): sp.set_color(C_BORDER2)
    ax_fft.grid(True, alpha=0.15, color=C_BORDER2)

    orden_gmf = fe["orden_gmf"]
    o_max     = orden_gmf * 3.5
    for datos, color, lbl_txt in [
        (fe, "#f87171", f"Empuje  K={fe['kurt_limp']:.2f}"),
        (fr, "#60a5fa", f"Retroceso  K={fr['kurt_limp']:.2f}"),
    ]:
        mask  = (datos["ordenes"] > 0) & (datos["ordenes"] <= o_max)
        ords  = datos["ordenes"][mask]
        db    = datos["fft_db"][mask]
        suav  = datos["fft_suav"][mask]
        ax_fft.plot(ords, db,   color=color, linewidth=0.7, alpha=0.6, label=lbl_txt)
        ax_fft.plot(ords, suav, color=color, linewidth=1.1, linestyle="--", alpha=0.9)
        ax_fft.fill_between(ords, suav, db, alpha=0.10, color=color)

    for k, ls in [(1,"--"),(2,":"),(3,":")]:
        o_k = orden_gmf * k
        if o_k <= o_max:
            ax_fft.axvline(o_k, color=C_SOSP, linestyle=ls, linewidth=0.8,
                           alpha=0.6, label=f"{k}×GMF" if k==1 else None)

    ax_fft.set_title(
        f"Espectro en órdenes (dB)  |  GMF = orden {orden_gmf:.0f}",
        fontsize=8, color=C_TEXT_SUB)
    ax_fft.set_xlabel("Orden  (f / f_rot)", fontsize=7, color=C_TEXT_SUB)
    ax_fft.set_ylabel("dB", fontsize=7, color=C_TEXT_SUB)
    ax_fft.legend(fontsize=7, loc="upper right",
                  facecolor=C_SURFACE2, edgecolor=C_BORDER2, labelcolor=C_TEXT_SUB)

    fig.tight_layout(pad=1.2)
    canvas_fig.draw()

    # ── Panel estadísticos derecho ──
    for w in frame_stats_content.winfo_children():
        w.destroy()

    def stat_row(frame, nombre, val_e, val_r, color_e="#f87171", color_r="#60a5fa", warn=False):
        f = tk.Frame(frame, bg=C_SURFACE); f.pack(fill="x", pady=1)
        tk.Label(f, text=nombre, bg=C_SURFACE, fg=C_TEXT_DIM,
                 font=(C_MONO, 8), width=18, anchor="w").pack(side="left")
        c_e = C_SOSP if warn and float(str(val_e).replace(" dB","")) > 5 else color_e
        tk.Label(f, text=str(val_e), bg=C_SURFACE, fg=c_e,
                 font=(C_MONO, 8, "bold"), width=10, anchor="e").pack(side="left")
        tk.Label(f, text=str(val_r), bg=C_SURFACE, fg=color_r,
                 font=(C_MONO, 8, "bold"), width=10, anchor="e").pack(side="left")

    def section(frame, titulo):
        tk.Label(frame, text=titulo, bg=C_SURFACE, fg=C_TEXT_DIM,
                 font=(C_MONO, 7, "bold")).pack(anchor="w", pady=(8,2))
        tk.Frame(frame, bg=C_BORDER, height=1).pack(fill="x")

    # Cabecera columnas
    f_hdr = tk.Frame(frame_stats_content, bg=C_SURFACE)
    f_hdr.pack(fill="x", pady=(4,2))
    tk.Label(f_hdr, text="", bg=C_SURFACE, width=18).pack(side="left")
    tk.Label(f_hdr, text="EMPUJE", bg=C_SURFACE, fg="#f87171",
             font=(C_MONO, 8, "bold"), width=10, anchor="e").pack(side="left")
    tk.Label(f_hdr, text="RETROCESO", bg=C_SURFACE, fg="#60a5fa",
             font=(C_MONO, 8, "bold"), width=10, anchor="e").pack(side="left")

    # Mostrar número de ventanas y recorte de estabilización
    n_vent_e  = fe.get("n_ventanas", 1)
    n_vent_r  = fr.get("n_ventanas", 1)
    _i_es     = res.get("i_ret_estab", res["i_ret_inicio"])
    _recorte  = (_i_es - res["i_ret_inicio"]) / FS
    _rec_str  = f"  +{_recorte:.3f}s excluidos" if _recorte > 0.02 else "  sin recorte"
    f_nv = tk.Frame(frame_stats_content, bg=C_SURFACE); f_nv.pack(fill="x", pady=(4,0))
    tk.Label(f_nv,
             text=f"Ventanas: {n_vent_e} emp / {n_vent_r} ret",
             bg=C_SURFACE, fg=C_TEXT_DIM, font=(C_MONO, 7, "italic")).pack(anchor="w")
    tk.Label(f_nv,
             text=f"Ret trans:{_rec_str}",
             bg=C_SURFACE, fg="#34d399" if _recorte > 0.02 else C_TEXT_DIM,
             font=(C_MONO, 7, "italic")).pack(anchor="w")

    section(frame_stats_content, "── RMS  /  NIVEL GLOBAL (Eq.12)")
    stat_row(frame_stats_content, "Original",    f"{fe['rms']:.5f}",      f"{fr['rms']:.5f}")
    stat_row(frame_stats_content, "Limpio",       f"{fe['rms_limp']:.5f}", f"{fr['rms_limp']:.5f}")

    section(frame_stats_content, "── CURTOSIS  (Eq.4 — ref ≈ 3)")
    c_ke = C_MALO if fe["kurt_limp_worst"] >= kurt_umbral[0] else "#f87171"
    c_kr = C_MALO if fr["kurt_limp_worst"] >= kurt_umbral[0] else "#60a5fa"
    stat_row(frame_stats_content, "Original",    f"{fe['kurt']:.3f}",      f"{fr['kurt']:.3f}")
    stat_row(frame_stats_content, "Limpia (mean)", f"{fe['kurt_limp']:.3f}", f"{fr['kurt_limp']:.3f}",
             color_e=c_ke, color_r=c_kr)
    c_ke2 = C_MALO if fe["kurt_limp_worst"] >= kurt_umbral[0] else "#f87171"
    c_kr2 = C_MALO if fr["kurt_limp_worst"] >= kurt_umbral[0] else "#60a5fa"
    stat_row(frame_stats_content, "Limpia (peor vent)",
             f"{fe['kurt_limp_worst']:.3f}", f"{fr['kurt_limp_worst']:.3f}",
             color_e=c_ke2, color_r=c_kr2)
    stat_row(frame_stats_content, "Limpia p90",
             f"{fe['kurt_limp_p90']:.3f}", f"{fr['kurt_limp_p90']:.3f}")

    section(frame_stats_content, "── CEPSTRUM  (Eq.8)")
    c_cep_e = C_MALO if fe["cepstrum_max"] >= ceps_umbral[0] else C_TEXT_SUB
    c_cep_r = C_MALO if fr["cepstrum_max"] >= ceps_umbral[0] else C_TEXT_SUB
    stat_row(frame_stats_content, "Cp máximo (peor)",
             f"{fe['cepstrum_max']:.4f}", f"{fr['cepstrum_max']:.4f}",
             color_e=c_cep_e, color_r=c_cep_r)
    stat_row(frame_stats_content, "Cp p90",
             f"{fe['cepstrum_max_p90']:.4f}", f"{fr['cepstrum_max_p90']:.4f}")

    section(frame_stats_content, "── FONDO ESPECTRO  (Eq.6 — w=24 m=8)")
    stat_row(frame_stats_content, "SB (dB)",
             f"{fe['fondo_espectro']:.4f}", f"{fr['fondo_espectro']:.4f}")

    section(frame_stats_content, "── NIVEL DE ORDEN  (Eq.13)")
    stat_row(frame_stats_content, f"GMF ord {fe['orden_gmf']:.0f} (dB)",
             f"{fe['nivel_gmf']:.2f}", f"{fr['nivel_gmf']:.2f}")

    section(frame_stats_content, "── DENSIDAD RUIDO  (Eq.1)")
    stat_row(frame_stats_content, "Área sobre umbral",
             f"{fe['densidad_ruido']:.3f}", f"{fr['densidad_ruido']:.3f}")

    section(frame_stats_content, "── Nº FREC. SOBRE UMBRAL  (Eq.3)")
    stat_row(frame_stats_content, "Puntos sobre seuil",
             str(fe['n_frec_sobre_umbral']), str(fr['n_frec_sobre_umbral']))

    section(frame_stats_content, "── NIVEL MÁX. ESPECTRAL  (Eq.11)")
    stat_row(frame_stats_content, "Δl_max (dB)",
             f"{fe['nivel_max_espectral']:.3f}", f"{fr['nivel_max_espectral']:.3f}")

    section(frame_stats_content, "── AMPLITUD MÁX. ESPECTRAL  (Eq.9)")
    stat_row(frame_stats_content, "Δa_max (lineal)",
             f"{fe['amplitud_max_espectral']:.5f}", f"{fr['amplitud_max_espectral']:.5f}")

    section(frame_stats_content, "── SIDEBANDS")
    c_sb_e = C_MALO if fe["sb_ratio"] >= sb_umbral[0] else C_TEXT_SUB
    c_sb_r = C_MALO if fr["sb_ratio"] >= sb_umbral[0] else C_TEXT_SUB
    stat_row(frame_stats_content, "SB ratio (peor)",
             f"{fe['sb_ratio']:.3f}", f"{fr['sb_ratio']:.3f}",
             color_e=c_sb_e, color_r=c_sb_r)
    stat_row(frame_stats_content, "SB p90",
             f"{fe['sb_ratio_p90']:.3f}", f"{fr['sb_ratio_p90']:.3f}")

    section(frame_stats_content, "── IMPULSOS")
    inf_e = res["inf_e"]; inf_r = res["inf_r"]
    stat_row(frame_stats_content, "Defecto",
             str(inf_e["n_defecto"]), str(inf_r["n_defecto"]),
             color_e=C_MALO if inf_e["n_defecto"]>0 else "#f87171",
             color_r=C_MALO if inf_r["n_defecto"]>0 else "#60a5fa")
    stat_row(frame_stats_content, "Ruido",
             str(inf_e["n_ruido"]), str(inf_r["n_ruido"]))
    stat_row(frame_stats_content, "Sidebands",
             "✔ sí" if inf_e["hay_sidebands"] else "✖ no",
             "✔ sí" if inf_r["hay_sidebands"] else "✖ no",
             color_e=C_MALO if inf_e["hay_sidebands"] else C_TEXT_DIM,
             color_r=C_MALO if inf_r["hay_sidebands"] else C_TEXT_DIM)

    section(frame_stats_content, "── VALIDACIÓN DE GRABACIÓN")
    val       = res.get("validacion", {})
    v_nivel   = val.get("nivel",   "—")
    v_score   = val.get("score",   0)
    v_color   = val.get("color",   C_TEXT_DIM)
    v_razones = val.get("razones", [])
    v_bonos   = val.get("bonos",   [])

    # Badge principal
    f_vbadge = tk.Frame(frame_stats_content, bg=C_SURFACE)
    f_vbadge.pack(fill="x", pady=(2,4))
    tk.Label(f_vbadge,
             text=f"  {v_nivel}  —  Score: {v_score}/100",
             bg=v_color, fg="white",
             font=(C_MONO, 8, "bold"),
             padx=6, pady=2).pack(side="left")

    # Razones de rechazo / penalización
    for r_txt in v_razones:
        f_r = tk.Frame(frame_stats_content, bg=C_SURFACE)
        f_r.pack(fill="x", pady=1)
        tk.Label(f_r, text=r_txt, bg=C_SURFACE, fg=C_INVALIDO,
                 font=(C_MONO, 7), anchor="w", wraplength=240,
                 justify="left").pack(fill="x", padx=4)

    # Criterios positivos
    for b_txt in v_bonos:
        f_b = tk.Frame(frame_stats_content, bg=C_SURFACE)
        f_b.pack(fill="x", pady=1)
        col_b = C_VALIDO if b_txt.startswith("✔") else C_DUDOSO
        tk.Label(f_b, text=b_txt, bg=C_SURFACE, fg=col_b,
                 font=(C_MONO, 7), anchor="w", wraplength=240,
                 justify="left").pack(fill="x", padx=4)

    section(frame_stats_content, "── HISTORIAL TURNO")
    for res2 in ensayos[-10:]:
        if "error" in res2:
            continue
        et2   = res2.get("etiqueta")
        k2    = res2["kurt_max"]
        color = C_BUENO if et2=="BUENO" else C_MALO if et2=="MALO" else C_SOSP if es_sospechoso(res2) else C_TEXT_DIM
        marca = "▶ " if res2 is res else "   "
        f2 = tk.Frame(frame_stats_content, bg=C_SURFACE); f2.pack(fill="x", pady=1)
        tk.Label(f2, text=f"{marca}{res2['nombre'][:16]}", bg=C_SURFACE, fg=color,
                 font=(C_MONO, 7), anchor="w", width=20).pack(side="left")
        tk.Label(f2, text=f"{k2:.2f}", bg=C_SURFACE, fg=color,
                 font=(C_MONO, 7, "bold"), anchor="e", width=6).pack(side="right")


def navegar(delta):
    if not ensayos:
        return
    nuevo = idx_actual[0] + delta
    nuevo = max(0, min(len(ensayos)-1, nuevo))
    idx_actual[0] = nuevo
    mostrar_ensayo(nuevo)
    # Sincronizar listbox
    for i, idx in enumerate(indices_visibles):
        if idx == nuevo:
            listbox.selection_clear(0, "end")
            listbox.selection_set(i)
            listbox.see(i)
            break


# ==========================
# DECISIONES
# ==========================
def decidir(etiqueta):
    if not ensayos or idx_actual[0] >= len(ensayos):
        return
    res = ensayos[idx_actual[0]]
    if "error" in res:
        return
    res["etiqueta"] = etiqueta
    refrescar_lista()
    mostrar_ensayo(idx_actual[0])
    # Avanzar automáticamente al siguiente sin etiqueta
    for i in range(idx_actual[0]+1, len(ensayos)):
        if ensayos[i].get("etiqueta") is None and "error" not in ensayos[i]:
            idx_actual[0] = i
            mostrar_ensayo(i)
            for j, idx in enumerate(indices_visibles):
                if idx == i:
                    listbox.selection_clear(0, "end")
                    listbox.selection_set(j)
                    listbox.see(j)
                    break
            return


# Atajos de teclado
root.bind("<g>", lambda e: decidir("BUENO"))
root.bind("<G>", lambda e: decidir("BUENO"))
root.bind("<m>", lambda e: decidir("MALO"))
root.bind("<M>", lambda e: decidir("MALO"))
root.bind("<i>", lambda e: decidir("IGNORAR"))
root.bind("<I>", lambda e: decidir("IGNORAR"))
root.bind("<Left>",  lambda e: navegar(-1))
root.bind("<Right>", lambda e: navegar(+1))


# ==========================
# EXPORTAR DATASET
# ==========================
def exportar_dataset():
    etiquetados = [r for r in ensayos if "error" not in r and r.get("etiqueta") in ("BUENO","MALO")]
    if not etiquetados:
        messagebox.showwarning("Sin datos", "No hay ensayos etiquetados como BUENO o MALO aún.")
        return

    filas = []
    for res in etiquetados:
        fe = res["fe"]; fr = res["fr"]
        val = res.get("validacion", {})
        fila = {
            "archivo":                    res["nombre"],
            "hora":                       res["hora"],
            "pinon":                      clave_pinon[0],
            "etiqueta":                   res["etiqueta"],
            # ── Validación de grabación ──
            "val_score":                  val.get("score", 100),
            "val_nivel":                  val.get("nivel", "VÁLIDO"),
            # ── Features por ventanas deslizantes ──────────────────────────
            # Para cada criterio: _max (peor ventana), _mean (promedio), _p90
            # Señal original (con impulsos) y limpia (sin ruido externo)
            # ── RMS / Nivel global (Eq. 12) ──
            "rms_emp":             fe["rms_mean"],
            "rms_p90_emp":         fe["rms_p90"],
            "rms_max_emp":         fe["rms_max"],
            "rms_ret":             fr["rms_mean"],
            "rms_p90_ret":         fr["rms_p90"],
            "rms_max_ret":         fr["rms_max"],
            # ── Curtosis (Eq. 4) ──
            "kurt_emp":            fe["kurt_mean"],
            "kurt_p90_emp":        fe["kurt_p90"],
            "kurt_max_emp":        fe["kurt_max"],
            "kurt_limp_emp":       fe["kurt_limp_mean"],
            "kurt_limp_p90_emp":   fe["kurt_limp_p90"],
            "kurt_limp_max_emp":   fe["kurt_limp_max"],
            "kurt_ret":            fr["kurt_mean"],
            "kurt_p90_ret":        fr["kurt_p90"],
            "kurt_max_ret":        fr["kurt_max"],
            "kurt_limp_ret":       fr["kurt_limp_mean"],
            "kurt_limp_p90_ret":   fr["kurt_limp_p90"],
            "kurt_limp_max_ret":   fr["kurt_limp_max"],
            # ── Cepstrum (Eq. 8) ──
            "cepstrum_mean_emp":   fe["cepstrum_max_mean"],
            "cepstrum_p90_emp":    fe["cepstrum_max_p90"],
            "cepstrum_max_emp":    fe["cepstrum_max_max"],
            "cepstrum_mean_ret":   fr["cepstrum_max_mean"],
            "cepstrum_p90_ret":    fr["cepstrum_max_p90"],
            "cepstrum_max_ret":    fr["cepstrum_max_max"],
            # ── Fondo espectro (Eq. 6) ──
            "fondo_espectro_emp":  fe["fondo_espectro_mean"],
            "fondo_espectro_p90_emp": fe["fondo_espectro_p90"],
            "fondo_espectro_ret":  fr["fondo_espectro_mean"],
            "fondo_espectro_p90_ret": fr["fondo_espectro_p90"],
            # ── Nivel GMF (Eq. 13) ──
            "nivel_gmf_emp":       fe["nivel_gmf_mean"],
            "nivel_gmf_max_emp":   fe["nivel_gmf_max"],
            "nivel_gmf_ret":       fr["nivel_gmf_mean"],
            "nivel_gmf_max_ret":   fr["nivel_gmf_max"],
            # ── Densidad de ruido (Eq. 1) ──
            "densidad_ruido_emp":  fe["densidad_ruido_mean"],
            "densidad_ruido_p90_emp": fe["densidad_ruido_p90"],
            "densidad_ruido_ret":  fr["densidad_ruido_mean"],
            "densidad_ruido_p90_ret": fr["densidad_ruido_p90"],
            # ── Nº frec sobre umbral (Eq. 3) ──
            "n_frec_umbral_emp":   fe["n_frec_sobre_umbral_mean"],
            "n_frec_umbral_max_emp": fe["n_frec_sobre_umbral_max"],
            "n_frec_umbral_ret":   fr["n_frec_sobre_umbral_mean"],
            "n_frec_umbral_max_ret": fr["n_frec_sobre_umbral_max"],
            # ── Nivel máx. espectral (Eq. 11) ──
            "nivel_max_esp_emp":   fe["nivel_max_espectral_max"],
            "nivel_max_esp_p90_emp": fe["nivel_max_espectral_p90"],
            "nivel_max_esp_ret":   fr["nivel_max_espectral_max"],
            "nivel_max_esp_p90_ret": fr["nivel_max_espectral_p90"],
            # ── Amplitud máx. espectral (Eq. 9) ──
            "amp_max_esp_emp":     fe["amplitud_max_espectral_max"],
            "amp_max_esp_p90_emp": fe["amplitud_max_espectral_p90"],
            "amp_max_esp_ret":     fr["amplitud_max_espectral_max"],
            "amp_max_esp_p90_ret": fr["amplitud_max_espectral_p90"],
            # ── Sidebands ratio ──
            "sb_ratio_mean_emp":   fe["sb_ratio_mean"],
            "sb_ratio_p90_emp":    fe["sb_ratio_p90"],
            "sb_ratio_max_emp":    fe["sb_ratio_max"],
            "sb_ratio_mean_ret":   fr["sb_ratio_mean"],
            "sb_ratio_p90_ret":    fr["sb_ratio_p90"],
            "sb_ratio_max_ret":    fr["sb_ratio_max"],
            # ── Ratios entre flancos ──
            "ratio_rms":           fe["rms_mean"] / (fr["rms_mean"] + 1e-12),
            "ratio_kurt_limp":     fe["kurt_limp_max"] / (fr["kurt_limp_max"] + 1e-12),
            "ratio_cepstrum":      fe["cepstrum_max_max"] / (fr["cepstrum_max_max"] + 1e-12),
            "ratio_sb":            fe["sb_ratio_max"] / (fr["sb_ratio_max"] + 1e-12),
            # ── Impulsos (Dynae número de choques) ──
            "n_defecto_emp":              res["inf_e"]["n_defecto"],
            "n_ruido_emp":                res["inf_e"]["n_ruido"],
            "sidebands_emp":              int(res["inf_e"]["hay_sidebands"]),
            "n_defecto_ret":              res["inf_r"]["n_defecto"],
            "n_ruido_ret":                res["inf_r"]["n_ruido"],
            "sidebands_ret":              int(res["inf_r"]["hay_sidebands"]),
        }
        filas.append(fila)

    df_out    = pd.DataFrame(filas)
    fecha_hoy = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    carpeta   = os.path.dirname(ensayos[0]["ruta"])
    ruta_out  = os.path.join(carpeta, f"dataset_{fecha_hoy}.csv")
    df_out.to_csv(ruta_out, index=False)

    buenos = sum(1 for r in etiquetados if r["etiqueta"]=="BUENO")
    malos  = sum(1 for r in etiquetados if r["etiqueta"]=="MALO")
    messagebox.showinfo("Dataset exportado",
        f"Guardado en:\n{ruta_out}\n\n"
        f"Registros: {len(filas)}\n"
        f"Buenos: {buenos}  |  Malos: {malos}\n"
        f"Features por registro: {len(df_out.columns)-4}")


# ==========================
# CARGAR TURNO
# ==========================
def cargar_turno():
    carpeta = filedialog.askdirectory(title="Seleccionar carpeta del piñón")
    if not carpeta:
        return

    # Detectar piñón por nombre de carpeta
    nombre_carpeta = os.path.basename(carpeta).upper()
    clave = "ARBOL_SEC_14"
    for k in PINONES:
        if k in nombre_carpeta or nombre_carpeta in k:
            clave = k
            break
    clave_pinon[0] = clave

    # Actualizar combo
    for i, (k, _) in enumerate(OPCIONES):
        if k == clave:
            combo_pinon.current(i)
            break

    archivos = sorted(glob.glob(os.path.join(carpeta, "engrane_*.csv")))
    if not archivos:
        messagebox.showwarning("Sin archivos", f"No se encontraron archivos engrane_*.csv en:\n{carpeta}")
        return

    gmf      = PINONES[clave]["gmf"]
    M_diente = PINONES[clave]["M_diente"]

    ensayos.clear()
    lbl_turno.config(text=f"Cargando {len(archivos)} archivos...")
    root.update()

    def _procesar():
        for i, arch in enumerate(archivos):
            res = analizar_archivo(arch, gmf, M_diente)
            ensayos.append(res)
            root.after(0, lambda i=i: (
                lbl_turno.config(text=f"Procesando {i+1}/{len(archivos)}..."),
                canvas_pbar.coords(rect_pbar, 0, 0, int(160*(i+1)/len(archivos)), 4)
            ))

        # Calcular umbrales dinámicos (media + 3σ) para los 3 criterios
        def _dyn(vals, lb, ub):
            if len(vals) < 5: return lb
            mu = float(np.mean(vals)); sg = float(np.std(vals))
            return float(max(lb, min(mu + 3.0*sg, ub)))

        vals_kurt = [r["kurt_max"]  for r in ensayos if "error" not in r]
        vals_sb   = [r["sb_max"]    for r in ensayos if "error" not in r]
        vals_ceps = [r["ceps_max"]  for r in ensayos if "error" not in r]

        kurt_umbral[0] = _dyn(vals_kurt, 4.0, 15.0)
        sb_umbral[0]   = _dyn(vals_sb,   3.0, 50.0)
        if len(vals_ceps) >= 5:
            ceps_umbral[0] = float(np.mean(vals_ceps)) * CEPSTRUM_UMBRAL_COEF
        scale_umbral.set(kurt_umbral[0])

        fecha = datetime.datetime.now().strftime("%d/%m/%Y  %H:%M")
        n_vent_medio = np.mean([r.get("fe",{}).get("n_ventanas",1) for r in ensayos if "error" not in r])
        root.after(0, lambda: (
            lbl_turno.config(
                text=f"Turno {fecha}  ·  {len(archivos)} arch  ·  "
                     f"K≥{kurt_umbral[0]:.1f}  SB≥{sb_umbral[0]:.1f}  "
                     f"Cp≥{ceps_umbral[0]:.2f}  ({n_vent_medio:.0f} vent/zona)"),
            refrescar_lista(),
            _seleccionar_primer_sospechoso()
        ))

    threading.Thread(target=_procesar, daemon=True).start()


def _seleccionar_primer_sospechoso():
    """Selecciona automáticamente el primer sospechoso al cargar el turno."""
    for i, res in enumerate(ensayos):
        if "error" not in res and es_sospechoso(res) and res.get("etiqueta") is None:
            idx_actual[0] = i
            mostrar_ensayo(i)
            for j, idx in enumerate(indices_visibles):
                if idx == i:
                    listbox.selection_set(j)
                    listbox.see(j)
                    break
            return
    # Si no hay sospechosos, mostrar el primero
    if ensayos:
        idx_actual[0] = 0
        mostrar_ensayo(0)


# ==========================
# BARRA INFERIOR — CONTROLES
# ==========================
tk.Frame(root, bg=C_BORDER, height=1).pack(fill="x")
frame_bottom = tk.Frame(root, bg=C_SURFACE, height=44)
frame_bottom.pack(fill="x")
frame_bottom.pack_propagate(False)

hacer_boton(frame_bottom, "📁  CARGAR TURNO", cargar_turno,
            bg=C_ACENTO, fg="white", ancho=18, fs=10, bold=True
            ).pack(side="left", padx=12, pady=6)

tk.Frame(frame_bottom, bg=C_BORDER2, width=1).pack(side="left", fill="y", pady=6)
tk.Label(frame_bottom, text="Piñón:", bg=C_SURFACE, fg=C_TEXT_SUB,
         font=(C_MONO, 9)).pack(side="left", padx=(10,4))

nombres_combo = [nombre for _, nombre in OPCIONES]
combo_pinon   = ttk.Combobox(frame_bottom, values=nombres_combo,
                              state="readonly", width=38, font=(C_MONO, 9))
combo_pinon.current(1)
combo_pinon.pack(side="left")
combo_pinon.bind("<<ComboboxSelected>>",
    lambda e: clave_pinon.__setitem__(0, IDX_A_CLAVE.get(combo_pinon.current(), "ARBOL_SEC_14")))

tk.Label(frame_bottom, text="← → navegar   G=bueno   M=malo   I=ignorar",
         bg=C_SURFACE, fg=C_TEXT_DIM, font=(C_MONO, 8)
         ).pack(side="right", padx=16)

root.mainloop()
