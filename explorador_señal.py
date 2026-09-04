"""
explorador_señal.py
===================
Explorador de señales crudas engrane_*.csv

Muestra para cada archivo:
  1. Señal completa con zonas emp/ret marcadas
  2. Curtosis por ventana (0.1s) a lo largo del tiempo
  3. Espectro FFT del retroceso estabilizado (en órdenes)

Objetivo: estudiar la distribución real de los datos ANTES de
definir umbrales o etiquetar BUENO/MALO.
"""

import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import glob
import threading

# ============================================================
# CONFIGURACIÓN
# ============================================================
FS    = 48000
RPM   = 872.0
F_ROT = RPM / 60.0
T_ROT = 1.0 / F_ROT
M_ROT = int(T_ROT * FS)   # muestras por giro

def _actualizar_cinematica(rpm_nuevo):
    """Recalcula F_ROT, T_ROT, M_ROT y GMFs cuando cambia la RPM."""
    global RPM, F_ROT, T_ROT, M_ROT
    RPM   = float(rpm_nuevo)
    F_ROT = RPM / 60.0
    T_ROT = 1.0 / F_ROT
    M_ROT = int(T_ROT * FS)
    # Actualizar GMFs en PINONES
    for key, p in PINONES.items():
        p["gmf"] = p["dientes"] * F_ROT

PINONES = {
    "PIMA":         {"dientes": 26, "gmf": 26 * F_ROT},
    "ARBOL_SEC_14": {"dientes": 14, "gmf": 14 * F_ROT},
    "ARBOL_SEC_15": {"dientes": 15, "gmf": 15 * F_ROT},
}

MARGEN_BUSQUEDA   = 0.15
FACTOR_UMBRAL_SEP = 2.0
VENTANA_ENV_LENTA = 0.015
VENTANA_SEG       = 0.10
VENTANA_MUESTRAS  = int(VENTANA_SEG * FS)
PASO_SEG          = 0.05
PASO_MUESTRAS     = int(PASO_SEG * FS)
TOL_SB            = F_ROT * 0.6
TRAMO_SEG         = 0.5
TRAMO_MUESTRAS    = int(TRAMO_SEG * FS)

# ============================================================
# PALETA
# ============================================================
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
C_MALO     = "#ef4444"
C_DUDOSO   = "#f59e0b"
C_MONO     = "Consolas"

# ============================================================
# ANÁLISIS
# ============================================================
def detectar_separador(señal):
    """
    Detecta el punto de separación empuje→retroceso — máquina DEMM.

    FÍSICA: El freno de separación eleva el RMS por giro 5-20× respecto
    al engrane normal. Los golpes de diente elevan 2-4×. El separador
    siempre es el PRIMER GIRO que supera el umbral de frenado.

    MÉTODO RMS POR GIRO (T_ROT = 52.9ms = 2537 muestras):
    1. RMS de cada giro en zona 15%-85% de la señal.
    2. Nivel ref = mediana del primer tercio (empuje estable).
    3. Primer giro > 4× nivel_ref = separador.
       Fallback progresivo: 3×, 2.5×, máximo absoluto.
    """
    n     = len(señal)
    s     = señal.astype(np.float64)
    M_ROT = int(T_ROT * FS)

    # Envolvente rápida (15ms) — solo para visualización
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

    # Nivel ref = mediana del primer tercio (empuje estable, lejos del sep)
    n_ref     = max(2, n_giros // 3)
    nivel_ref = float(np.median(rms_giros[:n_ref]))
    if nivel_ref < 1e-9:
        nivel_ref = float(np.median(rms_giros)) + 1e-9

    # Primer giro sobre umbral decreciente = separador
    for factor in [4.0, 3.0, 2.5]:
        for i, rms in enumerate(rms_giros):
            if rms > nivel_ref * factor:
                return i0 + i * M_ROT, env

    return i0 + int(np.argmax(rms_giros)) * M_ROT, env


def detectar_freno(zona_ret):
    """
    Detecta el final útil del retroceso — máquina DEMM.
    Guardia temporal de 350ms: el freno nunca ocurre en los primeros 7 giros.
    Nivel ref = giros post-guardia (engrane estable).
    Buscar primer giro > 2.5× nivel_ref después de la guardia.
    Fallback 92%: operador extrajo pieza antes del freno.
    """
    n     = len(zona_ret)
    if n < int(FS * 0.15):
        return n

    s     = zona_ret.astype(np.float64)
    M_ROT = int(T_ROT * FS)
    n_giros = n // M_ROT
    if n_giros < 6:
        return int(n * 0.92)

    rms_giros = np.array([
        float(np.sqrt(np.mean(s[i*M_ROT:(i+1)*M_ROT]**2)))
        for i in range(n_giros)
    ])

    MIN_GIROS_GUARDIA = max(7, int(0.35 * FS / M_ROT))
    i_est_0 = MIN_GIROS_GUARDIA
    i_est_1 = min(i_est_0 + 5, n_giros - 1)
    if i_est_1 <= i_est_0 or i_est_0 >= n_giros:
        return int(n * 0.92)

    nivel_ref = float(np.median(rms_giros[i_est_0:i_est_1]))
    if nivel_ref < 1e-9:
        return int(n * 0.92)

    for i in range(MIN_GIROS_GUARDIA, n_giros):
        if rms_giros[i] > nivel_ref * 2.5:
            return max(0, i * M_ROT - M_ROT // 2)

    return int(n * 0.92)

def detectar_estabilizacion(zona_ret):
    """
    Detecta donde termina la transicion al inicio del retroceso — DEMM.

    Tras el separador vienen peaks altos de arranque (diente-diente).
    En AS (ARB_14/ARB_15) estos peaks duran tipicamente 4-5 giros (~265ms).

    ESTRATEGIA:
    - Saltar siempre GIROS_MINIMOS=5 giros desde el inicio del retroceso
      (cubre peaks de arranque del AS en todos los casos observados)
    - Desde ahi buscar 2 giros consecutivos estables
    - Nivel ref = percentil 20 de la segunda mitad (robusto a defectos)
    - Fallback: retornar despues de GIROS_MINIMOS si no converge
    """
    s     = zona_ret.astype(np.float64)
    n     = len(s)
    M_ROT = int(T_ROT * FS)
    n_giros = n // M_ROT
    if n_giros < 3:
        return 0

    rms_giros = np.array([
        float(np.sqrt(np.mean(s[i*M_ROT:(i+1)*M_ROT]**2)))
        for i in range(n_giros)
    ])

    # Nivel ref = percentil 20 de la segunda mitad
    segunda_mitad = rms_giros[n_giros // 2:]
    nivel_ref = float(np.percentile(segunda_mitad, 20))
    if nivel_ref < 1e-9:
        nivel_ref = float(np.median(rms_giros))
    if nivel_ref < 1e-9:
        return 0

    umbral_min    = nivel_ref * 0.35
    umbral_max    = nivel_ref * 1.80
    CONFIRM       = 2
    GIROS_MINIMOS = 5   # saltar siempre los primeros 5 giros (~265ms)

    inicio = min(GIROS_MINIMOS, n_giros - CONFIRM - 1)

    for i in range(inicio, n_giros - CONFIRM + 1):
        ventana = rms_giros[i:i+CONFIRM]
        if not all(umbral_min <= r <= umbral_max for v in ventana
                   for r in [v]):
            continue
        if np.max(ventana) / (np.min(ventana) + 1e-12) > 1.8:
            continue
        return max(0, i * M_ROT)

    # Fallback: retornar despues de GIROS_MINIMOS
    return min(GIROS_MINIMOS * M_ROT, n // 3)

def calcular_kurt_ventanas(zona):
    """
    Curtosis y RMS por giro completo (T_ROT ~ 52.9ms).
    Criterio empresa: umbral = mean(K) + 3*sigma(K).
    Un giro que supere ese umbral indica golpe en diente.
    Retorna tambien umbral_3sigma para trazar en el grafico.
    """
    s     = zona.astype(np.float64)
    n     = len(s)
    M_ROT = int(T_ROT * FS)
    kurt_v = []; rms_v = []; t_v = []
    for inicio in range(0, n - M_ROT + 1, M_ROT):
        seg = s[inicio:inicio + M_ROT]
        mu  = np.mean(seg)
        den = np.mean((seg - mu)**2)**2
        k   = float(np.mean((seg - mu)**4) / (den + 1e-12))
        rms = float(np.sqrt(np.mean(seg**2)))
        kurt_v.append(k)
        rms_v.append(rms)
        t_v.append(inicio / FS)
    t_v   = np.array(t_v)
    kurt_v = np.array(kurt_v)
    rms_v  = np.array(rms_v)
    return t_v, kurt_v, rms_v

def calcular_espectro(zona, gmf):
    """FFT en órdenes del tramo central del retroceso."""
    s = zona.astype(np.float64)
    n = len(s)
    ventana_h = np.hanning(n)
    fft_mag   = np.abs(np.fft.rfft(s * ventana_h)) * 2 / n
    freqs     = np.fft.rfftfreq(n, d=1.0/FS)
    fft_db    = 20 * np.log10(fft_mag + 1e-12)
    ordenes   = freqs / F_ROT

    wl = min(51, len(fft_db) - 1)
    if wl % 2 == 0: wl -= 1
    if wl < 3: wl = 3
    po = min(5, wl - 1)
    fft_suav = savgol_filter(fft_db, window_length=wl, polyorder=po)
    return ordenes, fft_db, fft_suav, gmf / F_ROT

def calcular_periodicidad(zona):
    """
    Analiza si los impulsos en la zona son periódicos a T_ROT.

    Dos técnicas:
    1. Autocorrelación de la envolvente → picos en τ = k × T_ROT
    2. Promedio síncrono → si hay diente dañado aparece pico consistente;
       si es ruido aleatorio se cancela al promediar

    ratio_syn > 3  → golpe periódico real muy probable
    ratio_syn 1-3  → señal con algo de estructura
    ratio_syn < 1  → ruido sin periodicidad
    """
    s     = zona.astype(np.float64)
    n     = len(s)
    M_ROT = int(T_ROT * FS)

    # ── Autocorrelación de la envolvente ──
    v_env = max(1, int(FS * 0.002))
    env   = np.sqrt(np.convolve(s**2, np.ones(v_env)/v_env, mode='same'))
    env   = env - np.mean(env)
    max_lag  = min(int(4.5 * M_ROT), n // 2)
    autocorr = np.correlate(env, env, mode='full')
    mid      = len(autocorr) // 2
    autocorr = autocorr[mid : mid + max_lag]
    if autocorr[0] > 0:
        autocorr = autocorr / autocorr[0]
    tau = np.arange(len(autocorr)) / FS

    # Detectar picos cerca de múltiplos de T_ROT
    tol_rot     = int(M_ROT * 0.08)
    t_rot_peaks = []
    for k in range(1, 5):
        centro = int(k * M_ROT)
        if centro >= len(autocorr): break
        inicio = max(0, centro - tol_rot)
        fin    = min(len(autocorr), centro + tol_rot)
        sub    = autocorr[inicio:fin]
        if len(sub) == 0: continue
        idx_pk = int(np.argmax(sub)) + inicio
        val_pk = float(autocorr[idx_pk])
        if val_pk > 0.08:
            t_rot_peaks.append((k, tau[idx_pk], val_pk))

    # ── Promedio síncrono ──
    n_ciclos     = n // M_ROT
    promedio_syn = np.zeros(M_ROT)
    if n_ciclos >= 2:
        for i in range(n_ciclos):
            promedio_syn += s[i*M_ROT : (i+1)*M_ROT]
        promedio_syn /= n_ciclos

    t_syn     = np.arange(M_ROT) / FS * 1000   # ms
    pico_syn  = float(np.max(np.abs(promedio_syn))) if n_ciclos >= 2 else 0.0
    ruido_syn = float(np.sqrt(np.mean(promedio_syn**2))) if n_ciclos >= 2 else 1e-9
    ratio_syn = pico_syn / (ruido_syn + 1e-12)

    return {
        "autocorr_tau":  tau,
        "autocorr_val":  autocorr,
        "t_rot_peaks":   t_rot_peaks,
        "promedio_syn":  promedio_syn,
        "t_syn":         t_syn,
        "pico_syn":      pico_syn,
        "ratio_syn":     ratio_syn,
        "n_ciclos":      n_ciclos,
        "M_ROT":         M_ROT,
    }



# ==========================
# VALIDACIÓN DE GRABACIÓN
# Constantes (sincronizadas con revisor_turno_v2)
# ==========================
VAL_RMS_EMP_RATIO  = 0.20    # ratio mínimo RMS_emp/RMS_ret
VAL_RMS_EMP_CV_MIN = 0.08    # CV mínimo del RMS del empuje
VAL_KURT_GOLPE     = 12.0    # Kurt_max → golpe externo si sin periodicidad
VAL_CLIP_PCT       = 0.5     # % muestras saturadas → clipping
VAL_DUR_RET_MIN    = 0.30    # duración mínima retroceso estabilizado (s)
VAL_DUR_TOTAL_MIN  = 1.80    # duración total mínima del archivo (s)
VAL_RMS_RET_ABS    = 3e-4    # RMS retroceso mínimo absoluto
TRAMO_SEG          = 0.5
TRAMO_MUESTRAS     = int(TRAMO_SEG * FS)
TOL_T              = 0.10
C_VALIDO           = "#22c55e"
C_DUDOSO_VAL       = "#f59e0b"
C_INVALIDO_VAL     = "#ef4444"
C_INVALIDO         = C_INVALIDO_VAL   # alias usado por validar_grabacion


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

    # Crest Factor (CF): pico / RMS
    # Impulsivo por defecto → CF alto. Máquina ruidosa uniforme → CF bajo (~2-3)
    peak     = float(np.max(np.abs(sig)))
    cf       = peak / (rms + 1e-12)

    # Shape Factor (SF): RMS / media(|x|)
    # Señal gaussiana pura ≈ 1.25. Sube con impulsividad, más estable que K.
    mean_abs = float(np.mean(np.abs(sig)))
    sf       = rms / (mean_abs + 1e-12)

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
        "cf": cf, "sf": sf,
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
        "rms", "kurt", "cf", "sf",
        "fondo_espectro", "nivel_gmf",
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
KURT_UMBRAL_INICIAL  = 5.0
SB_UMBRAL_INICIAL    = 8.0
CEPSTRUM_UMBRAL_COEF = 1.4
MIN_VENTANAS         = 3
VAL_SCORE_DUDOSO     = 40
VAL_SCORE_VALIDO     = 70

kurt_umbral  = [KURT_UMBRAL_INICIAL]
clave_pinon  = ["ARBOL_SEC_14"]


# Umbrales dinámicos — se actualizan al cargar el turno
sb_umbral    = [SB_UMBRAL_INICIAL]
ceps_umbral  = [999.0]   # se calcula como media_turno × CEPSTRUM_UMBRAL_COEF

def analizar_archivo(ruta, gmf):
    try:
        df    = pd.read_csv(ruta)
        señal = df["senal"].values.astype(np.float32)
        n     = len(señal)

        idx_sep, env_lenta = detectar_separador(señal)
        margen   = int(FS * 0.02)
        i_emp_i  = int(n * 0.05)
        i_emp_f  = max(0, idx_sep - margen)
        i_ret_i  = min(n, idx_sep + margen)
        zona_ret_prel = señal[i_ret_i:int(n*0.95)]
        idx_freno     = detectar_freno(zona_ret_prel)
        i_ret_f       = i_ret_i + idx_freno
        zona_ret_bruta= señal[i_ret_i:i_ret_f]
        idx_estab     = detectar_estabilizacion(zona_ret_bruta)
        i_ret_estab   = i_ret_i + idx_estab
        zona_ret      = zona_ret_bruta[idx_estab:]

        zona_emp = señal[i_emp_i:i_emp_f]

        # Tramo central (igual que revisor_turno): 0.5s del centro de cada zona
        # Se usa para espectro y periodicidad — consistente con el análisis del revisor
        def _tramo_central(zona):
            nz = len(zona)
            if nz <= TRAMO_MUESTRAS: return zona.copy()
            centro = nz // 2
            mitad  = TRAMO_MUESTRAS // 2
            return zona[centro - mitad : centro + mitad]

        tramo_emp = _tramo_central(zona_emp)
        tramo_ret = _tramo_central(zona_ret)

        # Curtosis por giro — empuje sin los últimos 2 giros antes del separador
        # (esos giros son la transición de salida del diente, no engrane estable)
        M_ROT_emp = int(T_ROT * FS)
        recorte_emp = 2 * M_ROT_emp
        zona_emp_util = zona_emp[:-recorte_emp] if len(zona_emp) > recorte_emp * 2 else zona_emp
        t_emp, k_emp, r_emp = calcular_kurt_ventanas(zona_emp_util)
        t_emp += i_emp_i / FS

        t_ret, k_ret, r_ret = calcular_kurt_ventanas(zona_ret)
        t_ret += i_ret_estab / FS

        # Espectro sobre tramo central (igual que revisor)
        if len(tramo_ret) >= VENTANA_MUESTRAS:
            ordenes, fft_db, fft_suav, orden_gmf = calcular_espectro(tramo_ret, gmf)
        else:
            ordenes = fft_db = fft_suav = np.array([0])
            orden_gmf = gmf / F_ROT

        # Espectro del empuje
        if len(tramo_emp) >= VENTANA_MUESTRAS:
            ordenes_emp, fft_db_emp, fft_suav_emp, _ = calcular_espectro(tramo_emp, gmf)
        else:
            ordenes_emp = fft_db_emp = fft_suav_emp = np.array([0])

        # Periodicidad sobre tramo central (igual que revisor)
        per_emp = calcular_periodicidad(tramo_emp) if len(tramo_emp) >= int(3*T_ROT*FS) else None
        per_ret = calcular_periodicidad(tramo_ret) if len(tramo_ret) >= int(3*T_ROT*FS) else None

        # Tiempos del tramo central para marcar en la señal
        nze = len(zona_emp); nzr = len(zona_ret)
        t_tramo_emp_i = (i_emp_i  + max(0, nze//2 - TRAMO_MUESTRAS//2)) / FS
        t_tramo_ret_i = (i_ret_estab + max(0, nzr//2 - TRAMO_MUESTRAS//2)) / FS
        t_tramo_emp_f = t_tramo_emp_i + len(tramo_emp)/FS
        t_tramo_ret_f = t_tramo_ret_i + len(tramo_ret)/FS

        # Stats resumen
        rms_ret  = float(np.sqrt(np.mean(zona_ret.astype(np.float64)**2))) if len(zona_ret) > 0 else 0
        rms_emp  = float(np.sqrt(np.mean(zona_emp.astype(np.float64)**2))) if len(zona_emp) > 0 else 0
        kurt_ret_max  = float(np.max(k_ret))  if len(k_ret)  > 0 else 0
        kurt_ret_mean = float(np.mean(k_ret)) if len(k_ret)  > 0 else 0
        cv_rms_ret    = float((np.max(r_ret) - np.mean(r_ret)) / (np.mean(r_ret) + 1e-12)) if len(r_ret) > 0 else 0

        def _kurt_global(z):
            z = z.astype(np.float64); mu = np.mean(z)
            return float(np.mean((z-mu)**4) / (np.mean((z-mu)**2)**2 + 1e-12))

        # Excluir último giro del retroceso — puede contener transición al freno
        # con amplitud creciente que distorsiona K y CF
        recorte_ret = int(T_ROT * FS)
        zona_ret_limpia = zona_ret[:-recorte_ret] if len(zona_ret) > recorte_ret * 3 else zona_ret

        # K global método Cycla — curtosis de toda la zona (un único valor diagnóstico)
        kurt_global_ret = _kurt_global(zona_ret_limpia) if len(zona_ret_limpia) > 10 else 3.0
        kurt_global_emp = _kurt_global(zona_emp_util)   if len(zona_emp_util) > 10 else 3.0

        # Crest Factor clásico: peak absoluto / RMS
        def _cf_global(z):
            z = z.astype(np.float64)
            rms = float(np.sqrt(np.mean(z**2)))
            return float(np.max(np.abs(z))) / (rms + 1e-12)

        # CF robusto: percentil 99.5 / RMS — ignora el 0.5% de peaks extremos aislados
        def _cf_p99(z):
            z = z.astype(np.float64)
            rms = float(np.sqrt(np.mean(z**2)))
            return float(np.percentile(np.abs(z), 99.5)) / (rms + 1e-12)

        # CF de envolvente: peak_env / RMS_env
        # La envolvente resalta modulaciones periódicas (golpe de diente)
        # ignorando la portadora de alta frecuencia
        def _cf_env(z):
            from scipy.signal import hilbert
            z = z.astype(np.float64)
            env = np.abs(hilbert(z))
            rms_env  = float(np.sqrt(np.mean(env**2)))
            peak_env = float(np.max(env))
            return peak_env / (rms_env + 1e-12)

        # Shape Factor global por zona
        def _sf_global(z):
            z = z.astype(np.float64)
            rms = float(np.sqrt(np.mean(z**2)))
            return rms / (float(np.mean(np.abs(z))) + 1e-12)

        cf_ret      = _cf_global(zona_ret_limpia) if len(zona_ret_limpia) > 10 else 0.0
        cf_emp      = _cf_global(zona_emp_util)   if len(zona_emp_util)   > 10 else 0.0
        cf_p99_ret  = _cf_p99(zona_ret_limpia)    if len(zona_ret_limpia) > 10 else 0.0
        cf_p99_emp  = _cf_p99(zona_emp_util)      if len(zona_emp_util)   > 10 else 0.0
        cf_env_ret  = _cf_env(zona_ret_limpia)    if len(zona_ret_limpia) > 10 else 0.0
        cf_env_emp  = _cf_env(zona_emp_util)      if len(zona_emp_util)   > 10 else 0.0
        sf_ret      = _sf_global(zona_ret_limpia) if len(zona_ret_limpia) > 10 else 1.25
        sf_emp      = _sf_global(zona_emp_util)   if len(zona_emp_util)   > 10 else 1.25

        # Ratio autocorrelación (ya calculado en per_ret/per_emp)
        autocorr_ratio_ret = per_ret["ratio_syn"] if per_ret else 0.0
        autocorr_ratio_emp = per_emp["ratio_syn"] if per_emp else 0.0

        # ÍNDICE COMPUESTO DE DEFECTO (IDX)
        # Combina K + CF + SF + autocorr para separar defecto vs máquina ruidosa
        #
        # Defecto real:     K alto + CF alto + SF elevado + autocorr > 1.5
        # Máquina ruidosa:  K moderado + CF bajo + SF ≈ 1.25 + autocorr ≈ 0
        #
        # Normalización orientativa (valores típicos PIMA OK):
        #   K_norm: K/3.6 (umbral del especialista)
        #   CF_norm: CF/3.0 (CF gaussiano ≈ 2.5-3.0)
        #   SF_norm: SF/1.25 (gaussiano puro ≈ 1.25)
        #   AC_norm: autocorr_ratio / 1.5 (umbral periodicidad débil)
        #
        # IDX > 1.5 → probable defecto real
        # IDX 1.0-1.5 → sospechoso
        # IDX < 1.0 → máquina ruidosa o sin defecto
        def _idx_defecto(k, cf_v, sf_v, ac):
            k_n  = k    / 3.6
            cf_n = cf_v / 3.0
            sf_n = sf_v / 1.25
            ac_n = ac   / 1.5
            # Promedio ponderado: K y CF tienen mayor peso
            return (0.35*k_n + 0.30*cf_n + 0.15*sf_n + 0.20*ac_n)

        idx_ret = _idx_defecto(kurt_global_ret, cf_ret, sf_ret, autocorr_ratio_ret)
        idx_emp = _idx_defecto(kurt_global_emp, cf_emp, sf_emp, autocorr_ratio_emp)

        # K por 10 ventanas iguales — solo para localización visual del golpe
        def _kurt_10_ventanas(z):
            n_v = 10; tam = len(z) // n_v
            if tam < 10: return np.full(n_v, 3.0), np.zeros(n_v)
            kv = np.array([_kurt_global(z[i*tam:(i+1)*tam]) for i in range(n_v)])
            tv = np.array([(i + 0.5) * tam / FS for i in range(n_v)])
            return kv, tv

        k10_ret, t10_ret = _kurt_10_ventanas(zona_ret_limpia)
        k10_emp, t10_emp = _kurt_10_ventanas(zona_emp_util)
        # Offsets temporales correctos
        t10_ret += i_ret_estab / FS
        t10_emp += i_emp_i / FS

        # ── Validación de grabación (misma lógica que revisor_turno_v2) ──
        tramo_emp_val = _tramo_central(zona_emp)
        tramo_ret_val = _tramo_central(zona_ret)
        _, _, te_limp_v, inf_e_v    = clasificar_impulsos(tramo_emp_val, gmf)
        _, _, tr_limp_v, inf_r_v    = clasificar_impulsos(tramo_ret_val, gmf)
        _, _, _,          inf_e_full = clasificar_impulsos(zona_emp, gmf)
        fe_v = calcular_features(tramo_emp_val, te_limp_v, gmf)
        fr_v = calcular_features(tramo_ret_val, tr_limp_v, gmf)
        kurt_max_v = max(fe_v["kurt_limp_worst"], fr_v["kurt_limp_worst"])

        res_validacion_input = {
            "fe": fe_v, "fr": fr_v,
            "inf_e": inf_e_v, "inf_r": inf_r_v, "inf_e_full": inf_e_full,
            "señal": señal,
            "kurt_max": kurt_max_v,
            "t_sep": idx_sep / FS,
            "i_emp_i":     i_emp_i,
            "i_emp_f":     i_emp_f,
            "i_ret_inicio": i_ret_i,
            "i_ret_estab":  i_ret_estab,
            "i_ret_fin":    i_ret_f,
            "n_dientes":   round(gmf / F_ROT),
        }
        validacion = validar_grabacion(res_validacion_input)

        return {
            "ok":          True,
            "ruta":        ruta,
            "nombre":      os.path.basename(ruta),
            "validacion":  validacion,
            "señal":       señal,
            "env_lenta":   env_lenta,
            "i_emp_i":     i_emp_i,  "i_emp_f":    i_emp_f,
            "i_ret_i":     i_ret_i,  "i_ret_estab":i_ret_estab,
            "i_ret_f":     i_ret_f,
            "t_sep":       idx_sep / FS,
            "t_freno":     i_ret_f / FS,
            "t_estab":     i_ret_estab / FS,
            # Curtosis ventanas por giro
            "t_emp": t_emp, "k_emp": k_emp, "r_emp": r_emp,
            "t_ret": t_ret, "k_ret": k_ret, "r_ret": r_ret,
            # Zonas crudas para análisis posicional de ciclicidad
            "zona_emp_raw": zona_emp_util.astype(np.float64),
            "zona_ret_raw": zona_ret.astype(np.float64),
            # K global método Cycla (criterio diagnóstico principal)
            "kurt_global_ret": kurt_global_ret,
            "kurt_global_emp": kurt_global_emp,
            "cf_ret":     cf_ret,      "cf_emp":     cf_emp,
            "cf_p99_ret": cf_p99_ret,  "cf_p99_emp": cf_p99_emp,
            "cf_env_ret": cf_env_ret,  "cf_env_emp": cf_env_emp,
            "sf_ret":  sf_ret,  "sf_emp":  sf_emp,
            "idx_ret": idx_ret, "idx_emp": idx_emp,
            "autocorr_ratio_ret": autocorr_ratio_ret,
            "autocorr_ratio_emp": autocorr_ratio_emp,
            # K por 10 ventanas (localización visual del golpe)
            "k10_ret": k10_ret, "t10_ret": t10_ret,
            "k10_emp": k10_emp, "t10_emp": t10_emp,
            # Espectro
            "ordenes": ordenes, "fft_db": fft_db,
            "fft_suav": fft_suav, "orden_gmf": orden_gmf,
            # Espectro empuje
            "ordenes_emp": ordenes_emp, "fft_db_emp": fft_db_emp,
            "fft_suav_emp": fft_suav_emp,
            # Stats
            "rms_ret": rms_ret, "rms_emp": rms_emp,
            "kurt_ret_max": kurt_ret_max,
            "kurt_ret_mean": kurt_ret_mean,
            "cv_rms_ret": cv_rms_ret,
            "dur_ret": (i_ret_f - i_ret_estab) / FS,
            "gmf": gmf,
            # Tramo central (zona de análisis real, igual que revisor)
            "t_tramo_emp_i": t_tramo_emp_i, "t_tramo_emp_f": t_tramo_emp_f,
            "t_tramo_ret_i": t_tramo_ret_i, "t_tramo_ret_f": t_tramo_ret_f,
            "tramo_seg":     TRAMO_SEG,
            # Periodicidad
            "per_emp": per_emp,
            "per_ret": per_ret,
        }
    except Exception as ex:
        return {"ok": False, "nombre": os.path.basename(ruta), "error": str(ex)}


# ============================================================
# ESTADO GLOBAL
# ============================================================
archivos_lista  = []    # lista de rutas
resultados_cache = {}   # ruta → dict analizado
idx_actual      = [0]
clave_pinon     = ["ARBOL_SEC_14"]

# Umbrales estadísticos por orden — cargados desde .pkl externo
# Estructura del pkl: {"ordenes", "ret": {"umbral", "media", "sigma"}, "emp": {...}}
umbral_pkl      = {}    # dict completo cargado desde archivo
umbral_dataset_ret = {}  # apunta a umbral_pkl["ret"] cuando está cargado
umbral_dataset_emp = {}  # apunta a umbral_pkl["emp"] cuando está cargado


def cargar_umbral_pkl(ruta_pkl):
    """
    Carga un archivo .pkl de umbrales generado por calcular_umbrales_espectro.py
    y actualiza las variables globales umbral_dataset_ret / umbral_dataset_emp.
    Estructura del pkl: {"ordenes", "ret": {"umbral","media","sigma","n_archivos"}, "emp": {...}}
    Retorna (ok:bool, mensaje:str).
    """
    global umbral_pkl, umbral_dataset_ret, umbral_dataset_emp
    import pickle
    try:
        with open(ruta_pkl, "rb") as f:
            datos = pickle.load(f)
        grilla = datos.get("ordenes", np.array([]))
        ret    = datos.get("ret")
        emp    = datos.get("emp")
        if len(grilla) < 2 or ret is None or ret.get("umbral") is None:
            return False, "El .pkl no tiene la estructura esperada (falta ret/umbral)."
        # Guardar el pkl completo — el dibujo accede como ud["ordenes"] y ud["ret"]["umbral"]
        umbral_dataset_ret = datos   # mismo objeto para retroceso
        umbral_dataset_emp = datos   # mismo objeto para empuje
        umbral_pkl = datos
        pinon  = datos.get("pinon", "?")
        n_arch = datos.get("n_total", ret.get("n_archivos", "?"))
        v_suav = datos.get("ventana_suav", "—")
        return True, (f"Umbrales cargados  ·  {pinon}  ·  "
                      f"{n_arch} archivos  ·  suavizado={v_suav} pts")
    except Exception as e:
        return False, f"Error al cargar .pkl: {e}"


# ============================================================
# UI
# ============================================================
root = tk.Tk()
root.title("Explorador de Señales — DEMM")
root.geometry("1400x860")
root.minsize(1000, 600)
root.configure(bg=C_BG)

# ── Header ──
frame_hdr = tk.Frame(root, bg=C_SURFACE, height=50)
frame_hdr.pack(fill="x")
frame_hdr.pack_propagate(False)
tk.Frame(frame_hdr, bg=C_ACENTO, width=3).pack(side="left", fill="y")
tk.Label(frame_hdr, text="HORSE", bg=C_SURFACE, fg=C_ACENTO,
         font=(C_MONO, 13, "bold")).pack(side="left", padx=14)
tk.Frame(frame_hdr, bg=C_BORDER, width=1).pack(side="left", fill="y", pady=8)
fh = tk.Frame(frame_hdr, bg=C_SURFACE); fh.pack(side="left", padx=12)
tk.Label(fh, text="EXPLORADOR DE SEÑALES  —  DEMM", bg=C_SURFACE, fg=C_TEXT,
         font=(C_MONO, 11, "bold")).pack(anchor="w")
lbl_info_hdr = tk.Label(fh, text="Sin archivos cargados", bg=C_SURFACE,
                         fg=C_TEXT_SUB, font=(C_MONO, 9)); lbl_info_hdr.pack(anchor="w")

# Contador y navegación en header
fh2 = tk.Frame(frame_hdr, bg=C_SURFACE); fh2.pack(side="right", padx=18)
lbl_idx = tk.Label(fh2, text="0 / 0", bg=C_SURFACE, fg=C_TEXT_SUB,
                   font=(C_MONO, 10, "bold")); lbl_idx.pack(anchor="e")

tk.Frame(root, bg=C_BORDER, height=1).pack(fill="x")

# ── Main layout: lista izquierda + gráficos derecha ──
frame_main = tk.Frame(root, bg=C_BG)
frame_main.pack(fill="both", expand=True)

# Panel lista izquierda
frame_lista = tk.Frame(frame_main, bg=C_SURFACE, width=240)
frame_lista.pack(side="left", fill="y")
frame_lista.pack_propagate(False)
tk.Frame(frame_lista, bg=C_BG, width=1).pack(side="right", fill="y")

tk.Label(frame_lista, text="ARCHIVOS", bg=C_SURFACE, fg=C_TEXT_SUB,
         font=(C_MONO, 8, "bold"), padx=12, pady=8).pack(anchor="w")

# Filtro rápido
frame_fil = tk.Frame(frame_lista, bg=C_SURFACE, padx=8)
frame_fil.pack(fill="x")
lbl_buscar = tk.Label(frame_fil, text="Buscar:", bg=C_SURFACE, fg=C_TEXT_DIM,
                       font=(C_MONO, 8)); lbl_buscar.pack(side="left")
var_buscar = tk.StringVar()
entry_buscar = tk.Entry(frame_fil, textvariable=var_buscar, bg=C_SURFACE2,
                         fg=C_TEXT, insertbackground=C_TEXT,
                         font=(C_MONO, 8), relief="flat", bd=4, width=16)
entry_buscar.pack(side="left", padx=4, fill="x", expand=True)
var_buscar.trace_add("write", lambda *a: refrescar_lista())

tk.Frame(frame_lista, bg=C_BORDER, height=1).pack(fill="x", pady=2)

# Filtro de validación
frame_val_filtro = tk.Frame(frame_lista, bg=C_SURFACE, padx=8)
frame_val_filtro.pack(fill="x", pady=2)
var_solo_validos = tk.BooleanVar(value=False)
tk.Checkbutton(frame_val_filtro, text="Solo válidos",
               variable=var_solo_validos,
               command=lambda: refrescar_lista(),
               bg=C_SURFACE, fg=C_TEXT_SUB, selectcolor=C_SURFACE2,
               activebackground=C_SURFACE, activeforeground=C_TEXT,
               font=(C_MONO, 8)).pack(side="left")
tk.Frame(frame_lista, bg=C_BORDER, height=1).pack(fill="x")

# Listbox
frame_scroll_lista = tk.Frame(frame_lista, bg=C_SURFACE)
frame_scroll_lista.pack(fill="both", expand=True)
scr_lista = tk.Scrollbar(frame_scroll_lista, bg=C_SURFACE2,
                          troughcolor=C_SURFACE, width=5, relief="flat")
scr_lista.pack(side="right", fill="y")
listbox = tk.Listbox(frame_scroll_lista, bg=C_SURFACE, fg=C_TEXT_SUB,
                     font=(C_MONO, 8), relief="flat", bd=0,
                     selectbackground=C_ACENTO, selectforeground="white",
                     activestyle="none", highlightthickness=0,
                     yscrollcommand=scr_lista.set)
listbox.pack(fill="both", expand=True)
scr_lista.config(command=listbox.yview)

# Estadístico rápido al pie de la lista
tk.Frame(frame_lista, bg=C_BORDER, height=1).pack(fill="x")
frame_stats_mini = tk.Frame(frame_lista, bg=C_SURFACE, padx=10, pady=6)
frame_stats_mini.pack(fill="x")
lbl_stats_mini = tk.Label(frame_stats_mini, text="", bg=C_SURFACE,
                            fg=C_TEXT_DIM, font=(C_MONO, 7), justify="left",
                            wraplength=210, anchor="w")
lbl_stats_mini.pack(fill="x")

# Panel gráficos
frame_graf = tk.Frame(frame_main, bg=C_BG)
frame_graf.pack(side="left", fill="both", expand=True)

# Sub-header gráficos
frame_graf_hdr = tk.Frame(frame_graf, bg=C_SURFACE, height=36)
frame_graf_hdr.pack(fill="x")
frame_graf_hdr.pack_propagate(False)
lbl_nombre_archivo = tk.Label(frame_graf_hdr, text="Sin archivo seleccionado",
                               bg=C_SURFACE, fg=C_TEXT, font=(C_MONO, 9, "bold"))
lbl_nombre_archivo.pack(side="left", padx=12)
lbl_stats_archivo  = tk.Label(frame_graf_hdr, text="", bg=C_SURFACE,
                               fg=C_TEXT_SUB, font=(C_MONO, 8))
lbl_stats_archivo.pack(side="left", padx=4)

# Navegación
frame_nav = tk.Frame(frame_graf_hdr, bg=C_SURFACE)
frame_nav.pack(side="right", padx=12)
tk.Button(frame_nav, text="← Ant.", command=lambda: navegar(-1),
          bg=C_SURFACE2, fg=C_TEXT_SUB, font=(C_MONO, 8),
          relief="flat", bd=0, padx=8, cursor="hand2").pack(side="left", padx=2)
tk.Button(frame_nav, text="Sig. →", command=lambda: navegar(+1),
          bg=C_SURFACE2, fg=C_TEXT_SUB, font=(C_MONO, 8),
          relief="flat", bd=0, padx=8, cursor="hand2").pack(side="left", padx=2)

tk.Frame(frame_graf, bg=C_BORDER, height=1).pack(fill="x")

# ── Figura 4 paneles: 2 filas × 2 columnas ──
fig = plt.figure(figsize=(12, 7.5), facecolor=C_BG)
gs  = gridspec.GridSpec(2, 2, figure=fig,
                         height_ratios=[1.4, 1.2],
                         width_ratios=[1.6, 1.0],
                         hspace=0.42, wspace=0.28)
ax_señal = fig.add_subplot(gs[0, :])   # fila 0, ocupa ambas columnas
ax_kurt  = fig.add_subplot(gs[1, 0])   # fila 1, col izquierda
ax_per   = fig.add_subplot(gs[1, 1])   # fila 1, col derecha (periodicidad)
ax_fft   = None   # se crea dinámicamente bajo ax_kurt cuando se requiere

# Redefinir layout: fila 0 = señal (ancho completo)
#                   fila 1 izq = kurt por ventana
#                   fila 1 der = promedio síncrono + autocorrelación
gs2 = gridspec.GridSpec(3, 2, figure=fig,
                          height_ratios=[1.6, 1.1, 1.1],
                          width_ratios=[1.6, 1.0],
                          hspace=0.48, wspace=0.28)
fig.clf()
ax_señal   = fig.add_subplot(gs2[0, :])
ax_fft_emp = fig.add_subplot(gs2[1, 0])  # FFT empuje — izquierda medio
ax_fft     = fig.add_subplot(gs2[2, 0])  # FFT retroceso — izquierda abajo
ax_syn     = fig.add_subplot(gs2[1, 1])  # promedio síncrono — derecha medio
ax_kurt    = fig.add_subplot(gs2[2, 1])  # kurtosis — derecha abajo

for ax in (ax_señal, ax_kurt, ax_fft, ax_syn, ax_fft_emp):
    ax.set_facecolor(C_SURFACE)
    ax.tick_params(colors=C_TEXT_SUB, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(C_BORDER2)
    ax.grid(True, alpha=0.13, color=C_BORDER2)

fig.subplots_adjust(left=0.06, right=0.97, top=0.95, bottom=0.07, hspace=0.45, wspace=0.28)
canvas_fig = FigureCanvasTkAgg(fig, master=frame_graf)
canvas_fig.get_tk_widget().pack(fill="both", expand=True)

# ── Panel diagnóstico IDX (entre canvas y barra inferior) ──
tk.Frame(root, bg=C_BORDER, height=1).pack(fill="x")
# ── Panel de diagnóstico NVH — RETROCESO y EMPUJE ──────────────────────
# Panel expandido con todos los parámetros en filas separadas y el IDX
# grande y prominente para fácil lectura
frame_diag = tk.Frame(root, bg=C_BG)
frame_diag.pack(fill="x")

def _hacer_recuadro_diag(parent, titulo, lado):
    """Recuadro de diagnóstico — K, CF clásico, CF p99.5, CF envolvente."""
    f = tk.Frame(parent, bg=C_SURFACE2, relief="flat", bd=0, padx=14, pady=6,
                 highlightbackground=C_BORDER2, highlightthickness=2)
    f.pack(side=lado, padx=(10,4) if lado=="left" else (4,10), pady=4)

    tk.Label(f, text=titulo, bg=C_SURFACE2, fg=C_TEXT_SUB,
             font=(C_MONO, 7, "bold")).grid(row=0, column=0, columnspan=6,
                                            sticky="w", pady=(0,3))

    def _fila(row, etiq, ancho_val=5, ancho_est=7):
        tk.Label(f, text=etiq, bg=C_SURFACE2, fg=C_TEXT_SUB,
                 font=(C_MONO, 8)).grid(row=row, column=0, sticky="e", padx=(0,3))
        lv = tk.Label(f, text="—", bg=C_SURFACE2, fg=C_TEXT,
                      font=(C_MONO, 10, "bold"), width=ancho_val, anchor="w")
        lv.grid(row=row, column=1, sticky="w")
        le = tk.Label(f, text="", bg=C_SURFACE2, fg=C_TEXT_SUB,
                      font=(C_MONO, 8), width=ancho_est, anchor="w")
        le.grid(row=row, column=2, sticky="w", padx=(1,0))
        return lv, le

    lbl_k_val,      lbl_k_est      = _fila(1, "K =")
    lbl_cf_p99_val, lbl_cf_p99_est = _fila(2, "CF.p99 =")

    return f, lbl_k_val, lbl_k_est, lbl_cf_p99_val, lbl_cf_p99_est

(frame_diag_ret,
 lbl_k_ret,   lbl_k_ret_est,
 lbl_cfp_ret, lbl_cfp_ret_est) = _hacer_recuadro_diag(frame_diag, "── RETROCESO ──", "left")

tk.Frame(frame_diag, bg=C_BORDER2, width=1).pack(side="left", fill="y", pady=6)

(frame_diag_emp,
 lbl_k_emp,   lbl_k_emp_est,
 lbl_cfp_emp, lbl_cfp_emp_est) = _hacer_recuadro_diag(frame_diag, "──  EMPUJE  ──", "left")

# ── Barra inferior ──
tk.Frame(root, bg=C_BORDER, height=1).pack(fill="x")
frame_bot = tk.Frame(root, bg=C_SURFACE, height=44)
frame_bot.pack(fill="x")
frame_bot.pack_propagate(False)

def hacer_boton(parent, texto, cmd, bg=C_SURFACE2, fg=C_TEXT, ancho=18, bold=False):
    btn = tk.Button(parent, text=texto, command=cmd, bg=bg, fg=fg,
                    activebackground=bg, activeforeground=fg,
                    relief="flat", bd=0,
                    font=(C_MONO, 9, "bold" if bold else "normal"),
                    width=ancho, cursor="hand2")
    return btn

hacer_boton(frame_bot, "📁  CARGAR CARPETA", lambda: cargar_carpeta(),
            bg=C_ACENTO, fg="white", ancho=20, bold=True
            ).pack(side="left", padx=10, pady=8)

hacer_boton(frame_bot, "📋  CARGAR DATASET", lambda: cargar_dataset(),
            bg="#1e3a5f", fg="#4f8ef7", ancho=20, bold=True
            ).pack(side="left", padx=(0,10), pady=8)

# ── Botón cargar umbrales .pkl ────────────────────────────────────────────
def _cmd_cargar_umbral():
    ruta = filedialog.askopenfilename(
        title="Cargar umbrales espectrales (.pkl)",
        filetypes=[("Pickle", "*.pkl"), ("Todos", "*.*")])
    if not ruta: return
    ok, msg = cargar_umbral_pkl(ruta)
    color = C_BUENO if ok else C_MALO
    lbl_umbral_estado.config(text=msg, fg=color)
    # Redibujar el archivo actual con el nuevo umbral
    if ok and archivos_lista and idx_actual[0] < len(archivos_lista):
        ruta_actual = archivos_lista[idx_actual[0]]
        if ruta_actual in resultados_cache:
            dibujar(resultados_cache[ruta_actual])

hacer_boton(frame_bot, "📊  CARGAR UMBRALES", _cmd_cargar_umbral,
            bg="#1a3a2a", fg="#22c55e", ancho=20, bold=True
            ).pack(side="left", padx=(0, 10), pady=8)

tk.Frame(frame_bot, bg=C_BORDER2, width=1).pack(side="left", fill="y", pady=6)
tk.Label(frame_bot, text="Piñón:", bg=C_SURFACE, fg=C_TEXT_SUB,
         font=(C_MONO, 9)).pack(side="left", padx=(10,4))

OPCIONES_PINON = [
    ("PIMA",         "PIMA  26d"),
    ("ARBOL_SEC_14", "ÁRBOL SEC. 14d"),
    ("ARBOL_SEC_15", "ÁRBOL SEC. 15d"),
]
combo_var = tk.StringVar(value="ÁRBOL SEC. 14d")
combo = ttk.Combobox(frame_bot, values=[n for _,n in OPCIONES_PINON],
                     textvariable=combo_var, state="readonly",
                     width=16, font=(C_MONO, 9))
combo.current(1)
combo.pack(side="left", padx=6)
combo.bind("<<ComboboxSelected>>", lambda e: clave_pinon.__setitem__(
    0, [k for k,n in OPCIONES_PINON if n == combo_var.get()][0]))

lbl_cargando = tk.Label(frame_bot, text="", bg=C_SURFACE, fg=C_DUDOSO,
                         font=(C_MONO, 8)); lbl_cargando.pack(side="left", padx=12)
lbl_umbral_estado = tk.Label(frame_bot, text="Sin umbrales cargados",
                              bg=C_SURFACE, fg=C_TEXT_DIM,
                              font=(C_MONO, 8))
lbl_umbral_estado.pack(side="left", padx=(0, 12))

# ── Campo RPM (tacómetro) ─────────────────────────────────────────────────
tk.Frame(frame_bot, bg=C_BORDER2, width=1).pack(side="left", fill="y", pady=6)
tk.Label(frame_bot, text="RPM:", bg=C_SURFACE, fg=C_TEXT_SUB,
         font=(C_MONO, 9)).pack(side="left", padx=(10, 4))
var_rpm = tk.StringVar(value="872")
entry_rpm = tk.Entry(frame_bot, textvariable=var_rpm, width=6,
                     bg=C_SURFACE2, fg=C_TEXT, font=(C_MONO, 9),
                     relief="flat", bd=3, justify="center")
entry_rpm.pack(side="left", ipady=3)

def _aplicar_rpm(event=None):
    try:
        rpm_nuevo = float(var_rpm.get().strip())
        if rpm_nuevo < 100 or rpm_nuevo > 5000:
            raise ValueError("RPM fuera de rango")
        _actualizar_cinematica(rpm_nuevo)
        # Limpiar caché y redibujar con nuevas RPM
        resultados_cache.clear()
        lbl_cargando.config(text=f"RPM={rpm_nuevo:.0f}  F_rot={F_ROT:.3f} Hz — recalculando...",
                            fg=C_DUDOSO)
        root.update()
        if archivos_lista and idx_actual[0] < len(archivos_lista):
            mostrar_por_indice(idx_actual[0])
        lbl_cargando.config(text=f"✓ RPM={rpm_nuevo:.0f}  F_rot={F_ROT:.3f} Hz",
                            fg=C_BUENO)
    except ValueError as e:
        lbl_cargando.config(text=f"RPM inválido: {e}", fg=C_MALO)

entry_rpm.bind("<Return>",   _aplicar_rpm)
entry_rpm.bind("<FocusOut>", _aplicar_rpm)
tk.Button(frame_bot, text="↵", command=_aplicar_rpm,
          bg=C_SURFACE2, fg=C_ACENTO, relief="flat", bd=0,
          font=(C_MONO, 9), cursor="hand2", padx=4
          ).pack(side="left", padx=(2, 0))
tk.Label(frame_bot,
         text="← → navegar    Clic en lista = saltar directamente",
         bg=C_SURFACE, fg=C_TEXT_DIM, font=(C_MONO, 7)
         ).pack(side="right", padx=16)


# ============================================================
# VISUALIZACIÓN
# ============================================================
indices_visibles = []

def refrescar_lista():
    buscar = var_buscar.get().lower()
    indices_visibles.clear()
    listbox.delete(0, "end")
    for i, ruta in enumerate(archivos_lista):
        nombre = os.path.basename(ruta)
        if buscar and buscar not in nombre.lower():
            continue
        indices_visibles.append(i)
        res = resultados_cache.get(ruta)
        # Aplicar filtro "solo válidos"
        if var_solo_validos.get() and res is not None and res.get("ok"):
            nivel_f = res.get("validacion", {}).get("nivel", "VÁLIDO")
            if nivel_f == "INVÁLIDO":
                continue
        if res is None:
            sym = "○"; color = C_TEXT_DIM
            txt = nombre.replace("engrane_","").replace(".csv","")[:28]
        elif not res["ok"]:
            sym = "✖"; color = C_MALO
            txt = nombre.replace("engrane_","").replace(".csv","")[:28]
        else:
            val   = res.get("validacion", {})
            nivel = val.get("nivel", "VÁLIDO")
            kurt  = res["kurt_ret_max"]
            txt   = nombre.replace("engrane_","").replace(".csv","")[:22]
            if nivel == "INVÁLIDO":
                sym = "✖"; color = C_INVALIDO_VAL
            elif nivel == "DUDOSO":
                sym = "~"; color = C_DUDOSO_VAL
            else:
                sym = "·"; color = C_TEXT_SUB
                if kurt > 15:  sym = "⚡"; color = C_DUDOSO
                if kurt > 30:  sym = "‼"; color = C_MALO
        listbox.insert("end", f"  {sym} {txt}")
        listbox.itemconfig("end", fg=color)


def dibujar(res):
    if not res["ok"]:
        for ax in (ax_señal, ax_kurt, ax_fft, ax_syn, ax_fft_emp):
            ax.clear(); ax.set_facecolor(C_SURFACE)
        ax_señal.text(0.5, 0.5, f"Error: {res['error']}",
                      transform=ax_señal.transAxes, ha="center",
                      color=C_MALO, fontsize=9)
        canvas_fig.draw(); return

    señal  = res["señal"]
    t_full = np.arange(len(señal)) / FS

    # ── Panel 1: Señal completa ──────────────────────────────
    ax_señal.clear(); ax_señal.set_facecolor(C_SURFACE)
    ax_señal.tick_params(colors=C_TEXT_SUB, labelsize=7)
    for sp in ax_señal.spines.values(): sp.set_color(C_BORDER2)
    ax_señal.grid(True, alpha=0.13, color=C_BORDER2)

    ax_señal.plot(t_full, señal, color="#3a3f56", linewidth=0.3, zorder=1)

    # Zonas
    ax_señal.axvspan(res["i_emp_i"]/FS, res["i_emp_f"]/FS,
                     alpha=0.20, color="#ef4444", label="Empuje")
    ax_señal.axvspan(res["i_ret_i"]/FS, res["i_ret_f"]/FS,
                     alpha=0.08, color="#4f8ef7")
    if res["i_ret_estab"] > res["i_ret_i"] + int(FS*0.02):
        ax_señal.axvspan(res["i_ret_i"]/FS, res["i_ret_estab"]/FS,
                         alpha=0.40, color="#0a0b0e", label="Trans. excl.")
    ax_señal.axvspan(res["i_ret_estab"]/FS, res["i_ret_f"]/FS,
                     alpha=0.20, color="#4f8ef7", label="Retroceso útil")

    ax_señal.axvline(res["t_sep"],   color=C_DUDOSO, lw=1.0, ls="--",
                     label=f"Sep {res['t_sep']:.2f}s")
    ax_señal.axvline(res["t_freno"], color="#a855f7", lw=1.0, ls="--",
                     label=f"Freno {res['t_freno']:.2f}s")
    if res["i_ret_estab"] > res["i_ret_i"] + int(FS*0.02):
        ax_señal.axvline(res["t_estab"], color="#34d399", lw=1.0, ls=":",
                         label=f"Estab {res['t_estab']:.2f}s")

    # Tramo central de análisis (igual que revisor_turno) — zona más intensa
    ax_señal.axvspan(res["t_tramo_emp_i"], res["t_tramo_emp_f"],
                     alpha=0.45, color="#ef4444",
                     label=f"Tramo emp ({res['tramo_seg']:.1f}s)")
    ax_señal.axvspan(res["t_tramo_ret_i"], res["t_tramo_ret_f"],
                     alpha=0.45, color="#4f8ef7",
                     label=f"Tramo ret ({res['tramo_seg']:.1f}s)")


    # ── Badge validación ──
    val_res  = res.get("validacion", {})
    val_niv  = val_res.get("nivel", "VÁLIDO")
    val_col  = {"VÁLIDO": C_VALIDO, "DUDOSO": C_DUDOSO_VAL,
                "INVÁLIDO": C_INVALIDO_VAL}.get(val_niv, C_VALIDO)
    val_razon = val_res.get("razones", [])
    val_txt  = val_razon[0][:80] if val_razon else ""

    ax_señal.set_title("Señal completa", fontsize=8, color=C_TEXT_SUB, pad=4)
    ax_señal.set_ylabel("Amplitud", fontsize=7, color=C_TEXT_SUB)
    ax_señal.set_xlabel("Tiempo (s)", fontsize=7, color=C_TEXT_SUB)
    ax_señal.legend(fontsize=6, loc="upper right",
                    facecolor=C_SURFACE2, edgecolor=C_BORDER2,
                    labelcolor=C_TEXT_SUB, ncol=3)

    # ── Panel 2: Curtosis — método Cycla NVH (umbral MAD robusto) ──────────────────
    # Criterio principal : K_global de toda la zona (un único valor diagnóstico)
    # Localización       : 10 ventanas iguales sobre cada flanco
    # Umbral ventanas    : mediana + 3.5×MAD (robusto — no se desplaza con los golpes)
    # Vecindad cíclica   : ventana alta con vecino en ±2 posiciones → cíclico
    # Esporádico         : ventana alta completamente aislada (sin vecinos en ±2)
    ax_kurt.clear(); ax_kurt.set_facecolor(C_SURFACE)
    ax_kurt.tick_params(colors=C_TEXT_SUB, labelsize=7)
    for sp in ax_kurt.spines.values(): sp.set_color(C_BORDER2)
    ax_kurt.grid(True, alpha=0.10, color=C_BORDER2)

    kg_ret   = res.get("kurt_global_ret", 3.0)
    kg_emp   = res.get("kurt_global_emp", 3.0)
    cf_ret   = res.get("cf_ret",  0.0)
    cf_emp   = res.get("cf_emp",  0.0)
    sf_ret   = res.get("sf_ret",  1.25)
    sf_emp   = res.get("sf_emp",  1.25)
    idx_ret  = res.get("idx_ret", 0.0)
    idx_emp  = res.get("idx_emp", 0.0)
    ac_ret   = res.get("autocorr_ratio_ret", 0.0)
    ac_emp   = res.get("autocorr_ratio_emp", 0.0)
    k10_ret  = res.get("k10_ret", np.array([3.0]*10))
    t10_ret = res.get("t10_ret", np.linspace(res["t_sep"], res["t_freno"], 10))
    k10_emp = res.get("k10_emp", np.array([3.0]*10))
    t10_emp = res.get("t10_emp", np.linspace(0, res["t_sep"], 10))

    def _umbral_robusto(kv):
        """
        Umbral robusto basado en mediana + MAD (Median Absolute Deviation).
        Inmune a golpes periódicos que afectan varias ventanas simultáneamente:
        - La mediana no se desplaza aunque el 40% de ventanas sean golpes
        - MAD = median(|xi - median(x)|) es equivalente robusto al sigma
        - Factor 4.5 × MAD equivale aproximadamente a 3σ para datos Gaussianos
          pero no se deja llevar por outliers (vs mean+3σ que sí lo hace)
        Retorna (base, umbral) donde base ≈ nivel típico de fondo.
        """
        if len(kv) < 3:
            return 3.0, 99.0
        med = float(np.median(kv))
        mad = float(np.median(np.abs(kv - med)))
        # Fallback: si MAD=0 (todas iguales), usar rango/6
        if mad < 0.05:
            mad = float(np.std(kv)) / 1.4826 + 0.01
        sigma_equiv = mad * 1.4826   # conversión a σ equivalente
        return med, med + 3.5 * sigma_equiv

    def _mask_ciclicos(kv, umbral):
        """
        Detecta golpes cíclicos vs esporádicos.
        Cíclico: ventana alta que tiene AL MENOS UN vecino alto dentro de ±2 posiciones
                 (captura patrones espaciados tipo 1,4,7 en 10 ventanas)
        Esporádico: ventana alta completamente aislada (ningún vecino alto en ±2)
        """
        n = len(kv); alto = kv > umbral
        ciclico    = np.zeros(n, dtype=bool)
        esporadico = np.zeros(n, dtype=bool)
        for i in range(n):
            if not alto[i]:
                continue
            # Buscar vecinos en ventana ±2 (no sólo inmediatos)
            vecinos = [j for j in range(max(0, i-2), min(n, i+3)) if j != i]
            tiene_vecino_alto = any(alto[j] for j in vecinos)
            if tiene_vecino_alto:
                ciclico[i] = True
            else:
                esporadico[i] = True
        return ciclico, esporadico

    # ── Paso 1: umbral robusto inicial (mediana+MAD) ──
    mean_ret_0, umbral_ret_0 = _umbral_robusto(k10_ret)
    mean_emp_0, umbral_emp_0 = _umbral_robusto(k10_emp)

    # ── Paso 2: clasificar con umbral robusto ──
    cic_r, esp_r = _mask_ciclicos(k10_ret, umbral_ret_0)
    cic_e, esp_e = _mask_ciclicos(k10_emp, umbral_emp_0)

    # ── Paso 3: recalcular umbral excluyendo los ya detectados como cíclicos
    #    (refina el nivel de fondo eliminando los propios golpes del cálculo)
    def _umbral_refinado(kv, excluir):
        kl = kv[~excluir]
        if len(kl) >= 3:
            return _umbral_robusto(kl)
        return _umbral_robusto(kv)

    mean_ret_l, umbral_ret_l = _umbral_refinado(k10_ret, cic_r)
    mean_emp_l, umbral_emp_l = _umbral_refinado(k10_emp, cic_e)

    # ── Paso 4: clasificación final con umbral refinado ──
    cic_r, esp_r = _mask_ciclicos(k10_ret, umbral_ret_l)
    cic_e, esp_e = _mask_ciclicos(k10_emp, umbral_emp_l)

    n_altos_ret = int(np.sum(cic_r)); n_altos_emp = int(np.sum(cic_e))
    n_esp_ret   = int(np.sum(esp_r)); n_esp_emp   = int(np.sum(esp_e))

    ancho_r = (t10_ret[-1] - t10_ret[0]) / len(t10_ret) * 0.75 if len(t10_ret) > 1 else 0.05
    ancho_e = (t10_emp[-1] - t10_emp[0]) / len(t10_emp) * 0.75 if len(t10_emp) > 1 else 0.05

    for i, (t, k) in enumerate(zip(t10_ret, k10_ret)):
        if esp_r[i]:   col, alpha = C_TEXT_DIM, 0.35
        elif cic_r[i]: col, alpha = C_MALO,     0.85
        elif k > umbral_ret_l * 0.85: col, alpha = C_DUDOSO, 0.70
        else:          col, alpha = "#60a5fa",  0.45
        ax_kurt.bar(t, k, width=ancho_r, color=col, alpha=alpha, align="center", zorder=3)

    for i, (t, k) in enumerate(zip(t10_emp, k10_emp)):
        if esp_e[i]:   col, alpha = C_TEXT_DIM, 0.35
        elif cic_e[i]: col, alpha = "#f87171",  0.85
        elif k > umbral_emp_l * 0.85: col, alpha = C_DUDOSO, 0.70
        else:          col, alpha = "#f87171",  0.30
        ax_kurt.bar(t, k, width=ancho_e, color=col, alpha=alpha, align="center", zorder=3)

    ax_kurt.axhline(umbral_ret_l, color="#60a5fa", lw=1.1, ls="--", alpha=0.8,
                    label=f"Umbral ret = {umbral_ret_l:.1f}")
    ax_kurt.axhline(mean_ret_l,   color="#60a5fa", lw=0.6, ls=":",  alpha=0.5)
    ax_kurt.axhline(umbral_emp_l, color="#f87171", lw=1.1, ls="--", alpha=0.8,
                    label=f"Umbral emp = {umbral_emp_l:.1f}")
    ax_kurt.axhline(mean_emp_l,   color="#f87171", lw=0.6, ls=":",  alpha=0.5)

    # ── Badges K_global — valor único diagnóstico por zona (método Cycla) ──
    # Umbrales orientativos basados en la documentación del especialista:
    #   K < 3.5  → Normal (distribución gaussiana pura ≈ 3.0)
    #   3.5–6    → Levemente elevado — posible inicio de defecto
    #   6–10     → Sospechoso — revisar
    #   > 10     → Golpe confirmado (nick en diente)
    def _color_k(k):
        if k > 10:  return C_MALO
        if k > 6:   return C_DUDOSO
        if k > 3.5: return "#facc15"   # amarillo suave
        return C_BUENO

    def _label_k(k):
        if k > 10:  return "GOLPE"
        if k > 6:   return "SOSP."
        if k > 3.5: return "ELEV."
        return "OK"

    col_kg_r = _color_k(kg_ret)
    col_kg_e = _color_k(kg_emp)

    def _color_idx(idx):
        if idx > 1.5: return C_MALO
        if idx > 1.0: return C_DUDOSO
        return C_BUENO

    def _label_idx(idx):
        if idx > 1.5: return "DEFECTO"
        if idx > 1.0: return "SOSP."
        return "OK"

    cf_p99_ret = res.get("cf_p99_ret", 0.0)
    cf_p99_emp = res.get("cf_p99_emp", 0.0)

    # CF.p99 — umbrales calibrados por piñón
    def _label_cfp(cf):
        if cf > CF_DEF:  return "[DEFECTO]"
        if cf > CF_SOSP: return "[SOSP.]"
        return "[OK]"
    def _color_cfp(cf):
        if cf > CF_DEF:  return C_MALO
        if cf > CF_SOSP: return C_DUDOSO
        return C_BUENO

    # Umbrales calibrados por piñón (documentación del proyecto)
    pinon_key_diag = res.get("pinon_key", clave_pinon[0])
    if "PIMA" in pinon_key_diag.upper():
        K_DEFECTO = 3.6;  K_SOSP = 3.5;  CF_DEF = 3.2;  CF_SOSP = 3.0
    else:  # AS14 / AS15
        K_DEFECTO = 5.0;  K_SOSP = 4.5;  CF_DEF = 3.5;  CF_SOSP = 3.2

    def _diagnostico_conjunto(kg, cfp):
        """
        Diagnóstico combinado K + CF.p99 — lógica AND por piñón:
          PIMA:   K > 3.6 AND CF.p99 > 3.2
          AS:     K > 4.4 AND CF.p99 > 3.3
        """
        k_alto  = kg  > K_DEFECTO
        k_sosp  = kg  > K_SOSP
        cf_alto = cfp > CF_DEF
        cf_sosp = cfp > CF_SOSP
        if k_alto and cf_alto:
            return "DEFECTO",  C_MALO
        if k_sosp or cf_sosp:
            return "SOSP.",    C_DUDOSO
        return "OK",           C_BUENO

    def _actualizar_zona(lbl_k, lbl_k_e, lbl_cfp, lbl_cfp_e,
                         frame, kg, col_kg, cfp):
        diag, col_diag = _diagnostico_conjunto(kg, cfp)
        lbl_k.config(   text=f"{kg:.2f}",          fg=col_kg)
        lbl_k_e.config( text=f"[{_label_k(kg)}]",  fg=col_kg)
        lbl_cfp.config( text=f"{cfp:.2f}",          fg=_color_cfp(cfp))
        lbl_cfp_e.config(text=_label_cfp(cfp),      fg=_color_cfp(cfp))
        # Borde = diagnóstico conjunto AND
        frame.config(highlightbackground=col_diag,
                     highlightthickness=2, highlightcolor=col_diag)

    _actualizar_zona(lbl_k_ret, lbl_k_ret_est, lbl_cfp_ret, lbl_cfp_ret_est,
                     frame_diag_ret, kg_ret, col_kg_r, cf_p99_ret)

    _actualizar_zona(lbl_k_emp, lbl_k_emp_est, lbl_cfp_emp, lbl_cfp_emp_est,
                     frame_diag_emp, kg_emp, col_kg_e, cf_p99_emp)

    # Eliminar badges matplotlib del gráfico (ya no se necesitan)
    for ann in getattr(ax_kurt, '_idx_badges', []):
        try: ann.remove()
        except Exception: pass
    ax_kurt._idx_badges = []

    ax_kurt.axvline(res["t_sep"], color=C_DUDOSO, lw=0.9, ls="--", alpha=0.6)
    if res["i_ret_estab"] > res["i_ret_i"] + int(FS*0.02):
        ax_kurt.axvline(res["t_estab"], color="#34d399", lw=0.8, ls=":", alpha=0.5)

    hay_golpe    = (n_altos_ret > 0 or n_altos_emp > 0)
    hay_k_alto   = (kg_ret > 10 or kg_emp > 10)
    hay_k_sosp   = (kg_ret > 6  or kg_emp > 6)
    if hay_golpe or hay_k_alto:
        estado_k = f"GOLPE  ret:{n_altos_ret}v  emp:{n_altos_emp}v"
        col_tit  = C_MALO
    elif hay_k_sosp:
        estado_k = f"SOSPECHOSO  K_ret={kg_ret:.1f}  K_emp={kg_emp:.1f}"
        col_tit  = C_DUDOSO
    else:
        estado_k = f"Normal  K_ret={kg_ret:.1f}  K_emp={kg_emp:.1f}"
        col_tit  = C_BUENO
    esp_str = f"  ({n_esp_ret+n_esp_emp} esp. excl.)" if (n_esp_ret+n_esp_emp) > 0 else ""

    ax_kurt.set_title(
        f"Curtosis  [fórmula Cycla]  —  10 ventanas/flanco  │  {estado_k}{esp_str}",
        fontsize=7, color=col_tit, pad=4)
    ax_kurt.set_ylabel("K (adimensional)", fontsize=7, color=C_TEXT_SUB)
    ax_kurt.set_xlabel("Tiempo (s)", fontsize=7, color=C_TEXT_SUB)
    ax_kurt.legend(fontsize=6, loc="upper left",
                   facecolor=C_SURFACE2, edgecolor=C_BORDER2,
                   labelcolor=C_TEXT_SUB, ncol=2)
    todos_k = np.concatenate([k10_emp, k10_ret])
    y_top = max(max(umbral_ret_l, umbral_emp_l)*1.35, float(np.max(todos_k))*1.20, 10.0)
    ax_kurt.set_ylim(bottom=0, top=y_top)

    # ── Panel 3: Espectro en órdenes ────────────────────────
    ax_fft.clear(); ax_fft.set_facecolor(C_SURFACE)
    ax_fft.tick_params(colors=C_TEXT_SUB, labelsize=7)
    for sp in ax_fft.spines.values(): sp.set_color(C_BORDER2)
    ax_fft.grid(True, alpha=0.13, color=C_BORDER2)

    ordenes   = res["ordenes"]
    fft_db    = res["fft_db"]
    fft_suav  = res["fft_suav"]
    orden_gmf = res["orden_gmf"]
    o_max     = orden_gmf * 3.5
    gmf_hz    = orden_gmf * F_ROT   # frecuencia GMF en Hz

    if len(ordenes) > 1:
        mask = (ordenes > 0) & (ordenes <= o_max)
        # Eje X en Hz (multiplicar órdenes × F_ROT)
        hz_vis = ordenes[mask] * F_ROT
        ax_fft.plot(hz_vis, fft_db[mask],
                    color="#60a5fa", linewidth=0.7, alpha=0.6, label="Espectro ret.")
        ax_fft.plot(hz_vis, fft_suav[mask],
                    color="#60a5fa", linewidth=1.3, ls="--", alpha=0.9, label="Tendencia")
        ax_fft.fill_between(hz_vis, fft_suav[mask], fft_db[mask],
                             alpha=0.08, color="#60a5fa")

        fft_vis  = fft_db[mask]
        ords_vis = ordenes[mask]   # se mantiene en órdenes para lógica interna

        # ── Umbral estadístico orden a orden (dataset completo) ──────────
        ud = umbral_dataset_ret
        ud_ret = ud.get("ret") if ud else None
        if ud and ud_ret and ud_ret.get("umbral") is not None:
            grilla_ud    = ud["ordenes"]
            umbral_crudo = np.interp(ords_vis, grilla_ud, ud_ret["umbral"])
            _wv = min(51, len(umbral_crudo) - 1)
            if _wv % 2 == 0: _wv -= 1
            if _wv >= 3:
                from scipy.signal import savgol_filter as _sgf
                umbral_curva = _sgf(umbral_crudo, window_length=_wv, polyorder=2)
            else:
                umbral_curva = umbral_crudo
            n_arch = ud_ret.get("n_archivos", "?")
            ax_fft.plot(hz_vis, umbral_curva, color=C_MALO, lw=1.2, ls="-.",
                        alpha=0.85, label=f"μ+3σ dataset ({n_arch} arch.)")
        else:
            suav_vis     = fft_suav[mask]
            sigma_fft    = float(np.std(fft_vis - suav_vis))
            umbral_curva = suav_vis + 3.0 * sigma_fft
            ax_fft.plot(hz_vis, umbral_curva, color=C_MALO, lw=0.9, ls="-.",
                        alpha=0.75, label=f"Tend.+3σ local (σ={sigma_fft:.1f} dB)")

        # Peaks que superan el umbral — marcados con X roja
        idx_peaks, _ = find_peaks(fft_vis, height=umbral_curva.min(), distance=3)
        idx_peaks = np.array([i for i in idx_peaks if fft_vis[i] > umbral_curva[i]])
        if len(idx_peaks) > 0:
            ax_fft.scatter(hz_vis[idx_peaks], fft_vis[idx_peaks],
                           marker="x", color=C_MALO, s=55, linewidths=1.5,
                           zorder=5, label=f"Picos anómalos ({len(idx_peaks)})")

        # ── Detección de sireneo en armónicos GMF ──────────────────────
        TOL_GMF   = 2.0
        C_SIRENEO = "#f59e0b"
        sireneo_ret = []

        for k in [1, 2, 3]:
            o_k = orden_gmf * k
            if o_k > o_max:
                continue
            banda = (ords_vis >= o_k - TOL_GMF) & (ords_vis <= o_k + TOL_GMF)
            if not np.any(banda):
                continue
            idx_max_banda = int(np.argmax(fft_vis[banda]))
            ords_banda    = ords_vis[banda]
            fft_banda     = fft_vis[banda]
            umb_banda     = umbral_curva[banda]
            val_peak      = float(fft_banda[idx_max_banda])
            ord_peak      = float(ords_banda[idx_max_banda])
            hz_peak       = ord_peak * F_ROT
            umb_peak      = float(umb_banda[idx_max_banda])

            supera = val_peak > umb_peak
            sireneo_ret.append((k, ord_peak, val_peak, umb_peak, supera))

            color_s = C_SIRENEO if supera else C_TEXT_DIM
            ax_fft.scatter([hz_peak], [val_peak],
                           marker="^", color=color_s, s=70,
                           linewidths=1.5, zorder=6,
                           label=f"GMF×{k}" if k == 1 else None)
            ax_fft.annotate(f"{val_peak:.1f} dB",
                            xy=(hz_peak, val_peak),
                            xytext=(4, 5), textcoords="offset points",
                            fontsize=6, color=color_s, fontweight="bold")

        n_sireneo_ret = sum(1 for _, _, _, _, sup in sireneo_ret if sup)
        if n_sireneo_ret > 0:
            ax_fft.text(0.02, 0.97,
                        f"⚠ SIRENEO  ({n_sireneo_ret} armónico{'s' if n_sireneo_ret>1 else ''} GMF)",
                        transform=ax_fft.transAxes,
                        fontsize=7, fontweight="bold",
                        color=C_SIRENEO, va="top",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor=C_SURFACE2, edgecolor=C_SIRENEO,
                                  alpha=0.85))

        # GMF y armónicos — líneas verticales en Hz
        for k, ls in [(1,"--"),(2,":"),(3,":")]:
            f_k = gmf_hz * k
            if f_k <= o_max * F_ROT:
                ax_fft.axvline(f_k, color=C_DUDOSO, lw=0.9, ls=ls, alpha=0.7,
                               label=f"{k}×GMF ({f_k:.0f} Hz)" if k == 1 else None)

        # Sidebands ±1×F_ROT en Hz
        for delta in [-1, +1]:
            f_sb = gmf_hz + delta * F_ROT
            if 0 < f_sb <= o_max * F_ROT:
                ax_fft.axvline(f_sb, color="#f87171", lw=0.6, ls=":", alpha=0.5)

    ax_fft.set_title(f"Espectro retroceso  |  GMF = {gmf_hz:.1f} Hz  (orden {orden_gmf:.0f})",
                      fontsize=8, color=C_TEXT_SUB, pad=4)
    ax_fft.set_xlabel("Frecuencia (Hz)", fontsize=7, color=C_TEXT_SUB)
    ax_fft.set_ylabel("dB", fontsize=7, color=C_TEXT_SUB)
    ax_fft.legend(fontsize=6, loc="upper right",
                  facecolor=C_SURFACE2, edgecolor=C_BORDER2,
                  labelcolor=C_TEXT_SUB)

    # ── Panel Promedio Síncrono ──────────────────────────────
    ax_syn.clear(); ax_syn.set_facecolor(C_SURFACE)
    ax_syn.tick_params(colors=C_TEXT_SUB, labelsize=7)
    for sp in ax_syn.spines.values(): sp.set_color(C_BORDER2)
    ax_syn.grid(True, alpha=0.13, color=C_BORDER2)

    # Mostrar empuje y retroceso superpuestos si existen
    per_emp = res.get("per_emp")
    per_ret = res.get("per_ret")

    for per, color, lbl_txt in [
        (per_emp, "#f87171", "Empuje"),
        (per_ret, "#60a5fa", "Retroceso"),
    ]:
        if per is None or per["n_ciclos"] < 2: continue
        t_syn = per["t_syn"]
        p_syn = per["promedio_syn"]
        ratio = per["ratio_syn"]
        n_cic = per["n_ciclos"]
        col_r = C_BUENO if ratio < 2 else C_DUDOSO if ratio < 4 else C_MALO
        ax_syn.plot(t_syn, p_syn, color=color, linewidth=1.1,
                    label=f"{lbl_txt}  ratio={ratio:.1f}  ({n_cic} ciclos)")
        ax_syn.fill_between(t_syn, 0, p_syn, alpha=0.08, color=color)

    # Línea vertical en posición del pico del retroceso
    if per_ret and per_ret["n_ciclos"] >= 2:
        idx_pk = int(np.argmax(np.abs(per_ret["promedio_syn"])))
        t_pk   = per_ret["t_syn"][idx_pk]
        ratio  = per_ret["ratio_syn"]
        col_r  = C_BUENO if ratio < 2 else C_DUDOSO if ratio < 4 else C_MALO
        ax_syn.axvline(t_pk, color=col_r, lw=1.0, ls="--", alpha=0.7,
                       label=f"Pico ret @ {t_pk:.1f}ms")

    ax_syn.axhline(0, color=C_TEXT_DIM, lw=0.5, alpha=0.4)
    ax_syn.set_title(f"Promedio síncrono (1 giro = {T_ROT*1000:.1f}ms)",
                      fontsize=8, color=C_TEXT_SUB, pad=4)
    ax_syn.set_xlabel("Tiempo dentro del giro (ms)", fontsize=7, color=C_TEXT_SUB)
    ax_syn.set_ylabel("Amplitud media", fontsize=7, color=C_TEXT_SUB)
    ax_syn.legend(fontsize=6, loc="upper right",
                  facecolor=C_SURFACE2, edgecolor=C_BORDER2,
                  labelcolor=C_TEXT_SUB)

    # ── Panel Autocorrelación ────────────────────────────────
    # ── Panel FFT empuje ─────────────────────────────────────────
    ax_fft_emp.clear(); ax_fft_emp.set_facecolor(C_SURFACE)
    ax_fft_emp.tick_params(colors=C_TEXT_SUB, labelsize=7)
    for sp in ax_fft_emp.spines.values(): sp.set_color(C_BORDER2)
    ax_fft_emp.grid(True, alpha=0.13, color=C_BORDER2)

    ordenes_emp  = res.get("ordenes_emp",  np.array([0]))
    fft_db_emp   = res.get("fft_db_emp",   np.array([0]))
    fft_suav_emp = res.get("fft_suav_emp", np.array([0]))

    if len(ordenes_emp) > 1:
        mask_e  = (ordenes_emp > 0) & (ordenes_emp <= o_max)
        hz_vis_e = ordenes_emp[mask_e] * F_ROT   # eje X en Hz
        ax_fft_emp.plot(hz_vis_e, fft_db_emp[mask_e],
                        color="#f87171", linewidth=0.7, alpha=0.6, label="Espectro emp.")
        ax_fft_emp.plot(hz_vis_e, fft_suav_emp[mask_e],
                        color="#f87171", linewidth=1.3, ls="--", alpha=0.9, label="Tendencia")
        ax_fft_emp.fill_between(hz_vis_e, fft_suav_emp[mask_e], fft_db_emp[mask_e],
                                alpha=0.08, color="#f87171")

        fft_vis_e  = fft_db_emp[mask_e]
        ords_vis_e = ordenes_emp[mask_e]   # se mantiene en órdenes para lógica interna

        # ── Umbral estadístico orden a orden (dataset completo) ──────────
        ue = umbral_dataset_emp
        ue_emp = ue.get("emp") if ue else None
        if ue and ue_emp and ue_emp.get("umbral") is not None:
            grilla_ue      = ue["ordenes"]
            umbral_e_crudo = np.interp(ords_vis_e, grilla_ue, ue_emp["umbral"])
            _we = min(51, len(umbral_e_crudo) - 1)
            if _we % 2 == 0: _we -= 1
            if _we >= 3:
                from scipy.signal import savgol_filter as _sgf2
                umbral_e = _sgf2(umbral_e_crudo, window_length=_we, polyorder=2)
            else:
                umbral_e = umbral_e_crudo
            n_arch_e = ue_emp.get("n_archivos", "?")
            ax_fft_emp.plot(hz_vis_e, umbral_e, color=C_MALO, lw=1.2, ls="-.",
                            alpha=0.85, label=f"μ+3σ dataset ({n_arch_e} arch.)")
        else:
            suav_vis_e = fft_suav_emp[mask_e]
            sigma_e    = float(np.std(fft_vis_e - suav_vis_e))
            umbral_e   = suav_vis_e + 3.0 * sigma_e
            ax_fft_emp.plot(hz_vis_e, umbral_e, color=C_MALO, lw=0.9, ls="-.",
                            alpha=0.75, label=f"Tend.+3σ local (σ={sigma_e:.1f} dB)")

        idx_peaks_e, _ = find_peaks(fft_vis_e, height=umbral_e.min(), distance=3)
        idx_peaks_e = np.array([i for i in idx_peaks_e if fft_vis_e[i] > umbral_e[i]])
        if len(idx_peaks_e) > 0:
            ax_fft_emp.scatter(hz_vis_e[idx_peaks_e], fft_vis_e[idx_peaks_e],
                               marker="x", color=C_MALO, s=55, linewidths=1.5,
                               zorder=5, label=f"Picos anómalos ({len(idx_peaks_e)})")

        # ── Detección de sireneo en armónicos GMF ──────────────────────
        TOL_GMF    = 2.0
        C_SIRENEO  = "#f59e0b"
        sireneo_emp = []

        for k in [1, 2, 3]:
            o_k = orden_gmf * k
            if o_k > o_max:
                continue
            banda = (ords_vis_e >= o_k - TOL_GMF) & (ords_vis_e <= o_k + TOL_GMF)
            if not np.any(banda):
                continue
            idx_max_banda = int(np.argmax(fft_vis_e[banda]))
            ords_banda    = ords_vis_e[banda]
            fft_banda     = fft_vis_e[banda]
            umb_banda     = umbral_e[banda]
            val_peak      = float(fft_banda[idx_max_banda])
            ord_peak      = float(ords_banda[idx_max_banda])
            hz_peak_e     = ord_peak * F_ROT
            umb_peak      = float(umb_banda[idx_max_banda])

            supera = val_peak > umb_peak
            sireneo_emp.append((k, ord_peak, val_peak, umb_peak, supera))

            color_s = C_SIRENEO if supera else C_TEXT_DIM
            ax_fft_emp.scatter([hz_peak_e], [val_peak],
                               marker="^", color=color_s, s=70,
                               linewidths=1.5, zorder=6,
                               label=f"GMF×{k}" if k == 1 else None)
            ax_fft_emp.annotate(f"{val_peak:.1f} dB",
                                xy=(hz_peak_e, val_peak),
                                xytext=(4, 5), textcoords="offset points",
                                fontsize=6, color=color_s, fontweight="bold")

        n_sireneo_emp = sum(1 for _, _, _, _, sup in sireneo_emp if sup)
        if n_sireneo_emp > 0:
            ax_fft_emp.text(0.02, 0.97,
                            f"⚠ SIRENEO  ({n_sireneo_emp} armónico{'s' if n_sireneo_emp>1 else ''} GMF)",
                            transform=ax_fft_emp.transAxes,
                            fontsize=7, fontweight="bold",
                            color=C_SIRENEO, va="top",
                            bbox=dict(boxstyle="round,pad=0.3",
                                      facecolor=C_SURFACE2, edgecolor=C_SIRENEO,
                                      alpha=0.85))

        # GMF y armónicos — líneas verticales en Hz
        for k, ls in [(1, "--"), (2, ":"), (3, ":")]:
            f_k = gmf_hz * k
            if f_k <= o_max * F_ROT:
                ax_fft_emp.axvline(f_k, color=C_DUDOSO, lw=0.9, ls=ls, alpha=0.7,
                                   label=f"{k}×GMF ({f_k:.0f} Hz)" if k == 1 else None)

        # Sidebands ±1×F_ROT en Hz
        for delta in [-1, +1]:
            f_sb = gmf_hz + delta * F_ROT
            if 0 < f_sb <= o_max * F_ROT:
                ax_fft_emp.axvline(f_sb, color="#f87171", lw=0.6, ls=":", alpha=0.5)

    ax_fft_emp.set_title(f"Espectro empuje  |  GMF = {gmf_hz:.1f} Hz  (orden {orden_gmf:.0f})",
                          fontsize=8, color=C_TEXT_SUB, pad=4)
    ax_fft_emp.set_xlabel("Frecuencia (Hz)", fontsize=7, color=C_TEXT_SUB)
    ax_fft_emp.set_ylabel("dB", fontsize=7, color=C_TEXT_SUB)
    ax_fft_emp.legend(fontsize=6, loc="upper right",
                      facecolor=C_SURFACE2, edgecolor=C_BORDER2,
                      labelcolor=C_TEXT_SUB)

    fig.subplots_adjust(left=0.06, right=0.97, top=0.95, bottom=0.07, hspace=0.45, wspace=0.28)
    canvas_fig.draw()

    # Actualizar header con badge de validación
    val   = res.get("validacion", {})
    nivel = val.get("nivel", "VÁLIDO")
    v_col = {
        "VÁLIDO":   C_VALIDO,
        "DUDOSO":   C_DUDOSO_VAL,
        "INVÁLIDO": C_INVALIDO_VAL,
    }.get(nivel, C_TEXT)
    razones = val.get("razones", [])
    razon_txt = f"  ·  {razones[0][:70]}" if razones else ""

    lbl_nombre_archivo.config(
        text=f"[{nivel}]  {res['nombre'].replace('.csv','')}",
        fg=v_col)
    stats_base = (f"RMS_ret={res['rms_ret']:.5f}  ·  "
                  f"Kurt_max={res['kurt_ret_max']:.1f}  ·  "
                  f"Ret_útil={res['dur_ret']:.3f}s{razon_txt}")
    lbl_stats_archivo.config(text=stats_base, fg=C_TEXT_SUB)


def mostrar_por_indice(i):
    if i < 0 or i >= len(archivos_lista): return
    idx_actual[0] = i
    ruta = archivos_lista[i]
    lbl_idx.config(text=f"{i+1} / {len(archivos_lista)}")

    if ruta not in resultados_cache:
        lbl_nombre_archivo.config(text="Analizando...", fg=C_DUDOSO)
        root.update()
        gmf = PINONES[clave_pinon[0]]["gmf"]
        res = analizar_archivo(ruta, gmf)
        resultados_cache[ruta] = res

    dibujar(resultados_cache[ruta])

    # Sincronizar listbox
    for j, idx in enumerate(indices_visibles):
        if idx == i:
            listbox.selection_clear(0, "end")
            listbox.selection_set(j)
            listbox.see(j)
            break


def navegar(delta):
    nuevo = idx_actual[0] + delta
    nuevo = max(0, min(len(archivos_lista)-1, nuevo))
    mostrar_por_indice(nuevo)


def on_lista_select(event=None):
    sel = listbox.curselection()
    if not sel or not indices_visibles: return
    i_vis = sel[0]
    if i_vis >= len(indices_visibles): return
    mostrar_por_indice(indices_visibles[i_vis])


listbox.bind("<<ListboxSelect>>", on_lista_select)
root.bind("<Left>",  lambda e: navegar(-1))
root.bind("<Right>", lambda e: navegar(+1))


def cargar_carpeta():
    carpeta = filedialog.askdirectory(title="Seleccionar carpeta con engrane_*.csv")
    if not carpeta: return

    # Auto-detectar piñón
    nc = os.path.basename(carpeta).upper()
    for k, _ in OPCIONES_PINON:
        if k in nc or nc in k:
            clave_pinon[0] = k
            for i, (ck, cn) in enumerate(OPCIONES_PINON):
                if ck == k: combo.current(i); combo_var.set(cn)
            break

    archivos = sorted(glob.glob(os.path.join(carpeta, "engrane_*.csv")))
    if not archivos:
        from tkinter import messagebox
        messagebox.showwarning("Sin archivos",
            f"No se encontraron engrane_*.csv en:\n{carpeta}")
        return

    archivos_lista.clear()
    resultados_cache.clear()
    archivos_lista.extend(archivos)
    idx_actual[0] = 0

    lbl_info_hdr.config(text=f"{len(archivos)} archivos  ·  {carpeta}")
    lbl_cargando.config(text=f"Pre-analizando {len(archivos)} archivos...")
    refrescar_lista()
    root.update()

    def _pre_analizar():
        gmf = PINONES[clave_pinon[0]]["gmf"]
        for i, ruta in enumerate(archivos_lista):
            if ruta not in resultados_cache:
                res = analizar_archivo(ruta, gmf)
                resultados_cache[ruta] = res
            root.after(0, lambda i=i: (
                lbl_cargando.config(
                    text=f"Analizando {i+1}/{len(archivos_lista)}..."),
                refrescar_lista()
            ))

        # Estadísticos globales del turno
        vals_kurt = [r["kurt_ret_max"] for r in resultados_cache.values()
                     if r.get("ok")]
        vals_rms  = [r["rms_ret"] for r in resultados_cache.values()
                     if r.get("ok")]
        vals_cv   = [r["cv_rms_ret"] for r in resultados_cache.values()
                     if r.get("ok")]
        if vals_kurt:
            txt = (
                f"Kurt_max:  p50={np.median(vals_kurt):.1f}  "
                f"p90={np.percentile(vals_kurt,90):.1f}  "
                f"max={np.max(vals_kurt):.1f}\n"
                f"RMS_ret:  p50={np.median(vals_rms):.5f}  "
                f"max={np.max(vals_rms):.5f}\n"
                f"CV_rms:   p50={np.median(vals_cv):.2f}  "
                f"p90={np.percentile(vals_cv,90):.2f}  "
                f"max={np.max(vals_cv):.2f}"
            )
            root.after(0, lambda: (
                lbl_stats_mini.config(text=txt),
                lbl_cargando.config(text=""),
                mostrar_por_indice(0),
            ))
        else:
            root.after(0, lambda: lbl_cargando.config(text=""))

    threading.Thread(target=_pre_analizar, daemon=True).start()
    """
    Carga archivos desde un CSV de dataset o un TXT con rutas.
    CSV: necesita columna 'ruta' o 'archivo'.
    TXT: una ruta por línea.
    """
    from tkinter import messagebox

    ruta_ds = filedialog.askopenfilename(
        title="Seleccionar dataset CSV o lista TXT",
        filetypes=[("CSV / TXT", "*.csv *.txt"), ("Todos", "*.*")])
    if not ruta_ds:
        return

    rutas = []
    try:
        if ruta_ds.lower().endswith(".csv"):
            df_ds = pd.read_csv(ruta_ds)
            col_ruta = None
            for c in ["ruta", "archivo", "path", "file"]:
                if c in df_ds.columns:
                    col_ruta = c
                    break
            if col_ruta is None:
                messagebox.showerror("Error",
                    f"El CSV no tiene columna 'ruta' o 'archivo'.\n"
                    f"Columnas encontradas: {', '.join(df_ds.columns)}")
                return

            carpeta_base = os.path.dirname(ruta_ds)
            for r in df_ds[col_ruta].dropna():
                r = str(r)
                if not os.path.isabs(r):
                    r = os.path.join(carpeta_base, r)
                if os.path.exists(r):
                    rutas.append(r)

            # Auto-seleccionar piñón si el CSV tiene columna pinon
            if "pinon" in df_ds.columns:
                pinon_freq = df_ds["pinon"].mode()
                if len(pinon_freq) > 0:
                    p = str(pinon_freq.iloc[0]).upper()
                    for k, _ in OPCIONES_PINON:
                        if k in p or p in k:
                            clave_pinon[0] = k
                            for i, (ck, cn) in enumerate(OPCIONES_PINON):
                                if ck == k:
                                    combo.current(i)
                                    combo_var.set(cn)
                            break
        else:
            with open(ruta_ds, encoding="utf-8") as f:
                for linea in f:
                    r = linea.strip()
                    if r and os.path.exists(r):
                        rutas.append(r)

    except Exception as ex:
        messagebox.showerror("Error al leer dataset", str(ex))
        return

    if not rutas:
        messagebox.showwarning("Sin archivos",
            "No se encontraron archivos válidos en el dataset.\n"
            "Verifica que las rutas en el CSV sean correctas.")
        return

    archivos_lista.clear()
    resultados_cache.clear()
    archivos_lista.extend(rutas)
    idx_actual[0] = 0

    lbl_info_hdr.config(
        text=f"{len(rutas)} archivos desde dataset  ·  {os.path.basename(ruta_ds)}")
    lbl_cargando.config(text=f"Pre-analizando {len(rutas)} archivos...")
    refrescar_lista()
    root.update()

    def _pre_analizar_ds():
        gmf = PINONES[clave_pinon[0]]["gmf"]
        for i, ruta in enumerate(archivos_lista):
            if ruta not in resultados_cache:
                res = analizar_archivo(ruta, gmf)
                resultados_cache[ruta] = res
            root.after(0, lambda i=i: (
                lbl_cargando.config(
                    text=f"Analizando {i+1}/{len(archivos_lista)}..."),
                refrescar_lista()
            ))

        vals_kurt = [r["kurt_ret_max"] for r in resultados_cache.values()
                     if r.get("ok")]
        vals_rms  = [r["rms_ret"]      for r in resultados_cache.values()
                     if r.get("ok")]
        vals_cv   = [r["cv_rms_ret"]   for r in resultados_cache.values()
                     if r.get("ok")]
        if vals_kurt:
            txt = (
                f"Kurt_max:  p50={np.median(vals_kurt):.1f}  "
                f"p90={np.percentile(vals_kurt,90):.1f}  "
                f"max={np.max(vals_kurt):.1f}\n"
                f"RMS_ret:  p50={np.median(vals_rms):.5f}  "
                f"max={np.max(vals_rms):.5f}\n"
                f"CV_rms:   p50={np.median(vals_cv):.2f}  "
                f"p90={np.percentile(vals_cv,90):.2f}  "
                f"max={np.max(vals_cv):.2f}"
            )
            root.after(0, lambda: (
                lbl_stats_mini.config(text=txt),
                lbl_cargando.config(text=""),
                mostrar_por_indice(0),
            ))
        else:
            root.after(0, lambda: lbl_cargando.config(text=""))

    threading.Thread(target=_pre_analizar_ds, daemon=True).start()

root.mainloop()
