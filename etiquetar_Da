"""
etiquetar_dataset.py
====================
Etiquetado semi-automático de grabaciones DEMM para ML.

Segmentación y cálculo de K idénticos al explorador_dataset.py.
Usa funciones embebidas — no depende de demm_core para segmentar.
demm_core.py se usa solo para validación (nivel VÁLIDO/DUDOSO/INVÁLIDO).

────────────────────────────────────────────────────────────────────────
UMBRALES ACTUALES
────────────────────────────────────────────────────────────────────────
  PIMA   : K > 3.6  AND  CF.p99 > 3.2  → MALO
  AS     : K > 4.4  AND  CF.p99 > 3.3  → MALO  (ARB_14 / ARB_15)

  Lógica AND por zona:
    MALO    : (K_ret > K AND CF_ret > CF)  OR  (K_emp > K AND CF_emp > CF)
    REVISAR : un solo umbral superado en cualquier zona (XOR)
    BUENO   : todos los valores bajo umbral
    INVÁLIDO: grabación rechazada por el validador

────────────────────────────────────────────────────────────────────────
COMANDOS DE USO
────────────────────────────────────────────────────────────────────────
  # Etiquetar AS (AS_1):
  cd "C:/Users/mcorn/Downloads"
  py etiquetar_dataset.py /
      --carpeta "C:/Users/mcorn/OneDrive/Desktop/HORSE/DATOS/AS_1" /
      --modelo  "C:/Users/mcorn/OneDrive/Desktop/HORSE/modelo_identificador_20260310_0931.pkl" /
      --salida  "C:/Users/mcorn/OneDrive/Desktop/HORSE/etiquetado_AS.csv" /
      --workers 4

  # Etiquetar PIMA (PIMA_1):
  cd "C:/Users/mcorn/Downloads"
  py etiquetar_dataset.py /
      --carpeta "C:/Users/mcorn/OneDrive/Desktop/HORSE/DATOS/PIMA_1" /
      --modelo  "C:/Users/mcorn/OneDrive/Desktop/HORSE/modelo_identificador_20260310_0931.pkl" /
      --salida  "C:/Users/mcorn/OneDrive/Desktop/HORSE/etiquetado_PIMA.csv" /
      --workers 4

  # Solo medir (sin etiquetar, genera histogramas):
  py etiquetar_dataset.py /
      --carpeta "C:/Users/mcorn/OneDrive/Desktop/HORSE/DATOS/PIMA_1" /
      --modelo  "C:/Users/mcorn/OneDrive/Desktop/HORSE/modelo_identificador_20260310_0931.pkl" /
      --salida  "C:/Users/mcorn/OneDrive/Desktop/HORSE/medicion_PIMA_v3.csv" /
      --solo_medir --workers 4

────────────────────────────────────────────────────────────────────────
ARCHIVOS REQUERIDOS EN LA MISMA CARPETA
────────────────────────────────────────────────────────────────────────
  etiquetar_dataset.py          — este script
  demm_core.py                  — validación (opcional, recomendado)

SALIDA CSV — columnas:
  nombre, ruta, pinon, conf_pinon,
  etiqueta_auto, etiqueta_final, zona_defecto,
  K_ret, CF_p99_ret, K_emp, CF_p99_emp,
  nivel_validacion, razones,
  rms_ret, rms_emp, kurt_max_ret, kurt_max_emp
"""

import os
import sys
import argparse
import warnings
import pickle
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")

# ── Importar funciones core del revisor (segmentación + validación completa) ──
try:
    import demm_core as _core
    detectar_separador             = _core.detectar_separador
    detectar_freno                 = _core.detectar_freno
    detectar_estabilizacion        = _core.detectar_estabilizacion_retroceso
    calcular_features              = _core.calcular_features
    validar_grabacion              = _core.validar_grabacion
    tramo_central                  = _core.tramo_central
    CORE_DISPONIBLE = True
    print("✓ demm_core.py cargado — validación y segmentación completa del revisor")
except ImportError:
    CORE_DISPONIBLE = False
    print("ADVERTENCIA: demm_core.py no encontrado — usando segmentación simplificada")
    print("  Coloca demm_core.py en la misma carpeta que este script")

# ── Parámetros del sistema ───────────────────────────────────────────────────
FS      = 48000
RPM     = 1135
F_ROT   = RPM / 60.0          # 18.917 Hz
T_ROT   = 1.0 / F_ROT         # 52.9 ms
M_ROT   = int(T_ROT * FS)     # 2537 muestras/giro

PINONES = {
    "PIMA":   {"dientes": 26},
    "ARB_14": {"dientes": 14},
    "ARB_15": {"dientes": 15},
}
for v in PINONES.values():
    v["gmf"] = v["dientes"] * F_ROT

# ── Umbrales de etiquetado por piñón (calibrados con datos reales) ──────────
#   PIMA (26d):  K=3.6, CF.p99=3.2  — calibrado con histograma PIMA
#   AS (14/15d): K=4.0, CF.p99=3.3  — calibrado con datos reales ARB_14
UMBRALES = {
    "PIMA":   {"K": 3.6, "CF": 3.2},
    "ARB_14": {"K": 5.0, "CF": 3.5},
    "ARB_15": {"K": 5.0, "CF": 3.5},
}
# Fallback si el piñón no está en el diccionario
K_UMBRAL      = 3.6
CF_P99_UMBRAL = 3.2

# ── Parámetros de segmentación ───────────────────────────────────────────────
VENTANA_ENV_LENTA = 0.03
MARGEN_BUSQUEDA   = 0.10
FACTOR_UMBRAL_SEP = 2.5
VENTANA_SEG       = 0.5
PASO_SEG          = 0.25
VENTANA_MUESTRAS  = int(VENTANA_SEG * FS)
PASO_MUESTRAS     = int(PASO_SEG * FS)
TOL_SB            = F_ROT * 0.6

# ── Validación ───────────────────────────────────────────────────────────────
VAL_RMS_EMP_CV_MIN = 0.08
VAL_KURT_GOLPE     = 12.0
VAL_DUR_RET_MIN    = 0.30
VAL_DUR_TOTAL_MIN  = 1.80
VAL_CLIP_PCT       = 0.5
VAL_SCORE_VALIDO   = 70
VAL_SCORE_DUDOSO   = 40

# ── Modelo ML de identificación de piñón ─────────────────────────────────────
_modelo_pinon = None   # se carga bajo demanda

def cargar_modelo_pinon(ruta_pkl):
    """Carga el modelo identificador desde un .pkl."""
    global _modelo_pinon
    with open(ruta_pkl, "rb") as f:
        _modelo_pinon = pickle.load(f)
    clases = _modelo_pinon.get("clases", [])
    print(f"  Modelo cargado: clases={clases}  "
          f"features={len(_modelo_pinon.get('feature_cols',[]))}")

def _extraer_features_id(señal):
    """
    Features para el clasificador de piñón — misma función que probar_identificador.py.
    Usa el tramo central de 1s para ser robusto a segmentación.
    """
    n = len(señal)
    centro = n // 2
    mitad  = min(int(FS * 0.5), n // 2)
    seg    = señal[centro - mitad : centro + mitad].astype(np.float64)
    ns     = len(seg)

    rms  = float(np.sqrt(np.mean(seg**2)))
    mu   = np.mean(seg)
    kurt = float(np.mean((seg-mu)**4) / (np.mean((seg-mu)**2)**2 + 1e-12))
    v50  = int(FS * 0.050)
    rms_v = [np.sqrt(np.mean(seg[i:i+v50]**2))
             for i in range(0, ns-v50, v50//2)]
    cv_rms = float(np.std(rms_v) / (np.mean(rms_v) + 1e-12))

    ventana  = np.hanning(ns)
    fft_mag  = np.abs(np.fft.rfft(seg * ventana)) * 2 / ns
    freqs    = np.fft.rfftfreq(ns, d=1.0/FS)
    fft_db   = 20 * np.log10(fft_mag + 1e-12)
    wl = min(51, len(fft_db)-1); wl = wl if wl%2==1 else wl-1
    fft_suav = savgol_filter(fft_db, window_length=max(wl,3), polyorder=5)
    diff     = fft_db - fft_suav
    above    = diff > 0
    dx       = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0

    def e_banda(fc, bw=12.0):
        mask = (freqs >= fc-bw) & (freqs <= fc+bw)
        return float(np.sum(fft_mag[mask]**2))

    gmf_pima  = PINONES["PIMA"]["gmf"]
    gmf_arb14 = PINONES["ARB_14"]["gmf"]
    gmf_arb15 = PINONES["ARB_15"]["gmf"]
    e_total   = float(np.sum(fft_mag**2)) + 1e-20

    mask_ruido = ((freqs > 50) & (freqs < 200)) | ((freqs > 350) & (freqs < 450))
    e_ruido    = float(np.sum(fft_mag[mask_ruido]**2))
    e_1gmf14   = e_banda(gmf_arb14)
    mask_a     = freqs <= 2000
    freq_centroide = float(
        np.sum(freqs[mask_a] * fft_mag[mask_a]) / (np.sum(fft_mag[mask_a]) + 1e-12))
    e_low  = float(np.sum(fft_mag[(freqs >= 100) & (freqs <= 400)]**2))
    e_high = float(np.sum(fft_mag[(freqs >= 400) & (freqs <= 1200)]**2))

    return {
        "rms": rms, "kurt": kurt, "cv_rms": cv_rms,
        "fondo_espectro":  float(np.sqrt(np.mean(diff**2))),
        "densidad_ruido":  float(np.sum(diff[above]) * dx),
        "n_frec_umbral":   int(np.sum(above)),
        "nivel_max":       float(np.max(diff)),
        "ratio_gmf":       e_1gmf14 / e_total,
        "ratio_2gmf":      e_banda(gmf_arb14*2) / (e_1gmf14 + 1e-20),
        "ratio_3gmf":      e_banda(gmf_arb14*3) / (e_1gmf14 + 1e-20),
        "ratio_ruido_gmf": e_ruido / (e_1gmf14 + 1e-20),
        "freq_centroide":  freq_centroide,
        "ratio_hilo":      e_high / (e_low + 1e-20),
        "e_gmf_pima":      e_banda(gmf_pima)   + e_banda(gmf_pima*2),
        "e_gmf_arb14":     e_banda(gmf_arb14)  + e_banda(gmf_arb14*2),
        "e_gmf_arb15":     e_banda(gmf_arb15)  + e_banda(gmf_arb15*2),
        "e_2gmf_pima":     e_banda(gmf_pima*2),
        "e_2gmf_arb14":    e_banda(gmf_arb14*2),
        "e_2gmf_arb15":    e_banda(gmf_arb15*2),
        "freq_pico1":      float(freqs[np.argmax(fft_mag)]),
    }

def detectar_pinon(señal):
    """
    Identifica el piñón usando el modelo ML si está cargado.
    Fallback: regla espectral simple si no hay modelo.
    """
    if _modelo_pinon is not None:
        try:
            feats = _extraer_features_id(señal)
            pipe  = _modelo_pinon["pipeline"]
            le    = _modelo_pinon["label_enc"]
            cols  = _modelo_pinon["feature_cols"]
            x     = np.array([[feats.get(c, 0.0) for c in cols]])
            y_enc = pipe.predict(x)[0]
            clase = le.inverse_transform([y_enc])[0]
            probs = pipe.predict_proba(x)[0]
            conf  = float(np.max(probs))
            # Normalizar: ARB_14_JR / ARB_14_JH → ARB_14
            if "ARB_14" in clase: return "ARB_14", conf
            if "ARB_15" in clase: return "ARB_15", conf
            return "PIMA", conf
        except Exception:
            pass

    # Fallback espectral — sin modelo
    n  = len(señal)
    seg = señal[n//4 : 3*n//4].astype(np.float64)
    ns  = len(seg)
    fft_mag = np.abs(np.fft.rfft(seg * np.hanning(ns))) * 2 / ns
    freqs   = np.fft.rfftfreq(ns, d=1.0/FS)
    def e_banda(fc, bw=15.0):
        m = (freqs >= fc-bw) & (freqs <= fc+bw)
        return float(np.sum(fft_mag[m]**2))
    e_pima  = e_banda(PINONES["PIMA"]["gmf"])
    e_arb14 = e_banda(PINONES["ARB_14"]["gmf"])
    e_arb15 = e_banda(PINONES["ARB_15"]["gmf"])
    mejor   = max([("PIMA", e_pima), ("ARB_14", e_arb14), ("ARB_15", e_arb15)],
                  key=lambda x: x[1])
    return mejor[0], 0.0   # confianza 0 = fallback

# ── Lógica de etiquetado ─────────────────────────────────────────────────────
def etiquetar(k_ret, cf_ret, k_emp, cf_emp, pinon="PIMA"):
    """
    Lógica AND por zona con umbrales específicos por piñón:
      PIMA   : K=3.6, CF.p99=3.2
      AS     : K=4.0, CF.p99=3.3  (ARB_14 / ARB_15)

      MALO    : (K_ret>K AND CF_ret>CF) OR (K_emp>K AND CF_emp>CF)
      BUENO   : todos los valores bajo umbral
      REVISAR : caso intermedio (un solo umbral superado en cualquier zona)
    """
    umb = UMBRALES.get(pinon, UMBRALES.get("PIMA"))
    k_u  = umb["K"]
    cf_u = umb["CF"]

    malo_ret = (k_ret > k_u) and (cf_ret > cf_u)
    malo_emp = (k_emp > k_u) and (cf_emp > cf_u)
    sosp_ret = (k_ret > k_u) != (cf_ret > cf_u)  # XOR
    sosp_emp = (k_emp > k_u) != (cf_emp > cf_u)

    if malo_ret or malo_emp:
        zona = []
        if malo_ret: zona.append("RET")
        if malo_emp: zona.append("EMP")
        return "MALO", "+".join(zona)
    if sosp_ret or sosp_emp:
        return "REVISAR", "—"
    return "BUENO", "—"

# ── Procesamiento de un archivo ───────────────────────────────────────────────
# ── K_global y CF.p99 sobre zona limpia ──────────────────────────────────────
def _kurt_global(z):
    z = z.astype(np.float64)
    mu = np.mean(z)
    return float(np.mean((z-mu)**4) / (np.mean((z-mu)**2)**2 + 1e-12))

def _cf_p99(z):
    """CF robusto: percentil 99.5 del valor absoluto / RMS."""
    z = z.astype(np.float64)
    rms = float(np.sqrt(np.mean(z**2)))
    return float(np.percentile(np.abs(z), 99.5)) / (rms + 1e-12)


# ── Segmentación embebida — idéntica al explorador_dataset ────────────────────
def _detectar_separador(señal):
    n = len(señal); s = señal.astype(np.float64)
    i0 = int(n * 0.15); i1 = int(n * 0.85)
    zona = s[i0:i1]; n_giros = len(zona) // M_ROT
    if n_giros < 4:
        env = np.sqrt(np.convolve(s**2, np.ones(max(1,int(0.20*FS)))/max(1,int(0.20*FS)), mode='same'))
        return int(np.argmax(env[i0:i1])) + i0
    rms_g = np.array([float(np.sqrt(np.mean(zona[i*M_ROT:(i+1)*M_ROT]**2))) for i in range(n_giros)])
    n_ref = max(2, n_giros // 3)
    nivel_ref = float(np.median(rms_g[:n_ref]))
    if nivel_ref < 1e-9: nivel_ref = float(np.median(rms_g)) + 1e-9
    for factor in [4.0, 3.0, 2.5]:
        for i, rms in enumerate(rms_g):
            if rms > nivel_ref * factor:
                return i0 + i * M_ROT
    return i0 + int(np.argmax(rms_g)) * M_ROT

def _detectar_freno(zona_ret):
    n = len(zona_ret)
    if n < int(FS * 0.15): return n
    s = zona_ret.astype(np.float64); n_giros = n // M_ROT
    if n_giros < 6: return int(n * 0.92)
    rms_g = np.array([float(np.sqrt(np.mean(s[i*M_ROT:(i+1)*M_ROT]**2))) for i in range(n_giros)])
    MIN_G = max(7, int(0.35 * FS / M_ROT))
    i1 = min(MIN_G + 5, n_giros - 1)
    if i1 <= MIN_G or MIN_G >= n_giros: return int(n * 0.92)
    nivel_ref = float(np.median(rms_g[MIN_G:i1]))
    if nivel_ref < 1e-9: return int(n * 0.92)
    for i in range(MIN_G, n_giros):
        if rms_g[i] > nivel_ref * 2.5:
            return max(0, i * M_ROT - M_ROT // 2)
    return int(n * 0.92)

def _detectar_estabilizacion(zona_ret):
    s = zona_ret.astype(np.float64); n = len(s); n_giros = n // M_ROT
    if n_giros < 3: return 0
    rms_g = np.array([float(np.sqrt(np.mean(s[i*M_ROT:(i+1)*M_ROT]**2))) for i in range(n_giros)])
    segunda = rms_g[n_giros // 2:]
    nivel_ref = float(np.percentile(segunda, 20))
    if nivel_ref < 1e-9: nivel_ref = float(np.median(rms_g))
    if nivel_ref < 1e-9: return 0
    umbral_min = nivel_ref * 0.35; umbral_max = nivel_ref * 1.80
    CONFIRM = 2; GIROS_MINIMOS = 5
    inicio = min(GIROS_MINIMOS, n_giros - CONFIRM - 1)
    for i in range(inicio, n_giros - CONFIRM + 1):
        ventana = rms_g[i:i+CONFIRM]
        if not all(umbral_min <= r <= umbral_max for r in ventana): continue
        if np.max(ventana) / (np.min(ventana) + 1e-12) > 1.8: continue
        return max(0, i * M_ROT)
    return min(GIROS_MINIMOS * M_ROT, n // 3)

def _kurt_por_giro(z):
    """Kurtosis = media de kurtosis por giro — idéntico al explorador."""
    s = z.astype(np.float64); n_giros = len(s) // M_ROT
    if n_giros < 1:
        mu = np.mean(s)
        return float(np.mean((s-mu)**4) / (np.mean((s-mu)**2)**2 + 1e-12))
    vals = []
    for i in range(n_giros):
        g = s[i*M_ROT:(i+1)*M_ROT]; mu = np.mean(g)
        vals.append(float(np.mean((g-mu)**4) / (np.mean((g-mu)**2)**2 + 1e-12)))
    return float(np.mean(vals))

# ── Procesamiento de un archivo ───────────────────────────────────────────────
def procesar_archivo(ruta):
    nombre = os.path.basename(ruta)
    pinon  = "?"

    try:
        df    = pd.read_csv(ruta)
        señal = df["senal"].values.astype(np.float32)
        n     = len(señal)
        if n < int(FS * 1.0):
            return _fila_error(ruta, nombre, "?", "Archivo demasiado corto")

        # ── Paso 1: identificar piñón con ML ──────────────────────────────────
        pinon, conf_pinon = detectar_pinon(señal)
        gmf       = PINONES[pinon]["gmf"]
        n_dientes = PINONES[pinon]["dientes"]

        # ── Paso 2: segmentación embebida (idéntica al explorador) ─────────
        margen     = int(FS * 0.02)
        idx_sep    = _detectar_separador(señal)
        i_emp_i    = int(n * 0.05)
        i_emp_f    = max(0, idx_sep - margen)
        i_ret_i    = min(n, idx_sep + margen)
        zona_ret_prel  = señal[i_ret_i:int(n * 0.95)]
        idx_freno      = _detectar_freno(zona_ret_prel)
        i_ret_f        = i_ret_i + idx_freno
        zona_ret_bruta = señal[i_ret_i:i_ret_f]
        idx_estab      = _detectar_estabilizacion(zona_ret_bruta)
        i_ret_estab    = i_ret_i + idx_estab

        zona_emp = señal[i_emp_i:i_emp_f].astype(np.float64)
        zona_ret = zona_ret_bruta[idx_estab:].astype(np.float64)

        # Validación con demm_core si está disponible
        if CORE_DISPONIBLE:
            res_core    = _core.analizar_archivo(ruta, gmf, n_dientes)
            if res_core is not None:
                val         = _core.validar_grabacion(res_core)
                nivel_val   = val["nivel"]
                razones_val = val.get("razones", [])
            else:
                dur_ret   = len(zona_ret) / FS
                nivel_val = "VÁLIDO" if dur_ret >= 0.30 else "DUDOSO"
                razones_val = []
        else:
            dur_ret   = len(zona_ret) / FS
            nivel_val = "VÁLIDO" if dur_ret >= 0.30 else "DUDOSO"
            razones_val = [] if nivel_val == "VÁLIDO" else ["Retroceso corto"]

        # ── Paso 3: early-exit si es INVÁLIDO ────────────────────────────────
        if nivel_val == "INVÁLIDO":
            return {
                "nombre": nombre, "ruta": ruta, "pinon": pinon,
                "conf_pinon": round(conf_pinon, 3),
                "K_ret": None, "CF_p99_ret": None,
                "K_emp": None, "CF_p99_emp": None,
                "etiqueta_auto": "INVÁLIDO", "zona_defecto": "—",
                "revisado": False, "etiqueta_final": "INVÁLIDO",
                "nivel_validacion": "INVÁLIDO",
                "razones": " | ".join(razones_val) if razones_val else "—",
                "rms_ret": None, "rms_emp": None,
                "kurt_max_ret": None, "kurt_max_emp": None,
            }

        # ── Paso 4: K y CF.p99 — mismo método que el explorador ─────────────
        # Excluir último giro del retroceso (transición al freno)
        # Excluir últimos 2 giros del empuje (transición al separador)
        zona_ret_limpia = zona_ret[:-M_ROT]   if len(zona_ret) > M_ROT*3   else zona_ret
        zona_emp_util   = zona_emp[:-2*M_ROT] if len(zona_emp) > 2*M_ROT*2 else zona_emp

        # K global método Cycla — kurtosis de toda la zona limpia de una vez
        # Idéntico a kurt_global_ret / kurt_global_emp del explorador_dataset
        k_ret  = _kurt_global(zona_ret_limpia) if len(zona_ret_limpia) > 10 else 3.0
        k_emp  = _kurt_global(zona_emp_util)   if len(zona_emp_util)   > 10 else 3.0
        # CF.p99 — percentil 99.5 / RMS
        cf_ret = _cf_p99(zona_ret_limpia)      if len(zona_ret_limpia) > 10 else 0.0
        cf_emp = _cf_p99(zona_emp_util)        if len(zona_emp_util)   > 10 else 0.0

        rms_ret = float(np.sqrt(np.mean(zona_ret**2))) if len(zona_ret) > 0 else 0.0
        rms_emp = float(np.sqrt(np.mean(zona_emp**2))) if len(zona_emp) > 0 else 0.0

        # ── Paso 5: etiqueta automática ───────────────────────────────────────
        etiqueta, zona_def = etiquetar(k_ret, cf_ret, k_emp, cf_emp, pinon=pinon)

        # Grabación DUDOSA pero diagnóstico BUENO → revisión manual obligatoria
        if nivel_val == "DUDOSO" and etiqueta == "BUENO":
            etiqueta = "REVISAR"

        return {
            "nombre":           nombre,
            "ruta":             ruta,
            "pinon":            pinon,
            "conf_pinon":       round(conf_pinon, 3),
            "K_ret":            round(k_ret,  3),
            "CF_p99_ret":       round(cf_ret, 3),
            "K_emp":            round(k_emp,  3),
            "CF_p99_emp":       round(cf_emp, 3),
            "etiqueta_auto":    etiqueta,
            "zona_defecto":     zona_def,
            "revisado":         False,
            "etiqueta_final":   etiqueta if etiqueta != "REVISAR" else "",
            "nivel_validacion": nivel_val,
            "razones":          " | ".join(razones_val) if razones_val else "—",
            "rms_ret":          round(rms_ret, 5),
            "rms_emp":          round(rms_emp, 5),
            "kurt_max_ret":     round(k_ret, 2),
            "kurt_max_emp":     round(k_emp, 2),
        }

    except Exception as ex:
        return _fila_error(ruta, nombre, pinon, str(ex))

def _fila_error(ruta, nombre, pinon, msg):
    return {
        "nombre": nombre, "ruta": ruta, "pinon": pinon, "conf_pinon": 0.0,
        "K_ret": None, "CF_p99_ret": None,
        "K_emp": None, "CF_p99_emp": None,
        "etiqueta_auto": "ERROR", "zona_defecto": "—",
        "revisado": False, "etiqueta_final": "",
        "nivel_validacion": "ERROR",
        "razones": msg,
        "rms_ret": None, "rms_emp": None,
        "kurt_max_ret": None, "kurt_max_emp": None,
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def _resumen_distribucion(df, salida_base):
    """
    Imprime estadísticas de K y CF.p99 por piñón y genera un PNG con
    histogramas para calibrar umbrales visualmente.
    Solo se llama en modo --solo_medir.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    validos = df[df["nivel_validacion"] != "INVÁLIDO"].copy()
    pinones = sorted(validos["pinon"].dropna().unique())

    print("\n" + "=" * 70)
    print("DISTRIBUCIÓN K y CF.p99 — solo archivos válidos")
    print("=" * 70)

    for p in pinones:
        sub = validos[validos["pinon"] == p]
        n   = len(sub)
        if n == 0:
            continue
        print(f"\n  {p}  ({n} archivos válidos)")
        for col, nombre in [("K_ret","K retroceso"), ("CF_p99_ret","CF.p99 retroceso"),
                             ("K_emp","K empuje"),    ("CF_p99_emp","CF.p99 empuje")]:
            vals = sub[col].dropna()
            if len(vals) == 0:
                continue
            p10, p25, p50, p75, p90, p95, p99 = (
                vals.quantile([0.10,0.25,0.50,0.75,0.90,0.95,0.99]).values)
            print(f"    {nombre:<22}  "
                  f"p10={p10:5.2f}  p50={p50:5.2f}  p75={p75:5.2f}  "
                  f"p90={p90:5.2f}  p95={p95:5.2f}  p99={p99:5.2f}  "
                  f"max={vals.max():5.2f}")

    print("\n  → Busca el 'valle' entre p75 y p95 de cada piñón para fijar el umbral")
    print("  → El umbral ideal separa los dos grupos visibles en el histograma")

    # ── Histogramas ──────────────────────────────────────────────────────────
    n_pinones = len(pinones)
    if n_pinones == 0:
        return

    fig, axes = plt.subplots(n_pinones, 2, figsize=(12, 4*n_pinones),
                              facecolor="#0e0f11")
    if n_pinones == 1:
        axes = [axes]

    colores = {"PIMA": "#f59e0b", "ARB_14": "#22c55e", "ARB_15": "#a855f7"}

    for row, p in enumerate(pinones):
        sub  = validos[validos["pinon"] == p]
        col  = colores.get(p, "#4f8ef7")

        for col_idx, (feat, label, umbral_actual) in enumerate([
            ("K_ret",     f"K retroceso — {p}", UMBRALES.get(p, UMBRALES["PIMA"])["K"]),
            ("CF_p99_ret",f"CF.p99 retroceso — {p}", UMBRALES.get(p, UMBRALES["PIMA"])["CF"]),
        ]):
            ax = axes[row][col_idx]
            ax.set_facecolor("#13141a")
            for sp in ax.spines.values(): sp.set_color("#252730")
            ax.tick_params(colors="#6b7280", labelsize=8)

            vals = sub[feat].dropna().values
            if len(vals) == 0:
                ax.text(0.5, 0.5, "sin datos", color="#6b7280",
                        ha="center", va="center", transform=ax.transAxes)
                continue

            ax.hist(vals, bins=40, color=col, alpha=0.75, edgecolor="#0e0f11")
            ax.axvline(umbral_actual, color="#ef4444", linewidth=1.5,
                       linestyle="--", label=f"Umbral actual ({umbral_actual})")

            # Percentiles de referencia
            for q, qv in [(0.75, sub[feat].quantile(0.75)),
                          (0.90, sub[feat].quantile(0.90)),
                          (0.95, sub[feat].quantile(0.95))]:
                ax.axvline(qv, color="#6b7280", linewidth=0.8,
                           linestyle=":", alpha=0.7, label=f"p{int(q*100)}={qv:.2f}")

            ax.set_title(label, color="#e8eaf2", fontsize=9)
            ax.set_xlabel("Valor", color="#6b7280", fontsize=8)
            ax.set_ylabel("N archivos", color="#6b7280", fontsize=8)
            ax.legend(fontsize=6.5, facecolor="#13141a",
                      edgecolor="#252730", labelcolor="#e8eaf2")

    fig.suptitle("Distribución K y CF.p99 por piñón — calibración de umbrales",
                 color="#e8eaf2", fontsize=11, y=1.01)
    fig.tight_layout()

    png_path = salida_base.replace(".csv", "_distribucion.png")
    fig.savefig(png_path, dpi=130, bbox_inches="tight",
                facecolor="#0e0f11")
    plt.close(fig)
    print(f"\n  Histograma guardado en: {png_path}")
    print("  Abre la imagen para elegir los umbrales de ARB_14/ARB_15")


def main():
    parser = argparse.ArgumentParser(description="Etiquetado semi-automático DEMM")
    parser.add_argument("--carpeta", required=True,
                        help="Carpeta raíz con los CSV de grabaciones")
    parser.add_argument("--modelo",  default=None,
                        help="Ruta al modelo_identificador.pkl (recomendado)")
    parser.add_argument("--salida",  default="dataset_etiquetado.csv",
                        help="Nombre del CSV de salida (default: dataset_etiquetado.csv)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Hilos paralelos (default: 4)")
    parser.add_argument("--pinon",   default=None,
                        choices=["PIMA","ARB_14","ARB_15"],
                        help="Filtrar solo un tipo de piñón (opcional)")
    parser.add_argument("--solo_medir", action="store_true",
                        help="Solo calcula K y CF.p99 — no asigna etiquetas. "
                             "Útil para calibrar umbrales de nuevos piñones.")
    args = parser.parse_args()

    # Cargar modelo de identificación de piñón
    if args.modelo:
        if os.path.exists(args.modelo):
            print(f"Cargando modelo de piñón: {args.modelo}")
            cargar_modelo_pinon(args.modelo)
        else:
            print(f"ADVERTENCIA: modelo no encontrado en {args.modelo} — usando fallback espectral")
    else:
        print("ADVERTENCIA: --modelo no especificado — usando fallback espectral")
        print("  Recomendado: --modelo modelo_identificador.pkl")

    if args.solo_medir:
        print("\nMODO: solo medir — K y CF.p99 sin aplicar umbrales")
        print("  Objetivo: calibrar umbrales para ARB_14/ARB_15")
        print("  Resultado: CSV con valores + histograma PNG\n")

    # Buscar todos los CSV recursivamente
    archivos = []
    for dirpath, _, filenames in os.walk(args.carpeta):
        for f in filenames:
            if f.lower().endswith(".csv"):
                ruta = os.path.join(dirpath, f)
                archivos.append(ruta)

    if not archivos:
        print(f"No se encontraron archivos CSV en: {args.carpeta}")
        sys.exit(1)

    # Filtro por piñón — necesita señal, se aplica después de procesar
    # (el nombre no tiene info del piñón, lo detecta el ML)
    print(f"Archivos encontrados: {len(archivos)}")
    if not args.solo_medir:
        print(f"Umbrales por piñón:")
        for p, u in UMBRALES.items():
            print(f"  {p:<10} K > {u['K']}  AND  CF.p99 > {u['CF']}  → MALO")
    print(f"Workers:  {args.workers}")
    print("-" * 60)

    resultados = []
    completados = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futuros = {ex.submit(procesar_archivo, r): r for r in archivos}
        for fut in as_completed(futuros):
            completados += 1
            res = fut.result()

            # Filtro por piñón si se especificó --pinon
            if args.pinon and res.get("pinon") != args.pinon:
                continue

            # En modo solo_medir: sobrescribir etiqueta con "MEDIR"
            if args.solo_medir and res["etiqueta_auto"] not in ("INVÁLIDO","ERROR"):
                res["etiqueta_auto"]  = "MEDIR"
                res["etiqueta_final"] = ""

            resultados.append(res)
            etq   = res["etiqueta_auto"]
            sym   = {"BUENO":"✓","MALO":"✗","REVISAR":"~","INVÁLIDO":"✖",
                     "ERROR":"!","MEDIR":"?"}.get(etq,"?")
            conf_s = f"{res['conf_pinon']*100:.0f}%" if res.get('conf_pinon') else "—"
            print(f"  [{completados:4d}/{len(archivos)}] {sym} {res['nombre'][:38]:<38}  "
                  f"piñon={res['pinon']:<7}({conf_s})  "
                  f"ret K={str(res['K_ret'] or '—'):>6}  CF={str(res['CF_p99_ret'] or '—'):>5}  "
                  f"emp K={str(res['K_emp'] or '—'):>6}  CF={str(res['CF_p99_emp'] or '—'):>5}  "
                  f"[{res.get('zona_defecto','—')}]")

    if not resultados:
        print("No quedaron archivos tras el filtro de piñón.")
        sys.exit(1)

    # Guardar CSV
    df = pd.DataFrame(resultados)
    cols = ["nombre","pinon","conf_pinon","etiqueta_auto","etiqueta_final","zona_defecto",
            "K_ret","CF_p99_ret","K_emp","CF_p99_emp",
            "nivel_validacion","revisado","razones",
            "rms_ret","rms_emp","kurt_max_ret","kurt_max_emp","ruta"]
    df = df[[c for c in cols if c in df.columns]]
    df.sort_values(["pinon","K_ret"], ascending=[True, False], inplace=True)
    df.to_csv(args.salida, index=False, encoding="utf-8")

    # ── Modo solo_medir: mostrar distribución + histograma ───────────────────
    if args.solo_medir:
        _resumen_distribucion(df, args.salida)
        print(f"\nCSV guardado en: {args.salida}")
        print("\nPRÓXIMOS PASOS:")
        print("  1. Abre el PNG para ver la distribución de K y CF.p99 por piñón")
        print("  2. Abre el CSV y ordena por K_ret — revisa con el explorador")
        print("     los archivos con valores más altos para confirmar si son defecto")
        print("  3. Fija los nuevos umbrales en etiquetar_dataset.py:")
        print("     K_UMBRAL_ARB14, CF_P99_UMBRAL_ARB14, etc.")
        print("  4. Vuelve a correr sin --solo_medir para el etiquetado final")
        return

    # ── Modo normal: resumen de etiquetas ────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    for etq in ["BUENO","MALO","REVISAR","INVÁLIDO","ERROR"]:
        n = len(df[df["etiqueta_auto"] == etq])
        pct = n / len(df) * 100
        sym = {"BUENO":"✓","MALO":"✗","REVISAR":"~","INVÁLIDO":"✖","ERROR":"!"}.get(etq,"?")
        print(f"  {sym} {etq:<10} {n:>5}  ({pct:5.1f}%)")
    print(f"\n  TOTAL      {len(df):>5}")
    print(f"\n  → Para revisión manual: {len(df[df['etiqueta_auto']=='REVISAR'])} archivos")
    print(f"\nCSV guardado en: {args.salida}")

    # Generar lista de REVISAR para el explorador
    revisar = df[df["etiqueta_auto"] == "REVISAR"]["ruta"].tolist()
    if revisar:
        lista_path = args.salida.replace(".csv", "_REVISAR.txt")
        with open(lista_path, "w", encoding="utf-8") as f:
            f.write("\n".join(revisar))
        print(f"Lista REVISAR guardada en: {lista_path}")

if __name__ == "__main__":
    main()
