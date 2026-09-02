"""
calcular_umbrales_espectro.py
=============================
Calcula los umbrales espectrales estadísticos (μ + 3σ) orden a orden
a partir de todos los archivos válidos de un dataset etiquetado.

FLUJO:
  1. El usuario selecciona el CSV del dataset etiquetado (etiquetado_AS.csv,
     etiquetado_PIMA.csv, etc.) que debe tener columnas: ruta, etiqueta, pinon
  2. Se filtran solo los archivos con etiqueta BUENO, MALO o REVISAR
     (se excluyen INVÁLIDO y errores de lectura)
  3. Para cada archivo se calcula el espectro FFT del retroceso y del empuje
     sobre el tramo central de 0.5s (idéntico al explorador_señal)
  4. Todos los espectros se interpolan a una grilla común de N_GRID órdenes
  5. Se calcula media y sigma columna a columna (orden a orden)
  6. Se guarda el resultado en umbrales_<PINON>.pkl listo para el explorador

USO:
  py calcular_umbrales_espectro.py

REQUISITOS:
  pip install numpy pandas scipy
"""

import os
import sys
import pickle
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# ── Parámetros físicos (idénticos al explorador) ────────────────────────────
FS       = 48000
RPM      = 872.0
F_ROT    = RPM / 60.0
T_ROT    = 1.0 / F_ROT
M_ROT    = int(T_ROT * FS)

PINONES = {
    "PIMA":         {"dientes": 26, "gmf": 26 * F_ROT},
    "ARBOL_SEC_14": {"dientes": 14, "gmf": 14 * F_ROT},
    "ARBOL_SEC_15": {"dientes": 15, "gmf": 15 * F_ROT},
}

TRAMO_SEG      = 0.5
TRAMO_MUESTRAS = int(TRAMO_SEG * FS)
VENTANA_MUESTRAS = int(0.10 * FS)

# Grilla común de órdenes para interpolación
N_GRID = 512

# ── Paleta ───────────────────────────────────────────────────────────────────
C_BG      = "#0e0f11"
C_SURFACE = "#161820"
C_BORDER  = "#2a2d38"
C_TEXT    = "#e2e4ed"
C_SUB     = "#7a7f96"
C_ACENTO  = "#4f8ef7"
C_BUENO   = "#22c55e"
C_MALO    = "#ef4444"
C_REVISAR = "#f59e0b"
C_MONO    = "Consolas"

# ════════════════════════════════════════════════════════════════════════════
# SEGMENTACIÓN (idéntica al explorador_señal)
# ════════════════════════════════════════════════════════════════════════════

def _detectar_separador(señal):
    n = len(señal); s = señal.astype(np.float64)
    i0 = int(n * 0.15); i1 = int(n * 0.85)
    zona = s[i0:i1]; ng = len(zona) // M_ROT
    if ng < 4:
        v = max(1, int(0.20 * FS))
        env = np.sqrt(np.convolve(s**2, np.ones(v)/v, mode="same"))
        return int(np.argmax(env[i0:i1])) + i0
    rg  = np.array([float(np.sqrt(np.mean(zona[i*M_ROT:(i+1)*M_ROT]**2)))
                    for i in range(ng)])
    ref = float(np.median(rg[:max(2, ng//3)]))
    ref = ref if ref > 1e-9 else float(np.median(rg)) + 1e-9
    for f in [4.0, 3.0, 2.5]:
        for i, r in enumerate(rg):
            if r > ref * f:
                return i0 + i * M_ROT
    return i0 + int(np.argmax(rg)) * M_ROT


def _detectar_freno(zona_ret):
    n = len(zona_ret)
    if n < int(FS * 0.15): return n
    s = zona_ret.astype(np.float64); ng = n // M_ROT
    if ng < 6: return int(n * 0.92)
    rg  = np.array([float(np.sqrt(np.mean(s[i*M_ROT:(i+1)*M_ROT]**2)))
                    for i in range(ng)])
    MG  = max(7, int(0.35 * FS / M_ROT))
    i1  = min(MG + 5, ng - 1)
    if i1 <= MG or MG >= ng: return int(n * 0.92)
    ref = float(np.median(rg[MG:i1]))
    if ref < 1e-9: return int(n * 0.92)
    for i in range(MG, ng):
        if rg[i] > ref * 2.5:
            return max(0, i * M_ROT - M_ROT // 2)
    return int(n * 0.92)


def _detectar_estabilizacion(zona_ret):
    s = zona_ret.astype(np.float64); n = len(s); ng = n // M_ROT
    if ng < 3: return 0
    rg  = np.array([float(np.sqrt(np.mean(s[i*M_ROT:(i+1)*M_ROT]**2)))
                    for i in range(ng)])
    ref = float(np.percentile(rg[ng//2:], 20))
    if ref < 1e-9: ref = float(np.median(rg))
    if ref < 1e-9: return 0
    mn = ref * 0.35; mx = ref * 1.80; GM = 5; CN = 2
    ini = min(GM, ng - CN - 1)
    for i in range(ini, ng - CN + 1):
        v = rg[i:i+CN]
        if not all(mn <= r <= mx for r in v): continue
        if np.max(v) / (np.min(v) + 1e-12) > 1.8: continue
        return max(0, i * M_ROT)
    return min(GM * M_ROT, n // 3)


def _tramo_central(zona):
    n = len(zona)
    if n <= TRAMO_MUESTRAS: return zona
    ini = max(0, n // 2 - TRAMO_MUESTRAS // 2)
    return zona[ini: ini + TRAMO_MUESTRAS]


def calcular_espectro(zona, gmf):
    """FFT en órdenes — idéntica al explorador_señal."""
    s = zona.astype(np.float64); n = len(s)
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
    return ordenes, fft_db, fft_suav


def extraer_espectros(ruta_csv, gmf):
    """
    Lee un CSV de engrane, segmenta empuje y retroceso, y devuelve
    (ordenes_ret, fft_db_ret, ordenes_emp, fft_db_emp) o None si falla.
    """
    try:
        df = pd.read_csv(ruta_csv)
        col = "senal" if "senal" in df.columns else df.columns[1]
        señal = df[col].to_numpy(dtype=np.float32)
        n = len(señal)
        if n < int(FS * 1.5):
            return None

        margen   = int(FS * 0.02)
        idx_sep  = _detectar_separador(señal)

        # Empuje
        i_emp_i = int(n * 0.05)
        i_emp_f = max(0, idx_sep - margen)
        zona_emp = señal[i_emp_i:i_emp_f]
        recorte  = 2 * M_ROT
        zona_emp_util = zona_emp[:-recorte] if len(zona_emp) > recorte * 2 else zona_emp

        # Retroceso
        i_ret_i  = min(n, idx_sep + margen)
        zona_rp  = señal[i_ret_i: int(n * 0.95)]
        xf       = _detectar_freno(zona_rp)
        zona_ret = zona_rp[:xf]
        xe       = _detectar_estabilizacion(zona_ret)
        zona_ret = zona_ret[xe:]

        if len(zona_ret) < VENTANA_MUESTRAS or len(zona_emp_util) < VENTANA_MUESTRAS:
            return None

        tramo_ret = _tramo_central(zona_ret)
        tramo_emp = _tramo_central(zona_emp_util)

        ords_r, db_r, _ = calcular_espectro(tramo_ret, gmf)
        ords_e, db_e, _ = calcular_espectro(tramo_emp, gmf)

        return ords_r, db_r, ords_e, db_e

    except Exception:
        return None


# ════════════════════════════════════════════════════════════════════════════
# CÁLCULO DE UMBRALES
# ════════════════════════════════════════════════════════════════════════════

def calcular_umbrales(filas_ret, filas_emp, orden_gmf):
    """
    Recibe listas de arrays fft_db ya interpolados a la grilla común.
    Devuelve dict con toda la info necesaria para el explorador.
    """
    def _stats(filas):
        if len(filas) < 2:
            return None
        mat    = np.array(filas)          # (N_archivos × N_GRID)
        media  = np.mean(mat, axis=0)
        sigma  = np.std(mat,  axis=0)
        umbral = media + 3.0 * sigma
        return {"media": media, "sigma": sigma, "umbral": umbral,
                "n_archivos": len(filas)}

    orden_gmf_val = orden_gmf
    o_max  = orden_gmf_val * 3.5
    grilla = np.linspace(0.1, o_max, N_GRID)

    stats_ret = _stats(filas_ret)
    stats_emp = _stats(filas_emp)

    return {
        "ordenes":   grilla,
        "orden_gmf": orden_gmf_val,
        "o_max":     o_max,
        "ret":       stats_ret,
        "emp":       stats_emp,
    }


# ════════════════════════════════════════════════════════════════════════════
# INTERFAZ GRÁFICA
# ════════════════════════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Calcular Umbrales Espectrales — DEMM")
        self.geometry("680x560")
        self.resizable(False, False)
        self.configure(bg=C_BG)
        self._build()

    # ── UI ──────────────────────────────────────────────────────────────────
    def _build(self):
        # Header
        hdr = tk.Frame(self, bg=C_SURFACE, height=56)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Frame(hdr, bg=C_ACENTO, width=4).pack(side="left", fill="y")
        fh = tk.Frame(hdr, bg=C_SURFACE); fh.pack(side="left", padx=14)
        tk.Label(fh, text="UMBRALES ESPECTRALES  —  DEMM",
                 bg=C_SURFACE, fg=C_TEXT,
                 font=("Arial", 12, "bold")).pack(anchor="w")
        tk.Label(fh, text="Calcula μ+3σ orden a orden sobre el dataset completo",
                 bg=C_SURFACE, fg=C_SUB, font=("Arial", 9)).pack(anchor="w")
        tk.Frame(self, bg=C_BORDER, height=1).pack(fill="x")

        # Cuerpo
        body = tk.Frame(self, bg=C_BG, padx=28, pady=20)
        body.pack(fill="both", expand=True)

        def sec(txt):
            tk.Label(body, text=txt, bg=C_BG, fg=C_ACENTO,
                     font=("Arial", 9, "bold")).pack(anchor="w", pady=(14, 0))
            tk.Frame(body, bg=C_ACENTO, height=1).pack(fill="x", pady=(2, 8))

        def fila_ruta(lbl_txt, var, cmd):
            tk.Label(body, text=lbl_txt, bg=C_BG, fg=C_TEXT,
                     font=("Arial", 9)).pack(anchor="w")
            f = tk.Frame(body, bg=C_BG); f.pack(fill="x", pady=(2, 6))
            tk.Entry(f, textvariable=var, bg="#1e2028", fg=C_SUB,
                     font=("Arial", 9), relief="flat", bd=4,
                     state="readonly").pack(side="left", fill="x",
                                            expand=True, ipady=4)
            tk.Button(f, text="…", command=cmd, bg="#1e2028", fg=C_ACENTO,
                      relief="flat", bd=0, font=(C_MONO, 9),
                      cursor="hand2", width=3
                      ).pack(side="left", padx=(4, 0), ipady=4)

        # ── Entrada: CSV dataset ────────────────────────────────────────
        sec("DATASET ETIQUETADO")
        tk.Label(body,
                 text="CSV con columnas: ruta, etiqueta_final, pinon, nivel_validacion\n"
                      "Se usarán: etiqueta_final=BUENO/MALO  Y  nivel_validacion=VÁLIDO",
                 bg=C_BG, fg=C_SUB, font=("Arial", 8)).pack(anchor="w", pady=(0, 4))
        self.var_csv = tk.StringVar()
        fila_ruta("Archivo CSV del dataset:", self.var_csv, self._sel_csv)

        # ── Salida: carpeta y nombre ────────────────────────────────────
        sec("SALIDA")
        tk.Label(body,
                 text="Se generará un archivo por piñón detectado en el dataset\n"
                      "Ejemplo: umbrales_ARBOL_SEC_14.pkl, umbrales_PIMA.pkl",
                 bg=C_BG, fg=C_SUB, font=("Arial", 8)).pack(anchor="w", pady=(0, 4))
        self.var_salida = tk.StringVar(
            value=os.path.join(os.path.expanduser("~"),
                               "OneDrive", "Desktop", "HORSE"))
        fila_ruta("Carpeta de salida:", self.var_salida, self._sel_salida)

        # ── Progreso ────────────────────────────────────────────────────
        sec("PROGRESO")
        self.lbl_estado = tk.Label(body, text="Listo para calcular.",
                                    bg=C_BG, fg=C_SUB,
                                    font=(C_MONO, 8), anchor="w")
        self.lbl_estado.pack(fill="x")

        self.prog = ttk.Progressbar(body, orient="horizontal",
                                     mode="determinate", length=600)
        self.prog.pack(fill="x", pady=(6, 0))

        style = ttk.Style(); style.theme_use("default")
        style.configure("Custom.Horizontal.TProgressbar",
                         troughcolor=C_SURFACE, background=C_ACENTO,
                         bordercolor=C_BORDER, thickness=14)
        self.prog.configure(style="Custom.Horizontal.TProgressbar")

        self.lbl_resumen = tk.Label(body, text="",
                                     bg=C_BG, fg=C_TEXT,
                                     font=(C_MONO, 8), justify="left")
        self.lbl_resumen.pack(anchor="w", pady=(8, 0))

        # ── Botón ───────────────────────────────────────────────────────
        tk.Frame(body, bg=C_BORDER, height=1).pack(fill="x", pady=(16, 10))
        self.btn = tk.Button(body, text="▶  CALCULAR UMBRALES",
                              command=self._iniciar,
                              bg=C_ACENTO, fg="white",
                              activebackground="#3a7fe0",
                              activeforeground="white",
                              relief="flat", bd=0,
                              font=("Arial", 11, "bold"),
                              cursor="hand2", pady=12)
        self.btn.pack(fill="x")

    # ── Selectores ──────────────────────────────────────────────────────────
    def _sel_csv(self):
        r = filedialog.askopenfilename(
            title="Dataset etiquetado",
            filetypes=[("CSV", "*.csv"), ("Todos", "*.*")])
        if r: self.var_csv.set(r)

    def _sel_salida(self):
        d = filedialog.askdirectory(title="Carpeta de salida")
        if d: self.var_salida.set(d)

    # ── Lógica principal ────────────────────────────────────────────────────
    def _iniciar(self):
        ruta_csv = self.var_csv.get().strip()
        if not ruta_csv or not os.path.isfile(ruta_csv):
            messagebox.showwarning("Sin dataset",
                                   "Selecciona el CSV del dataset primero.")
            return
        carpeta_sal = self.var_salida.get().strip()
        if not carpeta_sal:
            messagebox.showwarning("Sin carpeta", "Selecciona la carpeta de salida.")
            return

        self.btn.config(state="disabled")
        self.lbl_resumen.config(text="")
        threading.Thread(target=self._procesar,
                          args=(ruta_csv, carpeta_sal), daemon=True).start()

    def _st(self, txt, color=None):
        """Actualiza el label de estado desde cualquier hilo."""
        self.after(0, lambda: self.lbl_estado.config(
            text=txt, fg=color or C_SUB))

    def _prog(self, valor, maximo):
        self.after(0, lambda: self.prog.config(value=valor, maximum=maximo))

    def _procesar(self, ruta_csv, carpeta_sal):
        try:
            # ── Leer dataset ────────────────────────────────────────────
            self._st("Leyendo dataset…")
            df = pd.read_csv(ruta_csv)

            # Normalizar nombres de columnas
            df.columns = [c.strip().lower() for c in df.columns]
            col_ruta   = next((c for c in df.columns
                               if c in ("ruta", "archivo", "path", "file")), None)
            # Preferir etiqueta_final sobre etiqueta_auto o etiqueta
            col_etiq   = next((c for c in df.columns
                               if c == "etiqueta_final"), None) or \
                         next((c for c in df.columns
                               if c in ("etiqueta", "etiqueta_auto", "label", "tag")), None)
            col_pinon  = next((c for c in df.columns
                               if "pinon" in c or "piñon" in c), None)
            col_nivel  = next((c for c in df.columns
                               if "nivel" in c and "valid" in c), None)

            if col_ruta is None:
                self.after(0, lambda: messagebox.showerror(
                    "Error", "El CSV no tiene columna 'ruta' o 'archivo'."))
                self._habilitar_btn(); return

            # Filtrar: etiqueta_final BUENO o MALO + nivel_validacion VÁLIDO
            ETIQUETAS_VALIDAS = {"bueno", "malo"}
            if col_etiq:
                mask = df[col_etiq].astype(str).str.strip().str.lower().isin(
                    ETIQUETAS_VALIDAS)
                df = df[mask].copy()
            if col_nivel:
                mask_val = df[col_nivel].astype(str).str.strip().str.upper() == "VÁLIDO"
                # fallback sin tilde
                mask_val |= df[col_nivel].astype(str).str.strip().str.upper() == "VALIDO"
                df = df[mask_val].copy()

            if len(df) == 0:
                self.after(0, lambda: messagebox.showwarning(
                    "Sin datos", "No hay archivos válidos en el dataset."))
                self._habilitar_btn(); return

            # Resolver rutas relativas
            base = os.path.dirname(ruta_csv)
            def resolver(r):
                r = str(r)
                return r if os.path.isabs(r) else os.path.join(base, r)
            df["_ruta"] = df[col_ruta].apply(resolver)

            # Agrupar por piñón — normalizar nombres cortos a nombres completos
            PINON_MAP = {
                "ARB_14":      "ARBOL_SEC_14",
                "ARB_15":      "ARBOL_SEC_15",
                "ARBOL_14":    "ARBOL_SEC_14",
                "ARBOL_15":    "ARBOL_SEC_15",
                "AS14":        "ARBOL_SEC_14",
                "AS15":        "ARBOL_SEC_15",
                "PIMA":        "PIMA",
            }
            if col_pinon:
                def _normalizar_pinon(p):
                    p = str(p).strip().upper()
                    return PINON_MAP.get(p, p)
                df["_pinon_norm"] = df[col_pinon].apply(_normalizar_pinon)
                grupos = {p: sub for p, sub in df.groupby("_pinon_norm")}
            else:
                grupos = {"DESCONOCIDO": df}

            total     = len(df)
            procesado = [0]
            resumen   = []

            # ── Procesar cada grupo de piñón ────────────────────────────
            for pinon_key, sub_df in grupos.items():
                # Determinar GMF
                gmf = None
                for k, v in PINONES.items():
                    if k in pinon_key or pinon_key in k:
                        gmf = v["gmf"]; break
                if gmf is None:
                    self._st(f"Piñón '{pinon_key}' no reconocido — omitido", C_REVISAR)
                    continue

                orden_gmf = gmf / F_ROT
                o_max     = orden_gmf * 3.5
                grilla    = np.linspace(0.1, o_max, N_GRID)

                filas_ret = []
                filas_emp = []
                n_ok = 0; n_err = 0

                rutas = sub_df["_ruta"].tolist()
                self._st(f"Procesando {pinon_key}  —  {len(rutas)} archivos…")

                for ruta in rutas:
                    procesado[0] += 1
                    self._prog(procesado[0], total)

                    if not os.path.isfile(ruta):
                        n_err += 1; continue

                    resultado = extraer_espectros(ruta, gmf)
                    if resultado is None:
                        n_err += 1; continue

                    ords_r, db_r, ords_e, db_e = resultado

                    # Interpolar a grilla común
                    if len(ords_r) > 1:
                        filas_ret.append(np.interp(grilla, ords_r, db_r))
                    if len(ords_e) > 1:
                        filas_emp.append(np.interp(grilla, ords_e, db_e))
                    n_ok += 1

                if n_ok < 2:
                    self._st(f"{pinon_key}: muy pocos archivos válidos ({n_ok}) — omitido",
                             C_MALO)
                    resumen.append(f"{pinon_key}: solo {n_ok} válidos — omitido")
                    continue

                # Calcular estadísticos
                umbral_dict = calcular_umbrales(filas_ret, filas_emp, orden_gmf)
                umbral_dict["pinon"]   = pinon_key
                umbral_dict["n_total"] = n_ok
                umbral_dict["n_error"] = n_err

                # Guardar .pkl
                nombre_pkl = f"umbrales_{pinon_key}.pkl"
                ruta_pkl   = os.path.join(carpeta_sal, nombre_pkl)
                os.makedirs(carpeta_sal, exist_ok=True)
                with open(ruta_pkl, "wb") as f:
                    pickle.dump(umbral_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

                n_ret = len(filas_ret)
                n_emp = len(filas_emp)
                msg   = (f"✓ {pinon_key}: {n_ok} archivos  "
                         f"(ret={n_ret}, emp={n_emp}, err={n_err})  →  {nombre_pkl}")
                self._st(msg, C_BUENO)
                resumen.append(msg)

            # ── Resumen final ────────────────────────────────────────────
            txt_res = "\n".join(resumen) if resumen else "Sin resultados."
            self.after(0, lambda: self.lbl_resumen.config(text=txt_res))
            self._st("Completado.", C_BUENO)

        except Exception as ex:
            self.after(0, lambda: messagebox.showerror("Error inesperado", str(ex)))
            self._st(f"Error: {ex}", C_MALO)

        finally:
            self._habilitar_btn()

    def _habilitar_btn(self):
        self.after(0, lambda: self.btn.config(state="normal"))


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = App()
    app.mainloop()
