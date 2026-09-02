"""
calcular_umbral_kurtosis.py
===========================
Calcula el umbral estadístico (media + 3×desviación estándar)
de la Kurtosis para los datos de un piñón.

Filtros aplicados:
  - nivel_validacion = VÁLIDO
  - etiqueta_final   = BUENO  (solo piñones sin defecto)

Muestra resultados por flanco (Retroceso y Empuje) y combinado.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import os

# ── Colores ──────────────────────────────────────────────────────────────────
C_BG       = "#f0f2f5"
C_SURFACE  = "#ffffff"
C_SURFACE2 = "#e8eaed"
C_TEXT     = "#1a1d27"
C_TEXT_DIM = "#7f8c8d"
C_ACENTO   = "#1a5fa8"
C_BUENO    = "#16a34a"
C_MALO     = "#dc2626"
C_REVISAR  = "#d97706"
C_BORDER   = "#cbd5e1"
C_MONO     = "Courier"


# ── Ventana principal ─────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Umbral Kurtosis μ+3σ — MathIA")
        self.geometry("700x620")
        self.minsize(600, 520)
        self.configure(bg=C_BG)
        self.resizable(True, True)
        self._df     = None
        self._ruta   = None
        self._build()

    def _build(self):
        # ── Header ───────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=C_SURFACE, height=54)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Frame(hdr, bg=C_ACENTO, width=4).pack(side="left", fill="y")
        fh = tk.Frame(hdr, bg=C_SURFACE); fh.pack(side="left", padx=14)
        tk.Label(fh, text="Umbral Kurtosis  μ + 3σ",
                 bg=C_SURFACE, fg=C_TEXT,
                 font=("Arial", 13, "bold")).pack(anchor="w")
        tk.Label(fh, text="Calcula el umbral estadístico de Kurtosis por piñón",
                 bg=C_SURFACE, fg=C_TEXT_DIM,
                 font=("Arial", 9)).pack(anchor="w")
        tk.Frame(self, bg=C_BORDER, height=1).pack(fill="x")

        # ── Cuerpo ────────────────────────────────────────────────────────────
        cuerpo = tk.Frame(self, bg=C_BG, padx=28, pady=20)
        cuerpo.pack(fill="both", expand=True)

        # Sección 1 — Cargar archivo
        tk.Label(cuerpo, text="1.  Cargar dataset",
                 bg=C_BG, fg=C_ACENTO,
                 font=("Arial", 10, "bold")).pack(anchor="w")
        tk.Frame(cuerpo, bg=C_BORDER, height=1).pack(fill="x", pady=(2, 10))

        f_carga = tk.Frame(cuerpo, bg=C_BG); f_carga.pack(fill="x")
        self.lbl_archivo = tk.Label(f_carga,
                                     text="Sin archivo cargado",
                                     bg=C_SURFACE2, fg=C_TEXT_DIM,
                                     font=(C_MONO, 8),
                                     anchor="w", relief="flat",
                                     padx=10, pady=8)
        self.lbl_archivo.pack(side="left", fill="x", expand=True)
        tk.Button(f_carga, text="📂  Abrir CSV",
                  command=self._cargar,
                  bg=C_ACENTO, fg="white",
                  activebackground="#1349a0", activeforeground="white",
                  relief="flat", bd=0,
                  font=("Arial", 9, "bold"),
                  cursor="hand2", padx=14, pady=8
                  ).pack(side="left", padx=(8, 0))

        # Sección 2 — Filtros
        tk.Label(cuerpo, text="2.  Filtros",
                 bg=C_BG, fg=C_ACENTO,
                 font=("Arial", 10, "bold")).pack(anchor="w", pady=(18, 0))
        tk.Frame(cuerpo, bg=C_BORDER, height=1).pack(fill="x", pady=(2, 10))

        f_filtros = tk.Frame(cuerpo, bg=C_BG); f_filtros.pack(fill="x")

        # Filtro nivel_validacion
        tk.Label(f_filtros, text="nivel_validacion:",
                 bg=C_BG, fg=C_TEXT,
                 font=("Arial", 9)).grid(row=0, column=0, sticky="w", pady=4)
        self.var_nivel = tk.StringVar(value="VÁLIDO")
        ttk.Combobox(f_filtros, textvariable=self.var_nivel,
                     values=["VÁLIDO", "Todos"],
                     state="readonly", width=14,
                     font=("Arial", 9)
                     ).grid(row=0, column=1, padx=(8, 32), sticky="w")

        # Filtro etiqueta_final
        tk.Label(f_filtros, text="etiqueta_final:",
                 bg=C_BG, fg=C_TEXT,
                 font=("Arial", 9)).grid(row=0, column=2, sticky="w", pady=4)
        self.var_etiq = tk.StringVar(value="BUENO")
        ttk.Combobox(f_filtros, textvariable=self.var_etiq,
                     values=["BUENO", "MALO", "BUENO + MALO", "Todos"],
                     state="readonly", width=14,
                     font=("Arial", 9)
                     ).grid(row=0, column=3, padx=(8, 0), sticky="w")

        # Sección 3 — Calcular
        tk.Label(cuerpo, text="3.  Calcular",
                 bg=C_BG, fg=C_ACENTO,
                 font=("Arial", 10, "bold")).pack(anchor="w", pady=(18, 0))
        tk.Frame(cuerpo, bg=C_BORDER, height=1).pack(fill="x", pady=(2, 10))

        tk.Button(cuerpo, text="▶  Calcular umbral  μ + 3σ",
                  command=self._calcular,
                  bg=C_ACENTO, fg="white",
                  activebackground="#1349a0", activeforeground="white",
                  relief="flat", bd=0,
                  font=("Arial", 11, "bold"),
                  cursor="hand2", pady=10
                  ).pack(fill="x")

        # Sección 4 — Resultados
        tk.Label(cuerpo, text="4.  Resultados",
                 bg=C_BG, fg=C_ACENTO,
                 font=("Arial", 10, "bold")).pack(anchor="w", pady=(18, 0))
        tk.Frame(cuerpo, bg=C_BORDER, height=1).pack(fill="x", pady=(2, 8))

        self.frame_res = tk.Frame(cuerpo, bg=C_SURFACE,
                                   relief="flat", bd=0)
        self.frame_res.pack(fill="both", expand=True)

        self.lbl_res = tk.Label(self.frame_res,
                                 text="Carga un CSV y presiona Calcular.",
                                 bg=C_SURFACE, fg=C_TEXT_DIM,
                                 font=("Arial", 10),
                                 justify="left", anchor="nw",
                                 padx=16, pady=12)
        self.lbl_res.pack(fill="both", expand=True)

    def _cargar(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar dataset CSV",
            filetypes=[("CSV", "*.csv"), ("Todos", "*.*")])
        if not ruta:
            return
        try:
            self._df    = pd.read_csv(ruta)
            self._ruta  = ruta
            nombre      = os.path.basename(ruta)
            n           = len(self._df)
            cols        = list(self._df.columns)
            self.lbl_archivo.config(
                text=f"{nombre}  ({n} filas)",
                fg=C_TEXT)
            # Mostrar columnas K disponibles
            k_cols = [c for c in cols if "K_" in c or "kurt" in c.lower()]
            self.lbl_res.config(
                text=f"✓ Archivo cargado: {nombre}\n"
                     f"  Filas: {n}\n"
                     f"  Columnas K detectadas: {', '.join(k_cols) if k_cols else 'ninguna'}\n\n"
                     "Presiona ▶ Calcular para obtener los umbrales.",
                fg=C_TEXT)
        except Exception as e:
            messagebox.showerror("Error al cargar", str(e))

    def _calcular(self):
        if self._df is None:
            messagebox.showwarning("Sin datos", "Carga un archivo CSV primero.")
            return

        df = self._df.copy()

        # ── Aplicar filtros ───────────────────────────────────────────────────
        n_orig = len(df)

        # Filtro nivel_validacion
        if self.var_nivel.get() != "Todos":
            col_niv = next((c for c in df.columns
                            if "nivel" in c.lower() and "valid" in c.lower()), None)
            if col_niv:
                df = df[df[col_niv].astype(str).str.strip().str.upper().isin(
                    ["VÁLIDO", "VALIDO"])].copy()

        # Filtro etiqueta_final
        sel_etiq = self.var_etiq.get()
        col_etiq = next((c for c in df.columns
                         if "etiqueta" in c.lower() and "final" in c.lower()), None)
        if col_etiq and sel_etiq != "Todos":
            if sel_etiq == "BUENO + MALO":
                df = df[df[col_etiq].astype(str).str.upper().isin(
                    ["BUENO", "MALO"])].copy()
            else:
                df = df[df[col_etiq].astype(str).str.upper() == sel_etiq].copy()

        n_filtrado = len(df)

        if n_filtrado == 0:
            messagebox.showwarning(
                "Sin datos",
                f"No quedaron filas después de aplicar los filtros.\n"
                f"Prueba cambiando etiqueta_final a 'Todos'.")
            return

        # ── Buscar columnas de kurtosis ───────────────────────────────────────
        col_k_ret = next((c for c in df.columns
                          if c.lower() in ("k_ret", "kurt_ret", "kurtosis_ret")), None)
        col_k_emp = next((c for c in df.columns
                          if c.lower() in ("k_emp", "kurt_emp", "kurtosis_emp")), None)

        if col_k_ret is None and col_k_emp is None:
            messagebox.showerror(
                "Columnas no encontradas",
                "No se encontraron columnas de Kurtosis.\n"
                "Se esperan: K_ret, K_emp (o kurt_ret, kurt_emp).\n\n"
                f"Columnas disponibles:\n{', '.join(df.columns)}")
            return

        # ── Calcular μ + 3σ ───────────────────────────────────────────────────
        resultados = {}

        for col, nombre_flanco in [(col_k_ret, "Retroceso"),
                                    (col_k_emp, "Empuje")]:
            if col is None:
                continue
            serie = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(serie) == 0:
                continue
            media  = float(serie.mean())
            std    = float(serie.std())
            mediana= float(serie.median())
            minv   = float(serie.min())
            maxv   = float(serie.max())
            umb    = media + 3 * std
            resultados[nombre_flanco] = {
                "n":       len(serie),
                "media":   media,
                "std":     std,
                "mediana": mediana,
                "min":     minv,
                "max":     maxv,
                "umbral":  umb,
                "col":     col,
            }

        # Combinado (todos los valores juntos)
        series_todas = []
        for col in [col_k_ret, col_k_emp]:
            if col:
                series_todas.append(pd.to_numeric(df[col], errors="coerce").dropna())
        if series_todas:
            combinado = pd.concat(series_todas)
            resultados["Combinado"] = {
                "n":       len(combinado),
                "media":   float(combinado.mean()),
                "std":     float(combinado.std()),
                "mediana": float(combinado.median()),
                "min":     float(combinado.min()),
                "max":     float(combinado.max()),
                "umbral":  float(combinado.mean()) + 3 * float(combinado.std()),
                "col":     "K_ret + K_emp",
            }

        # ── Formatear resultado ───────────────────────────────────────────────
        # Detectar nombre de piñón si hay columna
        col_pinon = next((c for c in df.columns if "pinon" in c.lower()), None)
        nombre_pinon = ""
        if col_pinon:
            valores = df[col_pinon].dropna().unique()
            nombre_pinon = f"  Piñón: {', '.join(str(v) for v in valores)}\n"

        lineas = []
        lineas.append(f"Archivo : {os.path.basename(self._ruta)}")
        lineas.append(f"Filtros : nivel={self.var_nivel.get()}"
                      f"  |  etiqueta={self.var_etiq.get()}")
        lineas.append(f"Filas   : {n_filtrado} usadas de {n_orig} totales")
        if nombre_pinon.strip():
            lineas.append(f"Piñón   : {nombre_pinon.strip()}")
        lineas.append("")
        lineas.append("─" * 56)

        for flanco, r in resultados.items():
            lineas.append(f"  {flanco}  ({r['col']})")
            lineas.append(f"    n        = {r['n']}")
            lineas.append(f"    Media    = {r['media']:.4f}")
            lineas.append(f"    Std      = {r['std']:.4f}")
            lineas.append(f"    Mediana  = {r['mediana']:.4f}")
            lineas.append(f"    Mín      = {r['min']:.4f}")
            lineas.append(f"    Máx      = {r['max']:.4f}")
            lineas.append("")
            lineas.append(
                f"  ┌─────────────────────────────────────────────┐")
            lineas.append(
                f"  │  UMBRAL μ+3σ ({flanco[:3]}) = {r['umbral']:>8.4f}          │")
            lineas.append(
                f"  └─────────────────────────────────────────────┘")
            lineas.append("")

        lineas.append("─" * 56)

        texto = "\n".join(lineas)

        # Limpiar frame de resultados y mostrar texto
        for w in self.frame_res.winfo_children():
            w.destroy()

        txt = tk.Text(self.frame_res,
                      bg=C_SURFACE, fg=C_TEXT,
                      font=(C_MONO, 9),
                      relief="flat", bd=0,
                      padx=14, pady=10,
                      wrap="none")
        sb_y = tk.Scrollbar(self.frame_res, orient="vertical",
                             command=txt.yview,
                             width=6, relief="flat")
        txt.configure(yscrollcommand=sb_y.set)
        sb_y.pack(side="right", fill="y")
        txt.pack(fill="both", expand=True)

        txt.insert("1.0", texto)

        # Resaltar las líneas de umbral
        for flanco, r in resultados.items():
            patron = f"UMBRAL μ+3σ ({flanco[:3]})"
            idx = "1.0"
            while True:
                pos = txt.search(patron, idx, stopindex="end")
                if not pos:
                    break
                fin = f"{pos}+{len(patron)}c"
                txt.tag_add("umbral", pos, fin)
                idx = fin
        txt.tag_config("umbral", foreground=C_ACENTO,
                       font=(C_MONO, 9, "bold"))
        txt.config(state="disabled")

        # Botón copiar
        tk.Button(self.frame_res,
                  text="📋  Copiar resultados",
                  command=lambda t=texto: self._copiar(t),
                  bg=C_SURFACE2, fg=C_TEXT,
                  activebackground=C_SURFACE2,
                  relief="flat", bd=0,
                  font=("Arial", 8),
                  cursor="hand2", padx=10, pady=4
                  ).pack(anchor="e", padx=10, pady=(4, 6))

    def _copiar(self, texto):
        self.clipboard_clear()
        self.clipboard_append(texto)
        messagebox.showinfo("Copiado",
                            "Resultados copiados al portapapeles.")


if __name__ == "__main__":
    app = App()
    app.mainloop()
