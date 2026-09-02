"""
entrenar_clasificador.py
========================
Entrena un clasificador BUENO/MALO para piñones DEMM.

Modelos   : Random Forest, Gradient Boosting, SVM, Reg. Logística, LDA
Validación: Stratified K-Fold (5 folds)
Features  : K_ret, CF_p99_ret, K_emp, CF_p99_emp, rms_ret, rms_emp
Salida    : modelo_nvh_<pinon>_<fecha>.pkl

USO:
  py entrenar_clasificador.py          <- interfaz grafica
  py entrenar_clasificador.py --csv ruta.csv --salida C:/carpeta

REQUISITOS:
  pip install scikit-learn pandas numpy matplotlib
"""

import os, sys, pickle, warnings, argparse, threading, datetime
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm               import SVC
from sklearn.linear_model      import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing     import StandardScaler
from sklearn.pipeline          import Pipeline
from sklearn.model_selection   import StratifiedKFold, cross_validate, learning_curve
from sklearn.metrics           import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.inspection        import permutation_importance

warnings.filterwarnings("ignore")

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
C_AMARILLO = "#f59e0b"
C_MONO     = "Consolas"

FEATURES  = ["K_ret","CF_p99_ret","K_emp","CF_p99_emp","rms_ret","rms_emp"]
FEAT_DISP = ["K ret","CF ret","K emp","CF emp","RMS ret","RMS emp"]
LABEL_COL = "etiqueta_final"
CLASES    = ["BUENO","MALO"]

def cargar_dataset(ruta_csv):
    df    = pd.read_csv(ruta_csv)
    train = df[df[LABEL_COL].isin(CLASES)].dropna(subset=FEATURES).copy()
    X     = train[FEATURES].values.astype(np.float64)
    y     = (train[LABEL_COL] == "MALO").astype(int).values
    pinon = str(train["pinon"].mode()[0]) if "pinon" in train.columns else "DESCONOCIDO"
    return X, y, train, pinon

def construir_modelos():
    return {
        "Random Forest": Pipeline([("sc",StandardScaler()),("clf",RandomForestClassifier(n_estimators=300,max_depth=6,min_samples_leaf=3,class_weight="balanced",random_state=42,n_jobs=-1))]),
        "Gradient Boosting": Pipeline([("sc",StandardScaler()),("clf",GradientBoostingClassifier(n_estimators=200,max_depth=4,learning_rate=0.05,subsample=0.8,random_state=42))]),
        "SVM": Pipeline([("sc",StandardScaler()),("clf",SVC(kernel="rbf",C=2.0,gamma="scale",class_weight="balanced",probability=True,random_state=42))]),
        "Reg. Logistica": Pipeline([("sc",StandardScaler()),("clf",LogisticRegression(C=1.0,class_weight="balanced",max_iter=1000,random_state=42))]),
        "LDA": Pipeline([("sc",StandardScaler()),("clf",LinearDiscriminantAnalysis())]),
    }

def evaluar_modelos(X, y, cb_log=None):
    def log(m):
        if cb_log: cb_log(m)
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    res = {}
    for nombre, pipe in construir_modelos().items():
        log(f"  {nombre}...")
        sc = cross_validate(pipe, X, y, cv=cv, scoring=["accuracy","f1","roc_auc"], n_jobs=-1)
        res[nombre] = dict(
            acc_mean=float(np.mean(sc["test_accuracy"])), f1_mean=float(np.mean(sc["test_f1"])),
            auc_mean=float(np.mean(sc["test_roc_auc"])), acc_std=float(np.std(sc["test_accuracy"])),
            f1_std=float(np.std(sc["test_f1"])),         auc_std=float(np.std(sc["test_roc_auc"])),
        )
        log(f"    Acc={res[nombre]['acc_mean']:.3f}  F1={res[nombre]['f1_mean']:.3f}  AUC={res[nombre]['auc_mean']:.3f}")
    return res

def entrenar_final(X, y, resultados, cb_log=None):
    def log(m):
        if cb_log: cb_log(m)
    mejor = max(resultados, key=lambda k: resultados[k]["f1_mean"])
    log(f"\n  Mejor modelo: {mejor}")
    pipe = construir_modelos()[mejor]
    pipe.fit(X, y)
    y_pred = pipe.predict(X)
    y_prob = pipe.predict_proba(X)[:,1]
    report = classification_report(y, y_pred, target_names=["BUENO","MALO"], output_dict=True)
    cm     = confusion_matrix(y, y_pred)
    auc    = roc_auc_score(y, y_prob)
    fpr, tpr, _ = roc_curve(y, y_prob)
    pi = permutation_importance(pipe, X, y, n_repeats=20, random_state=42, n_jobs=-1)
    szs, tr_sc, val_sc = learning_curve(pipe, X, y,
        cv=StratifiedKFold(5,shuffle=True,random_state=42), scoring="f1",
        train_sizes=np.linspace(0.2,1.0,8), n_jobs=-1)
    return dict(pipe=pipe, mejor=mejor, report=report, cm=cm, auc=auc,
                fpr=fpr, tpr=tpr, pi=pi, y_pred=y_pred, y_prob=y_prob,
                lc_sizes=szs, lc_train=tr_sc, lc_val=val_sc)

def guardar_modelo(res, pinon, ruta_csv, carpeta):
    fecha = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    fname = f"modelo_nvh_{pinon.replace('/','_')}_{fecha}.pkl"
    ruta  = os.path.join(carpeta, fname)
    with open(ruta,"wb") as f:
        pickle.dump({"pipeline":res["pipe"],"feature_cols":FEATURES,"clases":CLASES,
                     "pinon":pinon,"modelo_tipo":res["mejor"],"fecha":fecha,
                     "csv_fuente":os.path.basename(ruta_csv)}, f)
    return ruta, fname

class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Entrenador Clasificador NVH — DEMM")
        self.root.geometry("1200x820")
        self.root.minsize(900,600)
        self.root.configure(bg=C_BG)
        self.ruta_csv    = tk.StringVar()
        self.carpeta_out = tk.StringVar()
        self._procesando = False
        self._res = self._resultados = None
        self._X = self._y = self._pinon = None
        self._build_header()
        self._build_main()
        self._build_footer()

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=C_SURFACE, height=50)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Frame(hdr, bg=C_ACENTO, width=3).pack(side="left", fill="y")
        tk.Label(hdr, text="HORSE", bg=C_SURFACE, fg=C_ACENTO, font=(C_MONO,13,"bold")).pack(side="left", padx=14)
        tk.Frame(hdr, bg=C_BORDER, width=1).pack(side="left", fill="y", pady=8)
        fh = tk.Frame(hdr, bg=C_SURFACE); fh.pack(side="left", padx=12)
        tk.Label(fh, text="ENTRENADOR CLASIFICADOR NVH  —  DEMM", bg=C_SURFACE, fg=C_TEXT, font=(C_MONO,11,"bold")).pack(anchor="w")
        self.lbl_info_hdr = tk.Label(fh, text="Sin dataset cargado", bg=C_SURFACE, fg=C_TEXT_SUB, font=(C_MONO,9))
        self.lbl_info_hdr.pack(anchor="w")
        tk.Frame(self.root, bg=C_BORDER, height=1).pack(fill="x")

    def _build_main(self):
        main = tk.Frame(self.root, bg=C_BG)
        main.pack(fill="both", expand=True)
        self._build_panel_izq(main)
        tk.Frame(main, bg=C_BORDER, width=1).pack(side="left", fill="y")
        self._build_panel_der(main)

    def _build_panel_izq(self, parent):
        panel = tk.Frame(parent, bg=C_SURFACE, width=260)
        panel.pack(side="left", fill="y"); panel.pack_propagate(False)
        pad = dict(padx=12)

        def sec(txt):
            tk.Label(panel, text=txt, bg=C_SURFACE, fg=C_TEXT_SUB, font=(C_MONO,8,"bold"), pady=8, **pad).pack(anchor="w")
            tk.Frame(panel, bg=C_BORDER, height=1).pack(fill="x")

        sec("DATASET CSV")
        row = tk.Frame(panel, bg=C_SURFACE, pady=6, **pad); row.pack(fill="x")
        tk.Entry(row, textvariable=self.ruta_csv, bg=C_SURFACE2, fg=C_TEXT, insertbackground=C_TEXT,
                 font=(C_MONO,7), relief="flat", bd=3).pack(side="left", fill="x", expand=True, ipady=4)
        tk.Button(row, text="...", command=self._sel_csv, bg=C_BORDER2, fg=C_ACENTO, relief="flat",
                  font=(C_MONO,9,"bold"), cursor="hand2", padx=6).pack(side="left", padx=(4,0))

        sec("CARPETA SALIDA")
        row2 = tk.Frame(panel, bg=C_SURFACE, pady=6, **pad); row2.pack(fill="x")
        tk.Entry(row2, textvariable=self.carpeta_out, bg=C_SURFACE2, fg=C_TEXT, insertbackground=C_TEXT,
                 font=(C_MONO,7), relief="flat", bd=3).pack(side="left", fill="x", expand=True, ipady=4)
        tk.Button(row2, text="...", command=self._sel_carpeta, bg=C_BORDER2, fg=C_ACENTO, relief="flat",
                  font=(C_MONO,9,"bold"), cursor="hand2", padx=6).pack(side="left", padx=(4,0))

        tk.Frame(panel, bg=C_BORDER, height=1).pack(fill="x")
        stats = tk.Frame(panel, bg=C_SURFACE, **pad, pady=8); stats.pack(fill="x")
        self.lbl_stats = tk.Label(stats, text="—", bg=C_SURFACE, fg=C_TEXT_DIM, font=(C_MONO,8), justify="left")
        self.lbl_stats.pack(anchor="w")

        tk.Frame(panel, bg=C_BORDER, height=1).pack(fill="x")
        bf = tk.Frame(panel, bg=C_SURFACE, pady=10, **pad); bf.pack(fill="x")
        self.btn_train = tk.Button(bf, text="ENTRENAR", command=self._entrenar,
                                   bg=C_ACENTO, fg=C_BG, activebackground="#7ab3f7", activeforeground=C_BG,
                                   relief="flat", cursor="hand2", font=(C_MONO,10,"bold"), pady=8)
        self.btn_train.pack(fill="x")

        style = ttk.Style(); style.theme_use("default")
        style.configure("D.Horizontal.TProgressbar", troughcolor=C_SURFACE2, background=C_ACENTO,
                        darkcolor=C_ACENTO, lightcolor=C_ACENTO, bordercolor=C_SURFACE2, thickness=4)
        self.progress = ttk.Progressbar(panel, mode="indeterminate", style="D.Horizontal.TProgressbar")
        self.progress.pack(fill="x", padx=12)

        tk.Frame(panel, bg=C_BORDER, height=1).pack(fill="x", pady=(6,0))
        tk.Label(panel, text="LOG", bg=C_SURFACE, fg=C_TEXT_SUB, font=(C_MONO,7,"bold"), **pad, pady=4).pack(anchor="w")
        lf = tk.Frame(panel, bg=C_SURFACE2); lf.pack(fill="both", expand=True, padx=8, pady=(0,8))
        self.txt_log = tk.Text(lf, bg=C_SURFACE2, fg=C_TEXT_SUB, font=(C_MONO,7), relief="flat", bd=0,
                               state="disabled", wrap="word", padx=6, pady=6)
        scr = tk.Scrollbar(lf, command=self.txt_log.yview, bg=C_SURFACE2, troughcolor=C_SURFACE2, relief="flat", width=4)
        self.txt_log.configure(yscrollcommand=scr.set)
        self.txt_log.pack(side="left", fill="both", expand=True); scr.pack(side="right", fill="y")

        self.btn_guardar = tk.Button(panel, text="GUARDAR MODELO", command=self._guardar,
                                     bg=C_BUENO, fg=C_BG, activebackground="#16a34a", relief="flat",
                                     cursor="hand2", font=(C_MONO,9,"bold"), pady=6, state="disabled")
        self.btn_guardar.pack(fill="x", padx=12, pady=(0,8))

    def _build_panel_der(self, parent):
        self.frame_graf = tk.Frame(parent, bg=C_BG)
        self.frame_graf.pack(side="left", fill="both", expand=True)
        ghdr = tk.Frame(self.frame_graf, bg=C_SURFACE, height=36)
        ghdr.pack(fill="x"); ghdr.pack_propagate(False)
        self.lbl_graf_hdr = tk.Label(ghdr, text="Entrena un modelo para ver los graficos",
                                     bg=C_SURFACE, fg=C_TEXT_SUB, font=(C_MONO,9))
        self.lbl_graf_hdr.pack(side="left", padx=12)
        tk.Frame(self.frame_graf, bg=C_BORDER, height=1).pack(fill="x")
        self.fig = plt.figure(figsize=(11,7), facecolor=C_BG)
        self._setup_axes()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_graf)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _setup_axes(self):
        self.fig.clf()
        gs = gridspec.GridSpec(2,3, figure=self.fig, hspace=0.42, wspace=0.38,
                               left=0.07, right=0.97, top=0.93, bottom=0.09)
        self.ax_cv   = self.fig.add_subplot(gs[0,0])
        self.ax_roc  = self.fig.add_subplot(gs[0,1])
        self.ax_cm   = self.fig.add_subplot(gs[0,2])
        self.ax_imp  = self.fig.add_subplot(gs[1,0])
        self.ax_lc   = self.fig.add_subplot(gs[1,1])
        self.ax_dist = self.fig.add_subplot(gs[1,2])
        for ax in self.fig.get_axes():
            ax.set_facecolor(C_SURFACE)
            ax.tick_params(colors=C_TEXT_SUB, labelsize=7)
            for sp in ax.spines.values(): sp.set_color(C_BORDER2)
            ax.grid(True, alpha=0.08, color=C_BORDER2)
        if hasattr(self,"canvas"): self.canvas.draw()

    def _build_footer(self):
        tk.Frame(self.root, bg=C_BORDER, height=1).pack(fill="x")
        self.lbl_status = tk.Label(self.root, text="Listo", bg=C_SURFACE, fg=C_TEXT_DIM,
                                   font=(C_MONO,8), anchor="w", padx=12, pady=4)
        self.lbl_status.pack(fill="x")

    def _sel_csv(self):
        r = filedialog.askopenfilename(title="CSV de etiquetado", filetypes=[("CSV","*.csv"),("Todos","*.*")])
        if not r: return
        self.ruta_csv.set(r)
        if not self.carpeta_out.get(): self.carpeta_out.set(os.path.dirname(r))
        self._cargar_info(r)

    def _sel_carpeta(self):
        d = filedialog.askdirectory(title="Carpeta para el modelo")
        if d: self.carpeta_out.set(d)

    def _cargar_info(self, ruta):
        try:
            df    = pd.read_csv(ruta)
            train = df[df[LABEL_COL].isin(CLASES)].dropna(subset=FEATURES)
            malo  = len(train[train[LABEL_COL]=="MALO"])
            bueno = len(train[train[LABEL_COL]=="BUENO"])
            pinon = str(train["pinon"].mode()[0]) if "pinon" in train.columns else "?"
            rev   = len(df[df[LABEL_COL]=="REVISAR"])
            inv   = len(df[df[LABEL_COL].isin(["INVALIDO","INVALIDO"])])
            self.lbl_info_hdr.config(
                text=f"{os.path.basename(ruta)}  |  Pinon: {pinon}  |  BUENO {bueno}  MALO {malo}  (REVISAR {rev}  INVALIDO {inv} excluidos)",
                fg=C_ACENTO)
            self.lbl_stats.config(
                text=f"Total : {bueno+malo}\nBUENO : {bueno} ({100*bueno/(bueno+malo):.1f}%)\nMALO  : {malo} ({100*malo/(bueno+malo):.1f}%)\nPinon : {pinon}",
                fg=C_TEXT_SUB)
            self._X, self._y, _, self._pinon = cargar_dataset(ruta)
            self._dibujar_distribucion()
        except Exception as e:
            self.lbl_stats.config(text=f"Error: {e}", fg=C_MALO)

    def _log(self, msg):
        self.txt_log.configure(state="normal")
        self.txt_log.insert("end", msg+"\n")
        self.txt_log.see("end")
        self.txt_log.configure(state="disabled")

    def _status(self, msg, color=C_TEXT_DIM):
        self.lbl_status.config(text=msg, fg=color)

    def _dibujar_distribucion(self):
        if self._X is None: return
        ax = self.ax_dist; ax.clear(); ax.set_facecolor(C_SURFACE)
        for sp in ax.spines.values(): sp.set_color(C_BORDER2)
        ax.grid(True, alpha=0.08, color=C_BORDER2); ax.tick_params(colors=C_TEXT_SUB, labelsize=7)
        X, y = self._X, self._y
        for val, label, color in [(0,"BUENO",C_BUENO),(1,"MALO",C_MALO)]:
            ax.hist(X[y==val,0], bins=20, alpha=0.6, color=color, label=label, edgecolor="none")
        ax.axvline(4.4, color=C_AMARILLO, lw=1.2, ls="--", alpha=0.8, label="Umbral K")
        ax.set_title("Distribucion K_ret por clase", fontsize=8, color=C_TEXT_SUB, pad=4)
        ax.set_xlabel("K retroceso", fontsize=7, color=C_TEXT_SUB)
        ax.set_ylabel("N archivos",  fontsize=7, color=C_TEXT_SUB)
        ax.legend(fontsize=6, facecolor=C_SURFACE2, edgecolor=C_BORDER2, labelcolor=C_TEXT_SUB)
        self.canvas.draw()

    def _entrenar(self):
        if self._procesando: return
        csv = self.ruta_csv.get().strip()
        if not csv or not os.path.isfile(csv):
            self._status("Selecciona un CSV valido", C_MALO); return
        self._procesando = True
        self.txt_log.configure(state="normal"); self.txt_log.delete("1.0","end"); self.txt_log.configure(state="disabled")
        self.btn_train.config(state="disabled", text="Entrenando...", bg=C_SURFACE2, fg=C_TEXT_SUB)
        self.btn_guardar.config(state="disabled")
        self.progress.start(12)
        self._status("Entrenando modelos...", C_AMARILLO)
        def tarea():
            try:
                self.root.after(0, lambda: self._log("Cargando dataset..."))
                X, y, _, pinon = cargar_dataset(csv)
                self._X = X; self._y = y; self._pinon = pinon
                self.root.after(0, lambda: self._log(f"  {pinon}  BUENO:{int(len(y)-np.sum(y))}  MALO:{int(np.sum(y))}\n"))
                self.root.after(0, lambda: self._log("Evaluando modelos (CV 5-fold)..."))
                res = evaluar_modelos(X, y, cb_log=lambda m: self.root.after(0, lambda msg=m: self._log(msg)))
                self._resultados = res
                self.root.after(0, lambda: self._log("\nEntrenando modelo final..."))
                r = entrenar_final(X, y, res, cb_log=lambda m: self.root.after(0, lambda msg=m: self._log(msg)))
                self._res = r
                self.root.after(0, lambda: self._mostrar_graficos(r, res))
            except Exception as ex:
                import traceback; tb = traceback.format_exc()
                self.root.after(0, lambda: self._log(f"\nERROR: {ex}\n{tb}"))
                self.root.after(0, lambda: self._status(f"Error: {ex}", C_MALO))
                self.root.after(0, self._fin_proceso)
        threading.Thread(target=tarea, daemon=True).start()

    def _fin_proceso(self):
        self._procesando = False; self.progress.stop()
        self.btn_train.config(state="normal", text="ENTRENAR", bg=C_ACENTO, fg=C_BG)

    def _mostrar_graficos(self, r, res):
        self._fin_proceso(); mejor = r["mejor"]
        self.lbl_graf_hdr.config(
            text=f"Modelo seleccionado: {mejor}  |  F1={res[mejor]['f1_mean']:.3f}  AUC={res[mejor]['auc_mean']:.3f}  Acc={res[mejor]['acc_mean']:.3f}",
            fg=C_BUENO)
        self._status(f"Entrenamiento completo — {mejor}", C_BUENO)
        self.btn_guardar.config(state="normal")
        self._setup_axes()
        self._graf_cv(res, mejor); self._graf_roc(r); self._graf_cm(r)
        self._graf_importancia(r); self._graf_curva_aprendizaje(r); self._dibujar_distribucion()
        self.fig.tight_layout(pad=1.2); self.canvas.draw()

    def _graf_cv(self, res, mejor):
        ax = self.ax_cv; ax.clear(); ax.set_facecolor(C_SURFACE)
        for sp in ax.spines.values(): sp.set_color(C_BORDER2)
        ax.grid(True, alpha=0.08, color=C_BORDER2); ax.tick_params(colors=C_TEXT_SUB, labelsize=7)
        nombres = list(res.keys())
        x = np.arange(len(nombres)); w = 0.22
        for i,(met,etq,col) in enumerate(zip(["f1_mean","acc_mean","auc_mean"],["F1","Acc","AUC"],[C_ACENTO,C_BUENO,C_AMARILLO])):
            vals = [res[n][met] for n in nombres]
            bars = ax.bar(x+(i-1)*w, vals, w, label=etq, color=col, alpha=0.85, zorder=3)
            for bar,v in zip(bars,vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=5.5, color=col)
        ax.axvspan(nombres.index(mejor)-0.4, nombres.index(mejor)+0.4, alpha=0.07, color=C_BUENO, zorder=1)
        ax.set_xticks(x); ax.set_xticklabels([n.replace(" ","\n") for n in nombres], fontsize=6, color=C_TEXT_SUB)
        ax.set_ylim(0,1.08)
        ax.set_title("Comparacion CV 5-fold", fontsize=8, color=C_TEXT_SUB, pad=4)
        ax.legend(fontsize=6, facecolor=C_SURFACE2, edgecolor=C_BORDER2, labelcolor=C_TEXT_SUB)

    def _graf_roc(self, r):
        ax = self.ax_roc; ax.clear(); ax.set_facecolor(C_SURFACE)
        for sp in ax.spines.values(): sp.set_color(C_BORDER2)
        ax.grid(True, alpha=0.08, color=C_BORDER2); ax.tick_params(colors=C_TEXT_SUB, labelsize=7)
        ax.plot(r["fpr"], r["tpr"], color=C_ACENTO, lw=1.8, label=f"AUC={r['auc']:.3f}")
        ax.plot([0,1],[0,1], color=C_BORDER2, lw=0.8, ls="--")
        ax.fill_between(r["fpr"], r["tpr"], alpha=0.08, color=C_ACENTO)
        ax.set_title("Curva ROC", fontsize=8, color=C_TEXT_SUB, pad=4)
        ax.set_xlabel("Tasa Falsos Positivos", fontsize=7, color=C_TEXT_SUB)
        ax.set_ylabel("Tasa Verdaderos Positivos", fontsize=7, color=C_TEXT_SUB)
        ax.legend(fontsize=7, facecolor=C_SURFACE2, edgecolor=C_BORDER2, labelcolor=C_TEXT_SUB)
        ax.set_xlim(0,1); ax.set_ylim(0,1.02)

    def _graf_cm(self, r):
        ax = self.ax_cm; ax.clear(); ax.set_facecolor(C_SURFACE)
        for sp in ax.spines.values(): sp.set_color(C_BORDER2)
        ax.tick_params(colors=C_TEXT_SUB, labelsize=7)
        cm = r["cm"]
        ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["BUENO","MALO"], fontsize=7, color=C_TEXT_SUB)
        ax.set_yticklabels(["BUENO","MALO"], fontsize=7, color=C_TEXT_SUB)
        ax.set_xlabel("Predicho", fontsize=7, color=C_TEXT_SUB)
        ax.set_ylabel("Real",     fontsize=7, color=C_TEXT_SUB)
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                v = cm[i,j]
                ax.text(j, i, f"{v}\n({100*v/total:.1f}%)", ha="center", va="center",
                        fontsize=8, color="white" if v>cm.max()/2 else C_TEXT_SUB, fontweight="bold")
        ax.set_title("Matriz de Confusion (train)", fontsize=8, color=C_TEXT_SUB, pad=4)

    def _graf_importancia(self, r):
        ax = self.ax_imp; ax.clear(); ax.set_facecolor(C_SURFACE)
        for sp in ax.spines.values(): sp.set_color(C_BORDER2)
        ax.grid(True, alpha=0.08, color=C_BORDER2); ax.tick_params(colors=C_TEXT_SUB, labelsize=7)
        imp = r["pi"].importances_mean; orden = np.argsort(imp)
        cols = [FEAT_DISP[i] for i in orden]; vals = imp[orden]
        ax.barh(range(len(cols)), vals, color=[C_ACENTO if v>0 else C_MALO for v in vals], alpha=0.85, zorder=3)
        ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=7, color=C_TEXT_SUB)
        ax.set_title("Importancia de Features\n(permutation)", fontsize=8, color=C_TEXT_SUB, pad=4)
        ax.set_xlabel("Reduccion F1 media", fontsize=7, color=C_TEXT_SUB)

    def _graf_curva_aprendizaje(self, r):
        ax = self.ax_lc; ax.clear(); ax.set_facecolor(C_SURFACE)
        for sp in ax.spines.values(): sp.set_color(C_BORDER2)
        ax.grid(True, alpha=0.08, color=C_BORDER2); ax.tick_params(colors=C_TEXT_SUB, labelsize=7)
        szs=r["lc_sizes"]; tr=r["lc_train"]; val=r["lc_val"]
        ax.plot(szs, np.mean(tr, axis=1),  color=C_AMARILLO, lw=1.5, label="Train")
        ax.fill_between(szs, np.mean(tr,axis=1)-np.std(tr,axis=1), np.mean(tr,axis=1)+np.std(tr,axis=1), alpha=0.12, color=C_AMARILLO)
        ax.plot(szs, np.mean(val,axis=1),  color=C_ACENTO,   lw=1.5, label="Validacion")
        ax.fill_between(szs, np.mean(val,axis=1)-np.std(val,axis=1), np.mean(val,axis=1)+np.std(val,axis=1), alpha=0.12, color=C_ACENTO)
        ax.set_title("Curva de Aprendizaje (F1)", fontsize=8, color=C_TEXT_SUB, pad=4)
        ax.set_xlabel("Muestras de entrenamiento", fontsize=7, color=C_TEXT_SUB)
        ax.set_ylabel("F1", fontsize=7, color=C_TEXT_SUB)
        ax.legend(fontsize=6, facecolor=C_SURFACE2, edgecolor=C_BORDER2, labelcolor=C_TEXT_SUB)
        ax.set_ylim(0,1.05)

    def _guardar(self):
        if not self._res: return
        carpeta = self.carpeta_out.get().strip() or os.path.dirname(self.ruta_csv.get())
        os.makedirs(carpeta, exist_ok=True)
        ruta_pkl, fname = guardar_modelo(self._res, self._pinon, self.ruta_csv.get(), carpeta)
        self._log(f"\nModelo guardado: {fname}")
        self._status(f"Modelo guardado: {fname}", C_BUENO)
        if sys.platform == "win32": os.startfile(carpeta)

    def run(self):
        self.root.mainloop()

def modo_cli(args):
    print(f"\nCargando {args.csv}...")
    X, y, _, pinon = cargar_dataset(args.csv)
    print(f"  {pinon}  BUENO:{int(len(y)-np.sum(y))}  MALO:{int(np.sum(y))}")
    print("\nEvaluando modelos...")
    res = evaluar_modelos(X, y, cb_log=print)
    print("\nEntrenando modelo final...")
    r   = entrenar_final(X, y, res, cb_log=print)
    carpeta = args.salida or os.path.dirname(args.csv)
    ruta_pkl, fname = guardar_modelo(r, pinon, args.csv, carpeta)
    print(f"\nModelo guardado: {ruta_pkl}")
    print(classification_report(y, r["y_pred"], target_names=["BUENO","MALO"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",    default=None)
    parser.add_argument("--salida", default=None)
    args = parser.parse_args()
    if args.csv: modo_cli(args)
    else: App().run()
