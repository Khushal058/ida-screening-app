"""
gui_app.py  —  IDA AI Screening Tool (Upgraded Desktop GUI)
============================================================
Upgraded per faculty feedback:
  1. Crisp outcome: "Iron Deficiency: Present / Absent"
  2. Hindi / English language toggle
  3. Patient View (simple result only) vs Doctor View (full charts + analysis)
  4. Dashboard-style UI with medical theme

Run train_model.py FIRST, then:  python gui_app.py
"""

import os, sys, io, warnings
warnings.filterwarnings("ignore")
import joblib
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
for path in ["ida_model.joblib", "feature_list.joblib"]:
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Please run 'python train_model.py' first.")
        sys.exit(1)

model          = joblib.load("ida_model.joblib")
FEATURE_COLUMNS = joblib.load("feature_list.joblib")
scaler         = joblib.load("scaler.joblib")      if os.path.exists("scaler.joblib")     else None
model_meta     = joblib.load("model_meta.joblib")  if os.path.exists("model_meta.joblib") else {"name":"Model","uses_scaled_input":False}
USES_SCALED    = model_meta.get("uses_scaled_input", False)

# ---------------------------------------------------------------------------
# Doctor View password (local desktop app — change this before sharing the
# .exe / source with others if you want a different password)
# ---------------------------------------------------------------------------
DOCTOR_PASSWORD = "ssims2026"

# ---------------------------------------------------------------------------
# Language strings
# ---------------------------------------------------------------------------
LANG = {
    "en": {
        "title":        "IDA Screening — AI Diagnostic Tool",
        "subtitle":     "Signs & symptoms only · No blood test required",
        "lang_btn":     "🇮🇳 हिंदी",
        "view_btn_doc": "🩺 Doctor View",
        "view_btn_pat": "👤 Patient View",
        "demo_head":    "PATIENT DEMOGRAPHICS",
        "symp_head":    "SIGNS & SYMPTOMS",
        "predict_btn":  "🔍 Run AI Prediction",
        "present":      "Iron Deficiency: Present",
        "absent":       "Iron Deficiency: Absent",
        "conf":         "Confidence",
        "advice_p":     "⚠ Please consult a doctor and confirm with a blood test.",
        "advice_a":     "✓ No iron deficiency indicated. See a doctor if symptoms persist.",
        "disclaimer":   "Screening aid only — not a substitute for clinical diagnosis.",
        "chart_lbl":    "View Chart:",
        "chart_fi":     "Feature Importance",
        "chart_shap":   "SHAP Summary",
        "chart_cmp":    "Model Comparison",
        "chart_pat":    "This Patient",
        "view_chart":   "View",
        "age": "Age (years)", "gender": "Gender", "address": "Address",
        "edu": "Education",   "occ": "Occupation", "income": "Monthly Income (₹)",
    },
    "hi": {
        "title":        "IDA जांच — AI निदान उपकरण",
        "subtitle":     "केवल लक्षणों पर · रक्त परीक्षण नहीं",
        "lang_btn":     "🇬🇧 English",
        "view_btn_doc": "🩺 डॉक्टर दृश्य",
        "view_btn_pat": "👤 मरीज़ दृश्य",
        "demo_head":    "मरीज़ की जानकारी",
        "symp_head":    "संकेत और लक्षण",
        "predict_btn":  "🔍 AI जांच करें",
        "present":      "आयरन की कमी: उपस्थित",
        "absent":       "आयरन की कमी: अनुपस्थित",
        "conf":         "विश्वास",
        "advice_p":     "⚠ कृपया डॉक्टर से मिलें और रक्त परीक्षण से पुष्टि करें।",
        "advice_a":     "✓ आयरन की कमी नहीं दिखती। लक्षण रहें तो डॉक्टर से मिलें।",
        "disclaimer":   "यह केवल जांच सहायक है — चिकित्सीय निदान का विकल्प नहीं।",
        "chart_lbl":    "चार्ट देखें:",
        "chart_fi":     "फीचर महत्व",
        "chart_shap":   "SHAP सारांश",
        "chart_cmp":    "मॉडल तुलना",
        "chart_pat":    "यह मरीज़",
        "view_chart":   "देखें",
        "age": "आयु (वर्ष)", "gender": "लिंग", "address": "पता",
        "edu": "शिक्षा",     "occ": "व्यवसाय", "income": "मासिक आय (₹)",
    }
}

# ---------------------------------------------------------------------------
# Field definitions (bilingual)
# ---------------------------------------------------------------------------
DEMO_FIELDS = [
    ("Age",       "age",    "age",    None),
    ("Gender",    "gender", "choice", [(1,"पुरुष / Male"),(2,"महिला / Female")]),
    ("Address",   "address","choice", [(1,"शहरी / Urban"),(2,"ग्रामीण / Rural")]),
    ("Education", "edu",    "choice", [
        (1,"अनपढ़ / Illiterate"),(2,"प्राथमिक / Primary"),(3,"माध्यमिक / Middle"),
        (4,"हाई स्कूल / High School"),(5,"इंटर / Intermediate"),(6,"स्नातक / Graduate"),
        (7,"स्नातकोत्तर / Post-Graduate"),
    ]),
    ("Occupation","occ",    "choice", [
        (1,"बेरोज़गार / Unemployed"),(2,"साधारण / Elementary"),
        (3,"ऑपरेटर / Operator"),(4,"दस्तकार / Craft"),
        (5,"कृषि / Agricultural"),(6,"दुकानदार / Shop Worker"),
        (7,"लिपिक / Clerk"),(8,"तकनीशियन / Technician"),
        (9,"पेशेवर / Professional"),(10,"प्रबंधक / Manager"),
    ]),
    ("Income",    "income", "choice", [
        (1,"₹7,988 से कम / Less than ₹7,988"),(2,"₹7,989–₹23,869"),
        (3,"₹23,870–₹39,829"),(4,"₹39,830–₹59,794"),
        (5,"₹59,795–₹79,755"),(6,"₹79,756–₹1,59,585"),(7,"₹1,59,586+"),
    ]),
]

S03 = [(0,"अनुपस्थित / Absent"),(1,"हल्का / Mild"),(2,"मध्यम / Moderate"),(3,"गंभीर / Severe")]
PA  = [(0,"अनुपस्थित / Absent"),(1,"उपस्थित / Present")]

SYMPTOM_FIELDS = [
    ("Paleness",        "असामान्य पीलापन / Abnormal Paleness",    "choice", [
        (0,"अनुपस्थित / Absent"),(1,"हल्का - नेत्र / Mild - conjunctiva"),
        (2,"मध्यम - त्वचा / Moderate - skin"),(3,"गंभीर - हथेली / Severe - palmar"),
    ]),
    ("Irritability",    "चिड़चिड़ापन / Irritability",            "choice", PA),
    ("Fatigue",         "थकान / Unexplained Fatigue",             "choice", [
        (0,"अनुपस्थित / Absent"),(1,"हल्की / Mild"),(2,"मध्यम / Moderate"),(3,"गंभीर / Severe"),
    ]),
    ("Tachycardia",     "तेज़ धड़कन / Tachycardia",              "choice", [
        (0,"अनुपस्थित / Absent"),(1,"101-110 bpm"),(2,"111-120 bpm"),(3,">120 bpm"),
    ]),
    ("Tongue",          "जीभ सूजन / Sore Tongue",                "choice", S03),
    ("Spleen",          "तिल्ली बढ़ना / Enlarged Spleen",        "choice", [
        (0,"अनुपस्थित / Absent"),(1,"<2-3 cm"),(2,"3-5 cm"),(3,">5 cm"),
    ]),
    ("Pica",            "मिट्टी खाना / Pica",                    "choice", PA),
    ("SOB",             "सांस फूलना / Shortness of Breath",      "choice", PA),
    ("EarNoise",        "कानों में आवाज़ / Ear Noise",           "choice", PA),
    ("Headache",        "सिरदर्द / Headache",                    "choice", PA),
    ("BrittleNails",    "नाखून टूटना / Brittle Nails",           "choice", PA),
    ("PoorSleep",       "नींद न आना / Poor Sleep",               "choice", [
        (0,"सामान्य / Normal"),(1,"हल्का / Mild"),(2,"मध्यम / Moderate"),(3,"गंभीर / Severe"),
    ]),
    ("Dizziness",       "चक्कर / Dizziness",                     "choice", PA),
    ("ColdIntolerance", "ठंड असहिष्णुता / Cold Intolerance",    "choice", PA),
    ("RLS",             "पैर बेचैनी / Restless Legs",            "choice", PA),
    ("BlueSclera",      "नीली आँखें / Blue Sclera",              "choice", PA),
    ("AngularStomatitis","मुँह के कोने / Angular Stomatitis",    "choice", PA),
    ("HairLoss",        "बालों का झड़ना / Hair Loss",            "choice", S03),
    ("Dysphagia",       "निगलने में दिक्कत / Dysphagia",        "choice", [
        (0,"अनुपस्थित / Absent"),(1,"कभी-कभी / Mild"),(2,"अक्सर / Moderate"),(3,"गंभीर / Severe"),
    ]),
    ("DrySkin",         "रूखी त्वचा / Dry Skin",                "choice", PA),
    ("Infections",      "बार-बार संक्रमण / Frequent Infections", "choice", PA),
    ("Appetite",        "भूख न लगना / Low Appetite",             "choice", PA),
    ("Bruising",        "आसानी से चोट / Easy Bruising",          "choice", PA),
]

# Colors
CLR_BG       = "#0f0f1a"
CLR_CARD     = "#1a1a2e"
CLR_BORDER   = "#2e2e50"
CLR_GREEN    = "#0F6E56"
CLR_GREEN2   = "#1a9070"
CLR_TEXT     = "#e0e0e0"
CLR_MUTED    = "#888888"
CLR_RED      = "#c0392b"
CLR_SUCCESS  = "#27ae60"
CLR_WHITE    = "#ffffff"

# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------
class IDAApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.lang      = "en"
        self.view_mode = "patient"   # "patient" or "doctor"
        self.field_widgets = {}
        self.last_row   = None       # for per-patient chart

        self.title("IDA AI Screening Tool")
        self.geometry("820x780")
        self.minsize(700, 600)
        self.configure(bg=CLR_BG)

        self._build_ui()

    # -----------------------------------------------------------------------
    @property
    def L(self):
        return LANG[self.lang]

    def toggle_lang(self):
        self.lang = "hi" if self.lang == "en" else "en"
        self._rebuild()

    def toggle_view(self):
        if self.view_mode == "patient":
            # Trying to enter Doctor View -> ask for password
            pw = simpledialog.askstring(
                "Doctor Login / डॉक्टर लॉगिन",
                "Enter doctor password / डॉक्टर पासवर्ड डालें:",
                show="*", parent=self
            )
            if pw is None:
                return  # user cancelled
            if pw != DOCTOR_PASSWORD:
                messagebox.showerror(
                    "Incorrect Password / गलत पासवर्ड",
                    "The password you entered is incorrect."
                )
                return
            self.view_mode = "doctor"
        else:
            self.view_mode = "patient"
        self._rebuild()

    def _rebuild(self):
        for w in self.winfo_children():
            w.destroy()
        self.field_widgets = {}
        self._build_ui()

    # -----------------------------------------------------------------------
    def _build_ui(self):
        L = self.L

        # ── Top bar ──────────────────────────────────────────────────────
        topbar = tk.Frame(self, bg=CLR_BG)
        topbar.pack(fill="x", padx=14, pady=(10, 0))

        tk.Button(topbar, text=L["lang_btn"], font=("Segoe UI", 9),
                  bg=CLR_CARD, fg=CLR_TEXT, relief="flat", bd=0,
                  padx=10, pady=5, cursor="hand2",
                  command=self.toggle_lang).pack(side="right", padx=(6,0))

        view_lbl = L["view_btn_doc"] if self.view_mode == "patient" else L["view_btn_pat"]
        tk.Button(topbar, text=view_lbl, font=("Segoe UI", 9),
                  bg=CLR_CARD, fg=CLR_TEXT, relief="flat", bd=0,
                  padx=10, pady=5, cursor="hand2",
                  command=self.toggle_view).pack(side="right")

        # ── Header ───────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=CLR_GREEN, pady=16)
        hdr.pack(fill="x", padx=14, pady=(8, 0))
        tk.Label(hdr, text="🩸  " + L["title"],
                 font=("Segoe UI", 15, "bold"),
                 fg=CLR_WHITE, bg=CLR_GREEN).pack()
        tk.Label(hdr, text=L["subtitle"],
                 font=("Segoe UI", 9),
                 fg="#c8ede5", bg=CLR_GREEN).pack()

        # ── Stats row (Doctor only) ───────────────────────────────────────
        if self.view_mode == "doctor":
            stats = tk.Frame(self, bg=CLR_BG)
            stats.pack(fill="x", padx=14, pady=(8,0))
            for lbl, val, sub in [
                ("Model Accuracy", "99.2%", "on test records"),
                ("AUC-ROC Score",  "0.999", "near-perfect"),
                ("Training Data",  "20,000","patient records"),
            ]:
                card = tk.Frame(stats, bg=CLR_CARD, bd=0, relief="flat",
                                padx=14, pady=10)
                card.pack(side="left", expand=True, fill="x", padx=4)
                tk.Label(card, text=lbl, font=("Segoe UI",8), fg=CLR_MUTED, bg=CLR_CARD).pack(anchor="w")
                tk.Label(card, text=val, font=("Segoe UI",18,"bold"), fg=CLR_WHITE, bg=CLR_CARD).pack(anchor="w")
                tk.Label(card, text=sub, font=("Segoe UI",8), fg=CLR_MUTED, bg=CLR_CARD).pack(anchor="w")

        # ── Scrollable form ───────────────────────────────────────────────
        outer = tk.Frame(self, bg=CLR_BG)
        outer.pack(fill="both", expand=True, padx=14, pady=8)

        canvas  = tk.Canvas(outer, bg=CLR_BG, bd=0, highlightthickness=0)
        vscroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)
        vscroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self.form = tk.Frame(canvas, bg=CLR_BG)
        canvas.create_window((0,0), window=self.form, anchor="nw")
        self.form.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(
            int(-1*(e.delta/120)), "units"))

        # Demographics
        self._section(L["demo_head"])
        for col, key, kind, opts in DEMO_FIELDS:
            self._field(col, L[key], kind, opts)

        # Symptoms
        self._section(L["symp_head"])
        for col, label, kind, opts in SYMPTOM_FIELDS:
            self._field(col, label, kind, opts)

        tk.Frame(self.form, height=10, bg=CLR_BG).pack()

        # ── Footer ───────────────────────────────────────────────────────
        footer = tk.Frame(self, bg=CLR_BG, pady=8)
        footer.pack(fill="x", padx=14)

        tk.Button(footer, text=L["predict_btn"],
                  font=("Segoe UI", 11, "bold"),
                  bg=CLR_GREEN, fg=CLR_WHITE, relief="flat",
                  padx=20, pady=10, cursor="hand2",
                  command=self.predict).pack(side="left")

        # Chart picker (Doctor only)
        if self.view_mode == "doctor":
            tk.Label(footer, text=L["chart_lbl"],
                     font=("Segoe UI",9), fg=CLR_MUTED,
                     bg=CLR_BG).pack(side="left", padx=(16,4))
            self.chart_var = tk.StringVar(value=L["chart_fi"])
            chart_dd = ttk.Combobox(footer, textvariable=self.chart_var,
                                    state="readonly", width=18,
                                    values=[L["chart_fi"], L["chart_shap"],
                                            L["chart_cmp"], L["chart_pat"]])
            chart_dd.pack(side="left", padx=2)
            tk.Button(footer, text=L["view_chart"],
                      font=("Segoe UI",9), bg=CLR_CARD, fg=CLR_TEXT,
                      relief="flat", padx=8, pady=5, cursor="hand2",
                      command=self.show_chart).pack(side="left", padx=4)

        # Result label
        self.result_var = tk.StringVar(
            value="Fill in the form and click 'Run AI Prediction'")
        self.result_lbl = tk.Label(self, textvariable=self.result_var,
                                   font=("Segoe UI", 13, "bold"),
                                   fg=CLR_MUTED, bg=CLR_BG, wraplength=760,
                                   justify="center", pady=10)
        self.result_lbl.pack(fill="x", padx=14)

        # Advice label
        self.advice_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self.advice_var,
                 font=("Segoe UI",9), fg=CLR_MUTED,
                 bg=CLR_BG, wraplength=760, justify="center").pack()

        # Disclaimer
        tk.Label(self, text=L["disclaimer"],
                 font=("Segoe UI",8), fg="#444466",
                 bg=CLR_BG, wraplength=760).pack(pady=(4,10))

    # -----------------------------------------------------------------------
    def _section(self, text):
        f = tk.Frame(self.form, bg="#16162a")
        f.pack(fill="x", pady=(10,4))
        tk.Label(f, text=text, font=("Segoe UI",9,"bold"),
                 fg=CLR_MUTED, bg="#16162a",
                 anchor="w").pack(fill="x", padx=10, pady=6)

    def _field(self, col_name, label, kind, options):
        row = tk.Frame(self.form, bg=CLR_BG)
        row.pack(fill="x", padx=10, pady=3)
        tk.Label(row, text=label, width=36, anchor="w",
                 font=("Segoe UI",9), fg=CLR_TEXT,
                 bg=CLR_BG).pack(side="left")
        if kind == "age":
            var = tk.StringVar(value="30")
            w   = ttk.Spinbox(row, from_=1, to=110, textvariable=var, width=12)
            w.pack(side="left")
            self.field_widgets[col_name] = ("age", var)
        else:
            display = [f"{c} - {t}" for c,t in options]
            var = tk.StringVar(value=display[0])
            w   = ttk.Combobox(row, textvariable=var, values=display,
                                state="readonly", width=44)
            w.pack(side="left")
            self.field_widgets[col_name] = ("choice", var)

    # -----------------------------------------------------------------------
    def _collect(self):
        vals = {}
        for col, (kind, var) in self.field_widgets.items():
            raw = var.get()
            if kind == "age":
                try:
                    v = int(raw)
                except ValueError:
                    raise ValueError(f"Age must be a number. Got: '{raw}'")
                if not 1 <= v <= 110:
                    raise ValueError("Age must be between 1 and 110.")
                vals[col] = v
            else:
                vals[col] = int(raw.split(" - ")[0])
        return vals

    def predict(self):
        L = self.L
        try:
            vals = self._collect()
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return

        row = pd.DataFrame([[vals[c] for c in FEATURE_COLUMNS]],
                            columns=FEATURE_COLUMNS)
        if USES_SCALED and scaler:
            row = pd.DataFrame(scaler.transform(row), columns=FEATURE_COLUMNS)
        self.last_row = row

        pred  = model.predict(row)[0]
        proba = model.predict_proba(row)[0][1]
        conf  = proba*100 if pred==1 else (1-proba)*100

        if pred == 1:
            self.result_var.set(f"{L['present']}   ({L['conf']}: {conf:.1f}%)")
            self.result_lbl.configure(fg=CLR_RED)
            self.advice_var.set(L["advice_p"])
        else:
            self.result_var.set(f"{L['absent']}   ({L['conf']}: {conf:.1f}%)")
            self.result_lbl.configure(fg=CLR_SUCCESS)
            self.advice_var.set(L["advice_a"])

    # -----------------------------------------------------------------------
    def show_chart(self):
        if not hasattr(self, "chart_var"):
            return
        L    = self.L
        sel  = self.chart_var.get()
        fmap = {
            L["chart_fi"]:   "feature_importance.png",
            L["chart_shap"]: "shap_summary.png",
            L["chart_cmp"]:  "model_comparison.png",
        }

        if sel == L["chart_pat"]:
            # Per-patient SHAP chart
            if self.last_row is None:
                messagebox.showinfo("No prediction",
                    "Run a prediction first, then view this chart.")
                return
            self._show_patient_chart()
            return

        path = fmap.get(sel)
        if not path or not os.path.exists(path):
            messagebox.showinfo("Not found",
                f"'{path}' not found. Run train_model.py to generate it.")
            return

        # Open chart in a new window
        win = tk.Toplevel(self)
        win.title(sel)
        win.configure(bg=CLR_BG)
        img  = Image.open(path)
        img.thumbnail((820, 700))
        photo = ImageTk.PhotoImage(img)
        lbl   = tk.Label(win, image=photo, bg=CLR_BG)
        lbl.image = photo
        lbl.pack(padx=10, pady=10)

    def _show_patient_chart(self):
        L = self.L
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(self.last_row)
            if isinstance(sv, list):
                sv_p = sv[1][0]
            elif sv.ndim == 3:
                sv_p = sv[0, :, 1]
            else:
                sv_p = sv[0]

            contrib = pd.Series(sv_p, index=FEATURE_COLUMNS)
            contrib = contrib[contrib != 0].sort_values()

            if len(contrib) == 0:
                messagebox.showinfo("Chart", "All contributions are zero — model is uncertain.")
                return

            colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in contrib.values]
            fig, ax = plt.subplots(figsize=(7, max(3, len(contrib)*0.35)))
            fig.patch.set_facecolor("#1a1a2e")
            ax.set_facecolor("#1a1a2e")
            contrib.plot(kind="barh", ax=ax, color=colors)
            ax.axvline(0, color="#555", linewidth=0.8, linestyle="--")
            ax.set_xlabel("Impact on prediction", color="white")
            ax.set_title(L["chart_pat"] + " — Symptom contributions", color="white")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#444")
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)

            win   = tk.Toplevel(self)
            win.title(L["chart_pat"])
            win.configure(bg=CLR_BG)
            img   = Image.open(buf)
            photo = ImageTk.PhotoImage(img)
            lbl   = tk.Label(win, image=photo, bg=CLR_BG)
            lbl.image = photo
            lbl.pack(padx=10, pady=10)

        except Exception as e:
            messagebox.showerror("Chart Error", str(e))


if __name__ == "__main__":
    app = IDAApp()
    app.mainloop()
