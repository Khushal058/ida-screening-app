# Iron Deficiency Anaemia – AI Screening Tool

A full AI model + GUI (desktop and web) that predicts whether a patient
likely has Iron Deficiency Anaemia (IDA), using **only signs, symptoms,
and basic demographics — no blood test required**. Built from the
research dataset and proposal provided by Dr. Ruchita Dixit, Dr. Amol
Patil, and Dr. Smita Selot (SSIMS, Bhilai).

## How this matches the research proposal

The proposal's "Methodology for developing AI" section lists 8 steps.
This project implements every step that is actually a coding/AI task:

| Proposal step | Implemented as |
|---|---|
| Data Preprocessing | `train_model.py` Step 2 — missing values, duplicates, encoding check |
| Feature Engineering | `train_model.py` Step 3 — selects only demographics + symptoms, excludes lab values |
| Model Selection | `train_model.py` Step 5 — Decision Tree, Random Forest, SVM, and Neural Network are all trained and compared; best one is picked automatically |
| Model Training (train/val/test split + tuning) | `train_model.py` Steps 4 & 6 — 70/15/15 split + GridSearchCV hyperparameter tuning |
| Evaluation & Validation | `train_model.py` Steps 7 & 8 — accuracy, precision, recall, F1, AUC-ROC, plus 5-fold cross-validation |
| User Interface (mobile/web) | `gui_app.py` (desktop) and `web_app.py` (browser — also works on phones) |
| Explainability (feature importance + SHAP) | `train_model.py` Step 9 — both a feature-importance chart and a SHAP summary chart are generated |
| Clinical Validation | **Not included** — this requires real doctors reviewing live predictions, a research step for your faculty/co-investigators, not a coding task |
| Deployment into hospital EHR | **Not included** — this requires hospital IT infrastructure and compliance approval, outside the scope of a student project |

## Why no lab values are used as model input

The proposal's AIM is to diagnose IDA *based on signs and symptoms only*,
because many patients avoid blood tests. So although the dataset includes
lab values (Hemoglobin, Serum Iron, TIBC, Transferrin Saturation,
Ferritin), the AI model deliberately does **not** use them as inputs —
only age, gender, address, education, occupation, income, and 22
observable signs/symptoms are used. The lab values were only used
historically (by the doctors) to confirm the original diagnosis labels.

## What's in this folder

| File | What it is |
|---|---|
| `10K_individual_with_Iron_deficiency.xlsx` | Original data: 10,000 patients with IDA |
| `10K_individual_without_Iron_deficiency.xlsx` | Original data: 10,000 patients without IDA |
| `combined_dataset.csv` | Both files merged and shuffled (20,000 records) |
| `train_model.py` | Run this FIRST. Builds and compares 4 AI models, tunes and validates the best one, and saves it. |
| `gui_app.py` | Desktop GUI (Tkinter) — run AFTER training. |
| `web_app.py` | Web/mobile-friendly GUI (Streamlit) — run AFTER training. |
| `ida_model.joblib` | The final, best-performing trained model |
| `feature_list.joblib` | Exact list/order of inputs the model expects |
| `scaler.joblib` | Feature scaler (used only if the winning model needs scaled inputs) |
| `model_meta.joblib` | Records which algorithm won and whether it needs scaled inputs |
| `feature_importance.png` | Chart: which symptoms matter most |
| `shap_summary.png` | Chart: SHAP explainability — how each symptom pushes the prediction up or down |
| `model_comparison.png` | Chart: how all 4 candidate algorithms compared |
| `model_report.txt` | Full step-by-step methodology + performance report |
| `requirements.txt` | Python libraries needed |

## How to run it (step by step)

**1. Open this folder in VS Code.**

**2. Open a terminal in VS Code** (Terminal menu → New Terminal).

**3. Install the required libraries** (copy-paste, run once):
```
pip install -r requirements.txt
```
If `pip` isn't recognized on Windows, try `python -m pip install -r requirements.txt`.

**4. Train the model** (copy-paste, run once — takes about 1-2 minutes,
since it now trains and compares 4 different algorithms plus tuning and
cross-validation):
```
python train_model.py
```
You'll see clearly labelled progress for every methodology step,
finishing with "All done!" and a final accuracy summary.

**5. Launch the app — choose ONE of the two options below:**

**Option A — Desktop app:**
```
python gui_app.py
```
A window opens. Fill in the dropdowns, click **Predict Diagnosis**. Use
the chart dropdown + "View Chart" button to see Feature Importance, SHAP,
or Model Comparison charts.

**Option B — Web app (also works on phones):**
```
streamlit run web_app.py
```
Opens automatically at `http://localhost:8501`. To use it on your phone
(same Wi-Fi network), check the terminal for a `Network URL` line and
open that address in your phone's browser. All three explainability
charts are available in expandable sections below the prediction.

**Note for Linux users:** if `gui_app.py` gives a `ModuleNotFoundError:
No module named 'tkinter'`, run `sudo apt install python3-tk` first.
(Not needed on Windows/Mac.)

## How the model works (for a viva/demo)

- **Algorithms compared**: Decision Tree, Random Forest, Support Vector
  Machine, and Neural Network (MLP) — all four are trained and evaluated
  on a held-out validation set; the best performer is automatically
  selected, then fine-tuned with hyperparameter search (GridSearchCV).
- **Data split**: 70% training / 15% validation / 15% test — the test set
  is never touched until the very final evaluation, so reported numbers
  are honest, not optimistic.
- **Robustness check**: 5-fold cross-validation confirms the model
  performs consistently across different data splits, not just one lucky
  split.
- **Performance** (see `model_report.txt` for full numbers): accuracy
  ≈ 99%, AUC-ROC ≈ 0.999 on the untouched test set.
- **Explainability — two methods**:
  - *Feature importance* (`feature_importance.png`) — a simple ranking of
    which symptoms the model relies on most.
  - *SHAP* (`shap_summary.png`) — a more advanced, per-prediction
    explanation showing not just which symptoms matter, but whether a
    high or low value of each one pushes the prediction toward or away
    from IDA. This directly matches the proposal's request for SHAP-based
    explainability.
  - Both agree: **Fatigue** and **Paleness** are by far the strongest
    predictors, which matches real-world clinical understanding of IDA —
    a good sanity check that the model learned something medically
    sensible rather than a random pattern.

## Important note

This tool is built for an academic/research project and is **not** a
certified medical device. It should always be explained to viewers as a
screening aid, not a replacement for clinical diagnosis or lab testing.
The "Clinical Validation" and "Deployment" stages described in the
proposal are real-world research steps that would need actual doctors
and hospital infrastructure — they are intentionally outside the scope of
this coded deliverable.

## Doctor View password — IMPORTANT security note

Both `web_app.py` and `gui_app.py` protect Doctor View (charts, model
stats, per-patient explanations) behind a password. The default password
is `ssims2026`.

**⚠️ If your GitHub repository is Public, do NOT rely on the password
written in the code** — anyone can view your source code on GitHub and
read the password directly. For the desktop app (`gui_app.py`) this is
lower risk since it usually isn't shared publicly, but for the deployed
web app you should set a real secret:

**On Streamlit Community Cloud:**
1. Go to your app's page on share.streamlit.io
2. Click the **⋮** menu (top right) → **Settings** → **Secrets**
3. Add this, replacing `yourpassword` with a real password only you and
   the doctors know:
   ```
   DOCTOR_PASSWORD = "yourpassword"
   ```
4. Save. The app will automatically use this instead of the default
   `ssims2026` fallback in the code — and this secret is **never**
   visible in your public GitHub repository.

**Running locally:** create a file `.streamlit/secrets.toml` (this file
should NOT be uploaded to GitHub — add it to `.gitignore`) with the same
`DOCTOR_PASSWORD = "yourpassword"` line.

