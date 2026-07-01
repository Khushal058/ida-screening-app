"""
web_app.py  —  IDA AI Screening Tool (Polished Dashboard Edition)
====================================================================
Premium, app-store-quality interface:
  - Card-based layout, icons, smooth visual hierarchy
  - Crisp outcome: "Iron Deficiency: Present / Absent"
  - Hindi / English toggle
  - Patient View (simple) vs Doctor View (full analytics)
  - Confidence gauge, symptom badges, patient explanation chart

Run train_model.py first, then:  python -m streamlit run web_app.py
"""

import os, io, warnings
warnings.filterwarnings("ignore")
import joblib, pandas as pd, numpy as np
import shap, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="IDA Screening | आयरन जांच", page_icon="🩸", layout="centered")

# ============================================================================
# STYLE
# ============================================================================
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 1rem; max-width: 760px;}

.hero {
    background: linear-gradient(120deg, #0F6E56 0%, #14856A 55%, #1a9e80 100%);
    border-radius: 18px; padding: 26px 28px; margin-bottom: 18px;
    box-shadow: 0 8px 24px rgba(15,110,86,0.25);
}
.hero-row { display:flex; align-items:center; gap:16px; }
.hero-icon { font-size:38px; line-height:1; }
.hero h1 { color:white; font-size:21px; font-weight:700; margin:0; }
.hero p  { color:rgba(255,255,255,0.85); font-size:13px; margin:5px 0 0 0; }

.stat-grid { display:flex; gap:10px; margin-bottom:18px; }
.stat-card {
    flex:1; background:var(--background-color,#1c1c2b); border:1px solid rgba(255,255,255,0.08);
    border-radius:14px; padding:14px 16px; position:relative; overflow:hidden;
}
.stat-card::before {
    content:""; position:absolute; top:0; left:0; width:4px; height:100%;
    background:linear-gradient(180deg,#1D9E75,#0F6E56);
}
.stat-icon { font-size:18px; margin-bottom:6px; opacity:0.85; }
.stat-label { font-size:11px; color:#9a9aae; font-weight:500; letter-spacing:.02em; }
.stat-value { font-size:24px; font-weight:800; color:#f2f2f5; margin:3px 0 1px 0; }
.stat-sub { font-size:10.5px; color:#6f6f85; }

.section-card {
    background:rgba(255,255,255,0.025); border:1px solid rgba(255,255,255,0.07);
    border-radius:16px; padding:18px 20px 8px 20px; margin-bottom:16px;
}
.section-title { display:flex; align-items:center; gap:8px; font-size:13px; font-weight:700;
    color:#cfcfe0; text-transform:uppercase; letter-spacing:.06em; margin-bottom:14px; }
.section-title .icon { font-size:16px; }

.toprow { display:flex; justify-content:flex-end; gap:8px; margin-bottom:10px; }

.result-box {
    border-radius:18px; padding:22px 24px; margin-top:18px;
    display:flex; align-items:center; gap:18px;
    animation: fadeIn .35s ease-in;
}
@keyframes fadeIn { from{opacity:0; transform:translateY(6px);} to{opacity:1; transform:translateY(0);} }
.result-box.present { background:linear-gradient(120deg,#2d1212,#3a1414); border:1px solid #c0392b55; }
.result-box.absent  { background:linear-gradient(120deg,#0e2a1c,#0f3322); border:1px solid #27ae6055; }
.result-icon-wrap {
    width:56px; height:56px; border-radius:50%; display:flex; align-items:center; justify-content:center;
    font-size:26px; flex-shrink:0;
}
.result-box.present .result-icon-wrap { background:rgba(231,76,60,0.18); }
.result-box.absent  .result-icon-wrap { background:rgba(46,204,113,0.18); }
.result-title { font-size:19px; font-weight:800; margin:0; }
.result-box.present .result-title { color:#ff6b5b; }
.result-box.absent  .result-title { color:#3ddc84; }
.result-advice { font-size:12.5px; color:#b8b8c8; margin-top:6px; }

.gauge-wrap { margin-top:10px; }
.gauge-track { width:100%; height:8px; border-radius:99px; background:rgba(255,255,255,0.08); overflow:hidden; }
.gauge-fill  { height:100%; border-radius:99px; transition: width .6s ease; }
.gauge-fill.present { background:linear-gradient(90deg,#e74c3c,#ff8a75); }
.gauge-fill.absent  { background:linear-gradient(90deg,#27ae60,#5be08c); }
.gauge-label { font-size:11px; color:#8a8aa0; margin-top:5px; }

.disclaimer-box {
    text-align:center; font-size:11.5px; color:#6f6f85; margin-top:26px;
    padding:12px; border-top:1px solid rgba(255,255,255,0.07);
}

div[data-testid="stForm"] { border:none; padding:0; }
.stButton button {
    border-radius:12px !important; font-weight:700 !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL
# ============================================================================
@st.cache_resource
def load_model():
    for p in ["ida_model.joblib", "feature_list.joblib"]:
        if not os.path.exists(p):
            st.error(f"Missing {p} — run train_model.py first.")
            st.stop()
    m      = joblib.load("ida_model.joblib")
    feats  = joblib.load("feature_list.joblib")
    scaler = joblib.load("scaler.joblib")     if os.path.exists("scaler.joblib")     else None
    meta   = joblib.load("model_meta.joblib") if os.path.exists("model_meta.joblib") else {"name":"Model","uses_scaled_input":False}
    return m, feats, scaler, meta

model, FEATURE_COLUMNS, scaler, model_meta = load_model()
USES_SCALED = model_meta.get("uses_scaled_input", False)

# ============================================================================
# LANGUAGE
# ============================================================================
LANG = {
"en": {
    "title":"IDA Screening", "subtitle":"AI diagnostic tool · Signs & symptoms only · No blood test required",
    "lang_btn":"🇮🇳 हिंदी", "view_doc":"🩺 Doctor view", "view_pat":"👤 Patient view",
    "acc":"Accuracy","auc":"AUC-ROC","data":"Training data",
    "demo":"Patient information","symp":"Signs & symptoms","btn":"Run AI prediction",
    "present":"Iron Deficiency: Present", "absent":"Iron Deficiency: Absent",
    "conf":"Confidence", "advice_p":"Consult a doctor and confirm with a blood test.",
    "advice_a":"No iron deficiency indicated. See a doctor if symptoms persist.",
    "disclaimer":"Screening aid only — not a substitute for clinical diagnosis. Built from a research dataset for academic purposes.",
    "why":"Why this result", "why_cap":"Red pushes toward IDA · Green pushes away · Longer bar = stronger effect",
    "fi":"Feature importance","shap":"SHAP summary","cmp":"Model comparison",
    "age":"Age (years)","gender":"Gender","address":"Address","edu":"Education","occ":"Occupation","income":"Monthly income (₹)",
},
"hi": {
    "title":"IDA जांच", "subtitle":"AI निदान उपकरण · केवल लक्षणों पर आधारित · रक्त परीक्षण आवश्यक नहीं",
    "lang_btn":"🇬🇧 English", "view_doc":"🩺 डॉक्टर दृश्य", "view_pat":"👤 मरीज़ दृश्य",
    "acc":"सटीकता","auc":"AUC-ROC","data":"प्रशिक्षण डेटा",
    "demo":"मरीज़ की जानकारी","symp":"संकेत और लक्षण","btn":"AI जांच करें",
    "present":"आयरन की कमी: उपस्थित", "absent":"आयरन की कमी: अनुपस्थित",
    "conf":"विश्वास", "advice_p":"कृपया डॉक्टर से मिलें और रक्त परीक्षण से पुष्टि करें।",
    "advice_a":"आयरन की कमी नहीं दिखती। लक्षण बने रहने पर डॉक्टर से मिलें।",
    "disclaimer":"यह केवल जांच सहायक है — चिकित्सीय निदान का विकल्प नहीं। शोध डेटा पर आधारित शैक्षणिक परियोजना।",
    "why":"यह परिणाम क्यों", "why_cap":"लाल = IDA की ओर · हरा = IDA से दूर · लंबा बार = अधिक प्रभाव",
    "fi":"फीचर महत्व","shap":"SHAP सारांश","cmp":"मॉडल तुलना",
    "age":"आयु (वर्ष)","gender":"लिंग","address":"पता","edu":"शिक्षा","occ":"व्यवसाय","income":"मासिक आय (₹)",
}}

DEMO_FIELDS = [
    ("Age","age","age",None),
    ("Gender","gender","choice",[(1,"पुरुष / Male"),(2,"महिला / Female")]),
    ("Address","address","choice",[(1,"शहरी / Urban"),(2,"ग्रामीण / Rural")]),
    ("Education","edu","choice",[(1,"अनपढ़ / Illiterate"),(2,"प्राथमिक / Primary"),(3,"माध्यमिक / Middle"),
        (4,"हाई स्कूल / High School"),(5,"इंटर / Intermediate"),(6,"स्नातक / Graduate"),(7,"स्नातकोत्तर / Post-Grad")]),
    ("Occupation","occ","choice",[(1,"बेरोज़गार / Unemployed"),(2,"साधारण / Elementary"),(3,"ऑपरेटर / Operator"),
        (4,"दस्तकार / Craft"),(5,"कृषि / Agricultural"),(6,"दुकानदार / Shop"),(7,"लिपिक / Clerk"),
        (8,"तकनीशियन / Technician"),(9,"पेशेवर / Professional"),(10,"प्रबंधक / Manager")]),
    ("Income","income","choice",[(1,"₹7,988 से कम / Less than ₹7,988"),(2,"₹7,989–₹23,869"),(3,"₹23,870–₹39,829"),
        (4,"₹39,830–₹59,794"),(5,"₹59,795–₹79,755"),(6,"₹79,756–₹1,59,585"),(7,"₹1,59,586+")]),
]
S03=[(0,"अनुपस्थित / Absent"),(1,"हल्का / Mild"),(2,"मध्यम / Moderate"),(3,"गंभीर / Severe")]
PA =[(0,"अनुपस्थित / Absent"),(1,"उपस्थित / Present")]
SYMPTOM_FIELDS = [
    ("Paleness","असामान्य पीलापन / Paleness","choice",[
        (0,"अनुपस्थित / Absent"),(1,"हल्का / Mild"),(2,"मध्यम / Moderate"),(3,"गंभीर / Severe")]),
    ("Irritability","चिड़चिड़ापन / Irritability","choice",PA),
    ("Fatigue","थकान / Fatigue","choice",[
        (0,"अनुपस्थित / Absent"),(1,"हल्की / Mild"),(2,"मध्यम / Moderate"),(3,"गंभीर / Severe")]),
    ("Tachycardia","तेज़ धड़कन / Tachycardia","choice",[
        (0,"अनुपस्थित / Absent"),(1,"हल्का / Mild (101-110)"),(2,"मध्यम / Moderate (111-120)"),(3,"गंभीर / Severe (>120)")]),
    ("Tongue","जीभ सूजन / Sore Tongue","choice",S03),
    ("Spleen","तिल्ली / Spleen","choice",[
        (0,"अनुपस्थित / Absent"),(1,"हल्का / Mild (<2-3cm)"),(2,"मध्यम / Moderate (3-5cm)"),(3,"गंभीर / Severe (>5cm)")]),
    ("Pica","मिट्टी खाना / Pica","choice",PA),
    ("SOB","सांस फूलना / Breathlessness","choice",PA),
    ("EarNoise","कान आवाज़ / Ear Noise","choice",PA),
    ("Headache","सिरदर्द / Headache","choice",PA),
    ("BrittleNails","नाखून / Brittle Nails","choice",PA),
    ("PoorSleep","नींद / Poor Sleep","choice",[
        (0,"सामान्य / Normal"),(1,"हल्का / Mild"),(2,"मध्यम / Moderate"),(3,"गंभीर / Severe")]),
    ("Dizziness","चक्कर / Dizziness","choice",PA),
    ("ColdIntolerance","ठंड / Cold Intolerance","choice",PA),
    ("RLS","पैर बेचैनी / Restless Legs","choice",PA),
    ("BlueSclera","नीली आँखें / Blue Sclera","choice",PA),
    ("AngularStomatitis","मुँह कोने / Mouth Corners","choice",PA),
    ("HairLoss","बाल झड़ना / Hair Loss","choice",S03),
    ("Dysphagia","निगलना / Dysphagia","choice",[
        (0,"अनुपस्थित / Absent"),(1,"कभी-कभी / Mild"),(2,"अक्सर / Moderate"),(3,"गंभीर / Severe")]),
    ("DrySkin","रूखी त्वचा / Dry Skin","choice",PA),
    ("Infections","संक्रमण / Infections","choice",PA),
    ("Appetite","भूख / Low Appetite","choice",PA),
    ("Bruising","चोट / Bruising","choice",PA),
]

if "lang" not in st.session_state: st.session_state.lang = "en"
if "view" not in st.session_state: st.session_state.view = "patient"
if "doctor_unlocked" not in st.session_state: st.session_state.doctor_unlocked = False
if "show_pw_box" not in st.session_state: st.session_state.show_pw_box = False
L = LANG[st.session_state.lang]

try:
    DOCTOR_PASSWORD = st.secrets["DOCTOR_PASSWORD"]
except Exception:
    DOCTOR_PASSWORD = "ssims2026"

# top controls
_, c2, c3 = st.columns([3,1,1])
with c2:
    if st.button(L["lang_btn"], use_container_width=True):
        st.session_state.lang = "hi" if st.session_state.lang=="en" else "en"; st.rerun()
with c3:
    if st.session_state.view == "patient":
        # Not in doctor view yet -> button either unlocks (if already unlocked once) or asks for password
        if st.button(L["view_doc"], use_container_width=True):
            if st.session_state.doctor_unlocked:
                st.session_state.view = "doctor"
            else:
                st.session_state.show_pw_box = True
            st.rerun()
    else:
        if st.button(L["view_pat"], use_container_width=True):
            st.session_state.view = "patient"; st.rerun()

# Password prompt (shown only when Doctor View was requested but not yet unlocked)
if st.session_state.show_pw_box and not st.session_state.doctor_unlocked:
    with st.form("doctor_login", clear_on_submit=False):
        pw_col1, pw_col2 = st.columns([3,1])
        with pw_col1:
            entered_pw = st.text_input("🔒 Doctor password / डॉक्टर पासवर्ड", type="password", label_visibility="collapsed", placeholder="Enter doctor password")
        with pw_col2:
            pw_submit = st.form_submit_button("Unlock", use_container_width=True)
        if pw_submit:
            if entered_pw == DOCTOR_PASSWORD:
                st.session_state.doctor_unlocked = True
                st.session_state.view = "doctor"
                st.session_state.show_pw_box = False
                st.rerun()
            else:
                st.error("Incorrect password / गलत पासवर्ड")

IS_DOCTOR = st.session_state.view == "doctor" and st.session_state.doctor_unlocked

# hero
st.markdown(f"""
<div class="hero"><div class="hero-row">
  <div class="hero-icon">🩸</div>
  <div><h1>{L['title']}</h1><p>{L['subtitle']}</p></div>
</div></div>
""", unsafe_allow_html=True)

# stats (doctor only)
if IS_DOCTOR:
    st.markdown(f"""
    <div class="stat-grid">
      <div class="stat-card"><div class="stat-icon">🎯</div><div class="stat-label">{L['acc']}</div>
        <div class="stat-value">99.2%</div><div class="stat-sub">test records</div></div>
      <div class="stat-card"><div class="stat-icon">📈</div><div class="stat-label">{L['auc']}</div>
        <div class="stat-value">0.999</div><div class="stat-sub">near-perfect</div></div>
      <div class="stat-card"><div class="stat-icon">🗂️</div><div class="stat-label">{L['data']}</div>
        <div class="stat-value">20,000</div><div class="stat-sub">patients</div></div>
    </div>
    """, unsafe_allow_html=True)

# form
values = {}
with st.form("ida_form"):
    st.markdown(f'<div class="section-card"><div class="section-title"><span class="icon">🧍</span>{L["demo"]}</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i,(col_name,key,kind,opts) in enumerate(DEMO_FIELDS):
        with cols[i%2]:
            label = L[key]
            if kind=="age":
                values[col_name] = st.number_input(label, 1, 110, 30, 1)
            else:
                lab = [f"{c} - {t}" for c,t in opts]
                ch = st.selectbox(label, lab, key=f"d_{col_name}")
                values[col_name] = int(ch.split(" - ")[0])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="section-card"><div class="section-title"><span class="icon">🩺</span>{L["symp"]}</div>', unsafe_allow_html=True)
    cols2 = st.columns(2)
    for i,(col_name,label,kind,opts) in enumerate(SYMPTOM_FIELDS):
        with cols2[i%2]:
            lab = [f"{c} - {t}" for c,t in opts]
            ch = st.selectbox(label, lab, key=f"s_{col_name}")
            values[col_name] = int(ch.split(" - ")[0])
    st.markdown('</div>', unsafe_allow_html=True)

    submitted = st.form_submit_button("🔍  " + L["btn"], use_container_width=True, type="primary")

# prediction
if submitted:
    row = pd.DataFrame([[values[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    if USES_SCALED and scaler:
        row = pd.DataFrame(scaler.transform(row), columns=FEATURE_COLUMNS)
    pred  = model.predict(row)[0]
    proba = model.predict_proba(row)[0][1]
    conf  = proba*100 if pred==1 else (1-proba)*100
    cls   = "present" if pred==1 else "absent"
    icon  = "⚠️" if pred==1 else "✅"
    title = L["present"] if pred==1 else L["absent"]
    advice= L["advice_p"] if pred==1 else L["advice_a"]

    st.markdown(f"""
    <div class="result-box {cls}">
      <div class="result-icon-wrap">{icon}</div>
      <div style="flex:1">
        <p class="result-title">{title}</p>
        <p class="result-advice">{advice}</p>
        <div class="gauge-wrap">
          <div class="gauge-track"><div class="gauge-fill {cls}" style="width:{conf:.0f}%"></div></div>
          <div class="gauge-label">{L['conf']}: {conf:.1f}%</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if IS_DOCTOR:
        try:
            with st.expander("🔎 " + L["why"], expanded=True):
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(row)
                sv_p = sv[1][0] if isinstance(sv,list) else (sv[0,:,1] if sv.ndim==3 else sv[0])
                contrib = pd.Series(sv_p, index=FEATURE_COLUMNS)
                contrib = contrib[contrib != 0].sort_values()
                if len(contrib):
                    colors = ["#e74c3c" if v>0 else "#2ecc71" for v in contrib.values]
                    fig, ax = plt.subplots(figsize=(7, max(3,len(contrib)*0.32)))
                    fig.patch.set_facecolor("#15151f"); ax.set_facecolor("#15151f")
                    contrib.plot(kind="barh", ax=ax, color=colors)
                    ax.axvline(0, color="#555", lw=0.8, ls="--")
                    ax.tick_params(colors="#ccc", labelsize=9)
                    ax.set_xlabel("Impact", color="#ccc")
                    ax.spines[:].set_color("#444")
                    plt.tight_layout()
                    buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=130, bbox_inches="tight"); plt.close(fig); buf.seek(0)
                    st.image(buf, use_container_width=True)
                st.caption(L["why_cap"])
        except Exception as e:
            st.info(f"Chart unavailable: {e}")

def img_bytes(p):
    with open(p,"rb") as f: return f.read()

if IS_DOCTOR:
    st.markdown(f'<div class="section-card"><div class="section-title"><span class="icon">📊</span>Model analytics</div>', unsafe_allow_html=True)
    if os.path.exists("feature_importance.png"):
        with st.expander(L["fi"]): st.image(img_bytes("feature_importance.png"), use_container_width=True)
    if os.path.exists("shap_summary.png"):
        with st.expander(L["shap"]): st.image(img_bytes("shap_summary.png"), use_container_width=True)
    if os.path.exists("model_comparison.png"):
        with st.expander(L["cmp"]): st.image(img_bytes("model_comparison.png"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f'<div class="disclaimer-box">{L["disclaimer"]}</div>', unsafe_allow_html=True)
