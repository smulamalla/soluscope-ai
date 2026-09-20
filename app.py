import streamlit as st
import numpy as np
import pandas as pd
import pickle
import base64
from io import BytesIO
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors, Lipinski

DATASET_INFO = {
    "organization": "MoleculeNet Benchmark Suite",
    "title": "ESOL (Delaney) Aqueous Solubility Dataset",
}

MODEL_INFO = {
    "name": "Support Vector Regressor (Tuned)",
    "hyperparameters": "kernel=rbf, C=100, gamma=scale",
    "features": "1024-bit ECFP (Morgan) fingerprints, radius 2",
    "test_r2": "0.730",
}

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded');

:root {
  --ink: #1A1A2E;
  --paper: #F5FAFE;
  --panel: #EAF3FB;
  --blue: #1B6FC9;
  --blue-deep: #124A85;
  --blue-soft: #DCEEFB;
  --blue-soft-hover: #C3E0F7;
  --safe: #1E8A5F;
  --safe-soft: #DCF3E7;
  --amber: #EFA23B;
  --amber-soft: #FBEACB;
  --danger: #C81E3A;
  --danger-soft: #F6D2D9;
  --line: #D9E4EF;
  --muted: #7C8A93;
  --subtitle: #1F262C;
}

.material-symbols-rounded {
  font-family: 'Material Symbols Rounded';
  font-weight: normal; font-style: normal; font-size: 18px; line-height: 1; vertical-align: middle;
}

.sidebar-footer {
    margin-top: 1.6rem; padding-top: 0.9rem; border-top: 1px solid var(--line);
    font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem; color: var(--muted);
    text-align: center; line-height: 1.5;
}

.stApp {
    background-color: #FFFFFF;
    font-family: 'IBM Plex Sans', sans-serif;
}

.block-container, [data-testid="stAppViewBlockContainer"], [data-testid="stMainBlockContainer"] {
    padding-left: 8% !important; padding-right: 8% !important;
}

h1, h2, h3 { font-family: 'IBM Plex Serif', serif; color: var(--ink) !important; }
[data-testid="stCaptionContainer"] { font-family: 'IBM Plex Sans', sans-serif; color: var(--subtitle); font-weight: 500; }

[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] label {
    font-family: 'IBM Plex Mono', monospace !important; font-size: 0.78rem !important;
    color: var(--muted) !important; text-transform: uppercase; letter-spacing: 0.04em;
}

[data-testid="stExpander"] summary {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: var(--blue-deep);
}

.stButton button {
    background: var(--blue-soft); color: var(--blue-deep); border: 1px solid var(--blue);
    border-radius: 10px; font-weight: 500; padding: 0.55rem 0.9rem; min-height: 58px;
    transition: all 0.18s ease;
}
.stButton button:hover {
    background: var(--blue-soft-hover); border-color: var(--blue-deep); color: var(--blue-deep);
}

[data-testid="stTextInput"] input {
    border-radius: 8px !important; border: 1px solid var(--line) !important; background: #FFFFFF !important;
    transition: border-color 0.18s ease, box-shadow 0.18s ease;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--blue) !important; box-shadow: 0 0 0 1px var(--blue-soft) !important;
}

.result-card { transition: transform 0.18s ease, box-shadow 0.18s ease; }
.result-card:hover { transform: translateY(-3px); box-shadow: 0 6px 14px rgba(26,26,46,0.10); }

.info-card { transition: transform 0.18s ease, box-shadow 0.18s ease; }
.info-card:hover { transform: translateY(-2px); box-shadow: 0 6px 14px rgba(26,26,46,0.08); }

.st-key-molecule_input_box {
    background: #FFFFFF; border: 1px solid var(--line); border-radius: 12px;
    padding: 1.3rem 1.4rem; box-shadow: 0 1px 3px rgba(26,26,46,0.05);
}

.st-key-molecule_input_box [data-testid="stTextInput"] {
    max-width: 500px;
    margin-left: auto;
    margin-right: auto;
}

header[data-testid="stHeader"] { background: transparent !important; }
</style>
"""

THEME_CSS = THEME_CSS.replace("</style>", """
@keyframes drift1 { 0% { transform: translate(0,0) scale(1); opacity: 0; } 8% { opacity: 1; } 100% { transform: translate(540px, -9px) scale(0.3); opacity: 0; } }
@keyframes drift2 { 0% { transform: translate(0,0) scale(1); opacity: 0; } 8% { opacity: 1; } 100% { transform: translate(540px, 7px) scale(0.3); opacity: 0; } }
@keyframes drift3 { 0% { transform: translate(0,0) scale(1); opacity: 0; } 8% { opacity: 1; } 100% { transform: translate(540px, -4px) scale(0.3); opacity: 0; } }
@keyframes drift4 { 0% { transform: translate(0,0) scale(1); opacity: 0; } 8% { opacity: 1; } 100% { transform: translate(540px, 9px) scale(0.3); opacity: 0; } }
@keyframes fadeInUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
.particle {
  opacity: 0; transform-box: fill-box; transform-origin: center;
  animation-duration: 4.5s; animation-timing-function: linear; animation-iteration-count: infinite;
}
.particle-1 { animation-name: drift1; animation-delay: 0s; }
.particle-2 { animation-name: drift2; animation-delay: 1.1s; }
.particle-3 { animation-name: drift3; animation-delay: 2.2s; }
.particle-4 { animation-name: drift4; animation-delay: 3.3s; }
.animate-in { animation: fadeInUp 0.45s ease-out; }
</style>""")

MOLECULE_DIVIDER = """
<div style="margin: 0.1rem 0 1.1rem 0; position:relative;">
<svg width="100%" height="42" viewBox="0 0 600 42" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M 8 12 H 600 V 20 Q 600 28 592 28 H 0 V 20 Q 0 12 8 12 Z" fill="#DCEEFB"/>
  <path d="M0,24 Q30,20 60,24 T120,24 T180,24 T240,24 T300,24 T360,24 T420,24 T480,24 T540,24 T600,24"
    fill="none" stroke="#BBDCF2" stroke-width="1.5"/>
  <circle class="particle particle-1" cx="6" cy="20" r="3.5" fill="#1B6FC9"/>
  <circle class="particle particle-2" cx="6" cy="20" r="3" fill="#1B6FC9"/>
  <circle class="particle particle-3" cx="6" cy="20" r="4" fill="#1B6FC9"/>
  <circle class="particle particle-4" cx="6" cy="20" r="3" fill="#1B6FC9"/>
</svg>
<div style="display:flex; justify-content:space-between; font-family:'IBM Plex Mono',monospace;
            font-size:0.65rem; color:#7C8A93; letter-spacing:0.05em; margin-top:2px;">
  <span>EXPLAINABLE AI &middot; RDKIT &middot; SCIKIT-LEARN</span>
  <span>EDUCATIONAL DEMO &middot; NOT EXPERIMENTALLY VALIDATED</span>
</div>
</div>
"""

def section_header(icon, text):
    """Renders an h2 section title with a themed icon."""
    st.markdown(
        f"""<h2 class="animate-in" style="font-size:1.45rem; margin-top:2.4rem; margin-bottom:0.3rem;">
        <span class="material-symbols-rounded" style="color:var(--blue); font-size:26px;
        margin-right:0.8rem !important; vertical-align:middle;">{icon}</span>{text}</h2>""",
        unsafe_allow_html=True,
    )


def image_to_base64(img):
    """Converts a PIL image (from RDKit's Draw functions) to a base64 PNG string,
    so it can be embedded directly inside an HTML card instead of rendered as a
    separate, unnested Streamlit element."""
    if isinstance(img, (bytes, bytearray)):
        return base64.b64encode(img).decode()
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def solubility_tier(log_s):
    """Maps a log-solubility value to a (label, accent color, background color) tuple."""
    if log_s > -1:
        return "Highly Soluble", "var(--safe)", "var(--safe-soft)"
    elif log_s > -3:
        return "Moderately Soluble", "var(--amber)", "var(--amber-soft)"
    else:
        return "Poorly Soluble", "var(--danger)", "var(--danger-soft)"


def solubility_card(label, log_s):
    """Renders a hoverable, full-width card showing the predicted solubility and its tier badge."""
    level, color, bg = solubility_tier(log_s)
    st.markdown(
        f"""<div class="result-card animate-in" style="background:#FFFFFF; border:1px solid var(--line);
        border-left:4px solid {color}; border-radius:10px; padding:1.1rem 1.4rem;
        box-shadow:0 1px 2px rgba(26,26,46,0.04); margin-bottom:0.6rem;">
        <div style="font-family:'IBM Plex Mono', monospace; font-size:0.72rem; color:var(--muted);
        text-transform:uppercase; letter-spacing:0.05em;">{label}</div>
        <div style="font-family:'IBM Plex Serif', serif; font-size:2.2rem; font-weight:600; color:var(--ink);">
        {log_s:.2f} <span style="font-size:1rem; color:var(--muted);">log(mol/L)</span></div>
        <div style="display:inline-block; background:{bg}; color:{color}; border:1px solid {color};
        border-radius:20px; padding:0.3rem 0.8rem; font-family:'IBM Plex Mono', monospace; font-weight:600;
        font-size:0.78rem; margin-top:0.4rem;">{level.upper()}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def get_molecular_properties(mol):
    """Computes standard physicochemical descriptors plus a Lipinski Rule of Five check."""
    violations = 0
    if Descriptors.MolWt(mol) > 500: violations += 1
    if Descriptors.MolLogP(mol) > 5: violations += 1
    if Lipinski.NumHDonors(mol) > 5: violations += 1
    if Lipinski.NumHAcceptors(mol) > 10: violations += 1
    return {
        "Molecular Weight": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "H-Bond Donors": Lipinski.NumHDonors(mol),
        "H-Bond Acceptors": Lipinski.NumHAcceptors(mol),
        "Rotatable Bonds": Descriptors.NumRotatableBonds(mol),
        "TPSA": round(Descriptors.TPSA(mol), 2),
        "Lipinski Violations": violations,
    }


def predict_solubility(model, transformers, features):
    """Runs the tuned SVR on a fingerprint vector and returns real-unit log solubility."""
    raw_pred = model.predict(features).reshape(-1, 1)
    return transformers[0].untransform(raw_pred)[0][0]


MAX_SMILES_LENGTH = 200
MAX_REASONABLE_ATOMS = 60  # ESOL is mostly small, drug-like molecules

def validate_smiles(smiles):
    """Validates a SMILES string. Returns (mol, error, warning):
    error blocks the prediction entirely; warning lets it proceed but flags
    that the input looks unlike anything the model was trained on."""
    smiles = smiles.strip()
    if not smiles:
        return None, "Please enter a SMILES string.", None
    if len(smiles) > MAX_SMILES_LENGTH:
        return None, f"SMILES string is too long (max {MAX_SMILES_LENGTH} characters).", None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "Invalid SMILES string. Please check the format and try again.", None
    if mol.GetNumHeavyAtoms() == 0:
        return None, "SMILES string does not contain any atoms.", None

    warning = None
    n_atoms = mol.GetNumHeavyAtoms()
    if n_atoms > MAX_REASONABLE_ATOMS:
        warning = (f"This molecule has {n_atoms} heavy atoms \u2014 larger than most molecules "
                   "in the training data. Treat this prediction with extra caution.")

    return mol, None, warning


@st.cache_resource
def load_transformers():
    with open("models/transformer.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_model():
    with open("models/best_solubility_model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_explainer_model():
    """The Random Forest used only for global feature-importance explanations, not for prediction."""
    with open("models/random_forest.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_top_bits():
    with open("models/top_bits.pkl", "rb") as f:
        return pickle.load(f)


# Page setup
st.set_page_config(page_title="SoluScope AI", page_icon="💧", layout="wide")
st.markdown(THEME_CSS, unsafe_allow_html=True)

st.title("SoluScope AI 💧")
st.markdown(
    """<div style="font-family:'IBM Plex Sans', sans-serif; font-size:0.65rem; color:#7C8A93;
    letter-spacing:0.05em; margin-top:0.25rem;">EXPLAINABLE MOLECULAR SOLUBILITY PREDICTION</div>""",
    unsafe_allow_html=True,
)
st.markdown(
    """<div style="font-family:'IBM Plex Sans', sans-serif; font-size:0.65rem; color:#7C8A93;
    letter-spacing:0.05em; margin-top:0.15rem; margin-bottom:0.9rem;">EDUCATIONAL USE ONLY &middot; NOT A SUBSTITUTE FOR EXPERIMENTAL MEASUREMENT</div>""",
    unsafe_allow_html=True,
)
st.markdown(MOLECULE_DIVIDER, unsafe_allow_html=True)

# Disclaimer gate
if "disclaimer_ack" not in st.session_state:
    st.session_state.disclaimer_ack = False

if not st.session_state.disclaimer_ack:
    st.markdown(
        """<div style="background:var(--amber-soft); border:1px solid var(--amber); border-left:4px solid var(--amber);
        border-radius:8px; padding:0.9rem 1.1rem; margin-bottom:1rem; font-family:'IBM Plex Sans', sans-serif; color:var(--ink);">
        <span class="material-symbols-rounded" style="color:var(--amber);">warning</span>
        <strong>Before you begin:</strong> SoluScope AI is an educational demonstration of a machine learning
        pipeline for predicting aqueous solubility. It is not a substitute for experimental measurement and
        should not be used for real drug development decisions without validation.
        </div>""",
        unsafe_allow_html=True,
    )
    if st.button("I understand, continue to SoluScope AI"):
        st.session_state.disclaimer_ack = True
        st.rerun()
    st.stop()

# Sidebar: model info, database, footer
with st.sidebar:
    with st.expander("Model Information", icon=":material/model_training:"):
        st.markdown(
            f"""<div class="info-card" style="background:#FFFFFF; border:1px solid var(--line);
            border-radius:10px; padding:0.9rem 1rem;">
            <div style="font-weight:600; font-size:0.95rem; color:var(--ink); margin-bottom:0.5rem;">{MODEL_INFO['name']}</div>
            <div style="font-family:'IBM Plex Sans', sans-serif; font-size:0.72rem; color:var(--muted); margin-top:5px;">
            <span style="color:var(--blue); font-weight:700; margin-right:0.4rem;">&rsaquo;</span>Hyperparameters: {MODEL_INFO['hyperparameters']}</div>
            <div style="font-family:'IBM Plex Sans', sans-serif; font-size:0.72rem; color:var(--muted); margin-top:5px;">
            <span style="color:var(--blue); font-weight:700; margin-right:0.4rem;">&rsaquo;</span>Features: {MODEL_INFO['features']}</div>
            <div style="font-family:'IBM Plex Sans', sans-serif; font-size:0.72rem; color:var(--muted); margin-top:5px;">
            <span style="color:var(--blue); font-weight:700; margin-right:0.4rem;">&rsaquo;</span>Test R²: {MODEL_INFO['test_r2']}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with st.expander("Database", icon=":material/database:"):
        st.markdown(
            f"""<div class="info-card" style="background:#FFFFFF; border:1px solid var(--line);
            border-radius:10px; padding:0.9rem 1rem;">
            <div style="font-weight:600; font-size:0.95rem; color:var(--ink);">{DATASET_INFO['organization']}</div>
            <div style="font-family:'IBM Plex Sans', sans-serif; font-size:0.72rem; color:var(--muted);
            margin-top:3px;">{DATASET_INFO['title']}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown(
        """<div class="sidebar-footer">
        <span class="material-symbols-rounded" style="font-size:14px; vertical-align:middle;">database</span>
        ESOL / MOLECULENET SOLUBILITY DATASET
        </div>""",
        unsafe_allow_html=True,
    )

# Load model, transformer, explainer assets once
model = load_model()
transformers = load_transformers()
explainer_rf = load_explainer_model()
top_bits = load_top_bits()

# Molecule Input
section_header("science", "Molecule Input")
st.caption("Enter a molecule's SMILES string below, then click Predict Solubility.")

_, input_col, _ = st.columns([1, 2, 1])
with input_col:
    with st.container(key="molecule_input_box"):
        smiles_input = st.text_input("SMILES String", value="", placeholder="\"CCO\" for ethanol, \"CC(=O)O\" for acetic acid, etc.", max_chars=MAX_SMILES_LENGTH)

st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_btn = st.button("Predict Solubility", type="primary", use_container_width=True)

if predict_btn:
    mol, error, warning = validate_smiles(smiles_input)
    if error:
        st.error(error)
    else:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        features = np.array(fp).reshape(1, -1)
        prediction = predict_solubility(model, transformers, features)

        st.session_state["baseline_smiles"] = smiles_input
        st.session_state["baseline_mol"] = mol
        st.session_state["baseline_prediction"] = prediction
        st.session_state["baseline_warning"] = warning

# Prediction Result
if "baseline_prediction" in st.session_state:
    mol = st.session_state["baseline_mol"]
    prediction = st.session_state["baseline_prediction"]

    section_header("water_drop", "Prediction Result")
    st.caption("Model-estimated aqueous solubility based on the molecule entered above.")

    # Layer 1: solubility card, full width
    solubility_card("Predicted Solubility", prediction)

    if st.session_state.get("baseline_warning"):
        st.warning(st.session_state["baseline_warning"])

    st.markdown(
        """<div style="font-family:'IBM Plex Sans', sans-serif; font-size:0.7rem; color:var(--muted);
        margin-top:0.2rem; margin-bottom:0.6rem;">
        <span class="material-symbols-rounded" style="font-size:14px; vertical-align:middle; color:var(--muted);">info</span>
        Computational estimate only &mdash; not a substitute for experimental solubility measurement.
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    # Layer 2: structure image card (left) + molecular properties card (right)
    col_struct, col_props = st.columns([1, 1])

    with col_struct:
        struct_b64 = image_to_base64(Draw.MolToImage(mol, size=(260, 260)))
        st.markdown(
            f"""<div class="info-card animate-in" style="background:#FFFFFF; border:1px solid var(--line);
            border-radius:12px; padding:1rem 1.1rem; box-shadow:0 1px 3px rgba(26,26,46,0.05);
            height:340px; display:flex; flex-direction:column;">
            <div style="font-family:'IBM Plex Serif', serif; font-weight:600; color:var(--ink);
            margin-bottom:0.7rem;">Molecule Structure</div>
            <div style="flex:1; display:flex; align-items:center; justify-content:center;">
            <img src="data:image/png;base64,{struct_b64}" width="220"/>
            </div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col_props:
        props = get_molecular_properties(mol)
        props_html = "".join(
            f"""<div style="display:flex; justify-content:space-between; font-size:0.85rem;
            margin-bottom:0.4rem;"><span style="color:var(--ink);">
            <span style="color:var(--blue); font-weight:700; margin-right:0.4rem;">&rsaquo;</span>{k}</span>
            <strong style="color:var(--ink);">{v}</strong></div>"""
            for k, v in props.items()
        )
        st.markdown(
            f"""<div class="info-card animate-in" style="background:#FFFFFF; border:1px solid var(--line);
            border-radius:12px; padding:1rem 1.1rem; box-shadow:0 1px 3px rgba(26,26,46,0.05); height:340px;">
            <div style="font-family:'IBM Plex Serif', serif; font-weight:600; color:var(--ink);
            margin-bottom:0.7rem;">Molecular Properties</div>{props_html}</div>""",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div style='height:1.5rem;'></div><hr style='border:none; border-top:1px solid var(--line); margin:0 0 1.2rem 0;'/>",
        unsafe_allow_html=True,
    )

    # Layer 3: Why This Prediction
    with st.expander("Why This Prediction?", expanded=False, icon=":material/psychology:"):
        st.caption("Substructures broadly important to the model's solubility predictions.")

        bit_info = {}
        AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024, bitInfo=bit_info)
        active_bits = [b for b in top_bits if b in bit_info][:4]

        col_left, col_right = st.columns(2)

        with col_left:
            if active_bits:
                bits_html = ""
                for b in active_bits:
                    bit_img = Draw.DrawMorganBit(mol, int(b), bit_info, useSVG=False)
                    bit_b64 = image_to_base64(bit_img)
                    bits_html += f"""<div style="display:inline-block; text-align:center; margin:4px;">
                    <img src="data:image/png;base64,{bit_b64}" width="150"/>
                    <div style="font-family:'IBM Plex Sans', sans-serif; font-size:0.65rem; color:var(--muted);">
                    Bit #{b}</div></div>"""
            else:
                bits_html = """<div style="color:var(--muted); font-size:0.85rem; padding:0.5rem 0;">
                None of the globally important bits are present in this molecule.</div>"""

            st.markdown(
                f"""<div class="info-card animate-in" style="background:#FFFFFF; border:1px solid var(--line);
                border-radius:12px; padding:1rem 1.1rem; box-shadow:0 1px 3px rgba(26,26,46,0.05);">
                <div style="font-family:'IBM Plex Serif', serif; font-weight:600; color:var(--ink);
                margin-bottom:0.7rem;">Key Substructures Present</div>{bits_html}</div>""",
                unsafe_allow_html=True,
            )

        with col_right:
            display_bits = top_bits[:6]
            importances = explainer_rf.feature_importances_[display_bits]
            max_imp = importances.max()
            bars_html = ""
            for b, imp in zip(display_bits, importances):
                pct_width = imp / max_imp * 100
                bars_html += f"""<div style="margin-bottom:0.55rem;">
                <div style="display:flex; justify-content:space-between; font-family:'IBM Plex Sans', sans-serif;
                font-size:0.72rem; color:var(--ink); margin-bottom:2px;">
                <span>Bit #{b}</span><span style="color:var(--blue); font-weight:600;">{imp:.3f}</span></div>
                <div style="background:var(--line); border-radius:4px; height:9px; overflow:hidden;">
                <div style="background:var(--blue); width:{pct_width}%; height:100%;"></div></div></div>"""

            st.markdown(
                f"""<div class="info-card animate-in" style="background:#FFFFFF; border:1px solid var(--line); border-radius:12px;
                padding:1rem 1.1rem; box-shadow:0 1px 3px rgba(26,26,46,0.05);">
                <div style="font-family:'IBM Plex Serif', serif; font-weight:600; color:var(--ink);
                margin-bottom:0.7rem;">Global Feature Importance</div>{bars_html}</div>""",
                unsafe_allow_html=True,
            )