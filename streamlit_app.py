# streamlit_app.py
import numpy as np
import pandas as pd
import streamlit as st

# IMPORTANT: use a headless backend for matplotlib on Streamlit Cloud
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="Interactive Nomogram", layout="wide")

# -------- utils --------
def logistic(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))

def sanitize_range(lo: float, hi: float, name: str):
    """Ensure (lo < hi). If not, auto-fix and warn."""
    if hi <= lo:
        st.warning(f"[{name}] max must be > min. Auto-fixing to min={lo}, max={lo+1}.")
        hi = lo + 1.0
    return float(lo), float(hi)

def range_based_points(betas, example, ranges, bin_vars):
    """
    betas: dict without intercept: {"CCI":..., "mRS":..., "Size":..., "Location":..., "Compartment":...}
    example: dict with patient values
    ranges: dict of (min, max) for continuous vars (CCI, mRS, Size) used only for points scaling
    bin_vars: list of binary var names
    Returns: (points_by_var: dict, total_points: float)
    Rule: 100 points = largest range effect among predictors
      - Continuous: range effect = |beta| * (max - min)
        patient points = 100 * |beta| * (x - min) / max_range_effect  (clamped at >=0)
      - Binary: range effect = |beta| (0→1)
        patient points = 100 * |beta| / max_range_effect if value==1 else 0
    """
    # range effects
    range_effect = {}
    for var in ["CCI", "mRS", "Size"]:
        b = float(betas[var])
        lo, hi = ranges[var]
        range_effect[var] = abs(b) * max(0.0, hi - lo)
    for var in bin_vars:
        range_effect[var] = abs(float(betas[var]))  # 0->1

    max_range = max(range_effect.values()) if range_effect else 1.0
    if max_range == 0:
        max_range = 1.0

    pts = {}
    for var in ["CCI", "mRS", "Size"]:
        b = float(betas[var])
        lo, _ = ranges[var]
        x = float(example[var])
        pts[var] = 100.0 * abs(b) * max(0.0, x - lo) / max_range
    for var in bin_vars:
        pts[var] = (100.0 * abs(float(betas[var])) / max_range) if int(example[var]) == 1 else 0.0

    pts = {k: float(np.round(v, 1)) for k, v in pts.items()}
    total = float(np.round(sum(pts.values()), 1))
    return pts, total

# -------- UI --------
st.title("Interactive Logistic Nomogram (no CSV)")

st.markdown(
    """
Enter your **model coefficients**, **clinical ranges** for scaling points, and the **patient values**.  
The app returns the **6-week independence probability** and **nomogram-like points** (0–100).
"""
)

# Sidebar: coefficients & ranges
with st.sidebar:
    st.header("1) Model coefficients")
    st.caption("Logit(p) = Intercept + β1·CCI + β2·mRS + β3·Size + β4·Location + β5·Compartment")

    intercept = st.number_input("Intercept", value=-2.6984, step=0.1, format="%.4f")
    beta_cci  = st.number_input("β (CCI)", value=0.1673, step=0.01, format="%.4f")
    beta_mrs  = st.number_input("β (mRS)", value=0.5453, step=0.01, format="%.4f")
    beta_size = st.number_input("β (Size, per 1 mm)", value=0.0007, step=0.0001, format="%.4f")
    beta_loc  = st.number_input("β (Location, 1=Infratentorial)", value=-0.5973, step=0.01, format="%.4f")
    beta_comp = st.number_input("β (Compartment, 1=Skull base)", value=0.6434, step=0.01, format="%.4f")

    st.markdown("---")
    st.header("2) Ranges for points scaling")
    st.caption("Used only for points (not for probability). Choose reasonable clinical ranges.")
    cci_min = st.number_input("CCI min", value=0, step=1, min_value=0)
    cci_max = st.number_input("CCI max", value=6, step=1, min_value=0)
    mrs_min = st.number_input("mRS min", value=0, step=1, min_value=0)
    mrs_max = st.number_input("mRS max", value=5, step=1, min_value=0)
    size_min = st.number_input("Size min (mm)", min_value=0.0, value=10.0, step=1.0)
    size_max = st.number_input("Size max (mm)", min_value=0.0, value=120.0, step=1.0)

    # sanitize ranges
    cci_min, cci_max = sanitize_range(float(cci_min), float(cci_max), "CCI")
    mrs_min, mrs_max = sanitize_range(float(mrs_min), float(mrs_max), "mRS")
    size_min, size_max = sanitize_range(float(size_min), float(size_max), "Size")

# Patient inputs
st.header("3) Patient input")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    val_cci = st.number_input("CCI (integer)", min_value=0, max_value=100, value=2, step=1)
with c2:
    val_mrs = st.number_input("mRS (integer)", min_value=0, max_value=10, value=3, step=1)
with c3:
    val_size = st.number_input("Size (mm)", min_value=0.0, value=40.0, step=1.0)
with c4:
    val_loc = st.radio("Location", options=[0, 1],
                       index=0, format_func=lambda v: "Infratentorial (1)" if v == 1 else "Supratentorial (0)")
with c5:
    val_comp = st.radio("Compartment", options=[0, 1],
                        index=0, format_func=lambda v: "Skull base (1)" if v == 1 else "Non-skull base (0)")

if st.button("Compute probability & points"):
    # 1) Probability
    z = (
        float(intercept)
        + float(beta_cci) * float(val_cci)
        + float(beta_mrs) * float(val_mrs)
        + float(beta_size) * float(val_size)
        + float(beta_loc) * int(val_loc)
        + float(beta_comp) * int(val_comp)
    )
    p = logistic(z)
    st.markdown(f"### Predicted 6-week independence probability: **{p:.3f}**")

    # 2) Points (range-based, 0–100 by max range effect)
    betas = {
        "CCI": float(beta_cci),
        "mRS": float(beta_mrs),
        "Size": float(beta_size),
        "Location": float(beta_loc),
        "Compartment": float(beta_comp),
    }
    example = {
        "CCI": float(val_cci),
        "mRS": float(val_mrs),
        "Size": float(val_size),
        "Location": int(val_loc),
        "Compartment": int(val_comp),
    }
    ranges = {
        "CCI": (float(cci_min), float(cci_max)),
        "mRS": (float(mrs_min), float(mrs_max)),
        "Size": (float(size_min), float(size_max)),
    }

    pts, total_pts = range_based_points(betas, example, ranges, bin_vars=["Location", "Compartment"])

    left, right = st.columns([1, 1])
    with left:
        st.write("**Nomogram-like points (range-based, max range = 100):**")
        st.json(pts)
        st.write(f"**Total points:** {total_pts:.1f}")
        st.write(f"**Final 6-week independence probability:** {p:.3f}")

    with right:
        fig, ax = plt.subplots()
        items = list(pts.items())
        ax.barh([k for k, _ in items], [v for _, v in items])
        ax.invert_yaxis()
        ax.set_xlabel("Points (max range = 100)")
        ax.set_title("Points by predictor")
        st.pyplot(fig)

st.caption("Adjust coefficients and ranges as needed. Probability uses your coefficients; points use the defined ranges.")
