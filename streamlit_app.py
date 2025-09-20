# app.py
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Interactive Nomogram", layout="wide")

# -------- utilidades --------
def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))

def range_based_points(betas, example, ranges, bin_vars):
    """
    betas: dict con betas (sin intercept): {"CCI":..., "mRS":..., "Size":..., "Location":..., "Compartment":...}
    example: dict con valores del paciente
    ranges: dict con (min,max) para continuas CCI/mRS/Size
    bin_vars: lista de variables binarias
    Devuelve: dict puntos por predictor, total points
    Regla: 100 puntos = mayor efecto de rango entre todos los predictores
    - Continuas: efecto rango = |beta| * (max - min); puntos paciente = 100 * |beta|*(x - min) / max_range_effect
    - Binarias: efecto rango = |beta|; puntos paciente = 100 * |beta| / max_range_effect si valor==1, si 0 -> 0
    """
    # efectos de rango
    range_effect = {}
    for var in ["CCI","mRS","Size"]:
        b = betas[var]
        lo, hi = ranges[var]
        range_effect[var] = abs(b) * max(0.0, hi - lo)
    for var in bin_vars:
        range_effect[var] = abs(betas[var])  # 0->1

    max_range = max(range_effect.values()) if range_effect else 1.0
    if max_range == 0:
        max_range = 1.0

    pts = {}
    for var in ["CCI","mRS","Size"]:
        b = betas[var]
        lo, _ = ranges[var]
        x = float(example[var])
        pts[var] = 100.0 * abs(b) * max(0.0, x - lo) / max_range
    for var in bin_vars:
        pts[var] = (100.0 * abs(betas[var]) / max_range) if int(example[var]) == 1 else 0.0

    # redondeo bonito
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

# Coeficientes del modelo (pre-cargados con un ejemplo típico)
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
    st.caption("Define reasonable clinical ranges (used only for points, not for probability).")
    cci_min, cci_max = st.number_input("CCI min", 0, 100, 0), st.number_input("CCI max", 0, 100, 6)
    mrs_min, mrs_max = st.number_input("mRS min", 0, 10, 0), st.number_input("mRS max", 0, 10, 5)
    size_min = st.number_input("Size min (mm)", min_value=0.0, value=10.0, step=1.0)
    size_max = st.number_input("Size max (mm)", min_value=0.0, value=120.0, step=1.0)

# Valores del paciente (interactivos)
st.header("3) Patient input")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    val_cci = st.number_input("CCI (integer)", min_value=0, max_value=100, value=2, step=1)
with c2:
    val_mrs = st.number_input("mRS (integer)", min_value=0, max_value=10, value=3, step=1)
with c3:
    val_size = st.number_input("Size (mm)", min_value=0.0, value=40.0, step=1.0)
with c4:
    val_loc = st.radio("Location", options=[0,1], index=0, format_func=lambda v: "Infratentorial (1)" if v==1 else "Supratentorial (0)")
with c5:
    val_comp = st.radio("Compartment", options=[0,1], index=0, format_func=lambda v: "Skull base (1)" if v==1 else "Non-skull base (0)")

# Botón calcular
if st.button("Compute probability & points"):
    # 1) probabilidad
    z = (intercept + beta_cci*val_cci + beta_mrs*val_mrs + beta_size*val_size
         + beta_loc*val_loc + beta_comp*val_comp)
    p = logistic(z)

    st.markdown(f"### Predicted 6-week independence probability: **{p:.3f}**")

    # 2) puntos tipo nomograma (escala 0-100 por máximo rango de efecto)
    betas = {
        "CCI": beta_cci,
        "mRS": beta_mrs,
        "Size": beta_size,
        "Location": beta_loc,
        "Compartment": beta_comp
    }
    example = {
        "CCI": val_cci, "mRS": val_mrs, "Size": val_size,
        "Location": val_loc, "Compartment": val_comp
    }
    ranges = {
        "CCI": (float(cci_min), float(cci_max)),
        "mRS": (float(mrs_min), float(mrs_max)),
        "Size": (float(size_min), float(size_max))
    }
    pts, total_pts = range_based_points(betas, example, ranges, bin_vars=["Location","Compartment"])

    left, right = st.columns([1,1])
    with left:
        st.write("**Nomogram-like points (range-based, max range = 100):**")
        st.json(pts)
        st.write(f"**Total points:** {total_pts:.1f}")
        st.write(f"**Final 6-week independence probability:** {p:.3f}")
    with right:
        fig, ax = plt.subplots()
        items = list(pts.items())
        ax.barh([k for k,_ in items], [v for _,v in items])
        ax.invert_yaxis()
        ax.set_xlabel("Points (max range = 100)")
        ax.set_title("Points by predictor")
        st.pyplot(fig)

# Nota al pie
st.caption("Adjust coefficients and ranges as needed. Probability uses your coefficients; points use the defined ranges.")

