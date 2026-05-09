"""
Black-76 Corn Futures Options Calculator — Streamlit App
CBOT corn (ZC): 5,000 bu/contract, 1¢/bu = $50
"""

import math
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ── Math helpers ──────────────────────────────────────────────────────────────

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def norm_pdf(x):
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)

def black76(F, K, T, sigma, r):
    if sigma <= 0 or T <= 0:
        return None
    sqrt_T = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    df = math.exp(-r * T)
    Nd1, Nd2 = norm_cdf(d1), norm_cdf(d2)
    phi = norm_pdf(d1)

    call = df * (F * Nd1 - K * Nd2)
    put  = df * (K * norm_cdf(-d2) - F * norm_cdf(-d1))

    call_delta = df * Nd1
    put_delta  = -df * norm_cdf(-d1)
    gamma      = df * phi / (F * sigma * sqrt_T)
    vega       = F * df * phi * sqrt_T / 100
    decay      = -F * df * phi * sigma / (2 * sqrt_T)
    call_theta = (decay + r * call) / 365
    put_theta  = (decay + r * put)  / 365
    call_rho   = -T * call / 100
    put_rho    = -T * put  / 100

    return dict(
        call=call, put=put, d1=d1, d2=d2,
        call_delta=call_delta, put_delta=put_delta,
        gamma=gamma, vega=vega,
        call_theta=call_theta, put_theta=put_theta,
        call_rho=call_rho, put_rho=put_rho,
    )

def implied_vol(price, F, K, T, r, opt_type):
    df = math.exp(-r * T)
    intrinsic = (max(0, (F - K) * df) if opt_type == "Call"
                 else max(0, (K - F) * df))
    if price < intrinsic - 1e-6:
        return None, 0, "Price below intrinsic"
    lo, hi = 1e-4, 5.0
    get_p = lambda s: (black76(F, K, T, s, r) or {}).get(
        "call" if opt_type == "Call" else "put", 0)
    if price < get_p(lo):
        return None, 0, "Below model floor"
    if price > get_p(hi):
        return None, 0, "Above model ceiling"
    mid = 0.25
    for i in range(100):
        mid = 0.5 * (lo + hi)
        theo = get_p(mid)
        if abs(theo - price) < 1e-5:
            return mid, i + 1, None
        if theo < price:
            lo = mid
        else:
            hi = mid
    return mid, 100, None


# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Black-76 Corn Options", layout="wide")

st.title("Black-76 corn futures options calculator")
st.caption("CBOT corn (ZC) · Pricing, implied vol solver, and strategy payoff at expiration")

# ── Sidebar — Market Inputs ───────────────────────────────────────────────────

with st.sidebar:
    st.header("Market inputs")
    F = st.slider("Futures price (¢/bu)", 200, 800, 450, 1)
    K = st.slider("Strike (¢/bu)", 200, 800, 460, 1)
    D = st.slider("Days to expiry", 1, 365, 60, 1)
    V = st.slider("Volatility (%, annualized)", 5.0, 100.0, 25.0, 0.5)
    R = st.slider("Risk-free rate (%)", 0.0, 10.0, 4.5, 0.1)

T     = D / 365
sigma = V / 100
r     = R / 100

o = black76(F, K, T, sigma, r)

# ── Section 1: Pricing & Greeks ───────────────────────────────────────────────

st.subheader("Pricing & Greeks")

c1, c2 = st.columns(2)
with c1:
    st.metric("Call premium",
              f"{o['call']:.2f} ¢/bu",
              f"${o['call'] * 50:,.0f} per contract")
with c2:
    st.metric("Put premium",
              f"{o['put']:.2f} ¢/bu",
              f"${o['put'] * 50:,.0f} per contract")

pct = (F - K) / K * 100
if abs(pct) < 0.5:
    moneyness = "ATM"
elif pct > 0:
    moneyness = f"Call ITM by {pct:.1f}%"
else:
    moneyness = f"Put ITM by {-pct:.1f}%"

c3, c4, c5 = st.columns(3)
c3.metric("Moneyness", moneyness)
c4.metric("d1 / d2", f"{o['d1']:.3f} / {o['d2']:.3f}")
c5.metric("Time to expiry", f"{T:.3f} yr")

st.markdown("**Greeks**")
greeks_data = {
    "Greek":          ["Delta", "Gamma", "Vega", "Theta", "Rho"],
    "Call":           [f"{o['call_delta']:.4f}", f"{o['gamma']:.5f}",
                       f"{o['vega']:.3f}",       f"{o['call_theta']:.4f}",
                       f"{o['call_rho']:.4f}"],
    "Put":            [f"{o['put_delta']:.4f}",  f"{o['gamma']:.5f}",
                       f"{o['vega']:.3f}",        f"{o['put_theta']:.4f}",
                       f"{o['put_rho']:.4f}"],
    "Interpretation": [
        "Δ premium per 1¢ move in F",
        "Δ delta per 1¢ move in F",
        "Δ premium per +1% vol (¢/bu)",
        "Δ premium per day (¢/bu)",
        "Δ premium per +1% rate (¢/bu)",
    ],
}
st.dataframe(pd.DataFrame(greeks_data), use_container_width=True, hide_index=True)

st.divider()

# ── Section 2: Implied Volatility Solver ─────────────────────────────────────

st.subheader("Implied volatility solver")

iv_col1, iv_col2, iv_col3 = st.columns(3)
with iv_col1:
    iv_type  = st.selectbox("Option type", ["Call", "Put"], key="iv_type")
with iv_col2:
    iv_K     = st.number_input("Strike (¢/bu)", value=460, step=1, key="iv_K")
with iv_col3:
    iv_price = st.number_input("Market price (¢/bu)", value=15.00, step=0.25, key="iv_price")

iv_vol, iv_iters, iv_err = implied_vol(iv_price, F, iv_K, T, r, iv_type)

r2_col1, r2_col2, r2_col3 = st.columns(3)
if iv_err or iv_vol is None:
    r2_col1.metric("Implied vol", iv_err or "No solution")
    r2_col2.metric("Theo at IV", "—")
    r2_col3.metric("Solver iterations", "—")
else:
    theo_res = black76(F, iv_K, T, iv_vol, r)
    theo_val = theo_res["call"] if iv_type == "Call" else theo_res["put"]
    r2_col1.metric("Implied vol",        f"{iv_vol * 100:.2f}%")
    r2_col2.metric("Theo at IV",         f"{theo_val:.2f} ¢/bu")
    r2_col3.metric("Solver iterations",  str(iv_iters))

st.divider()

# ── Section 3: Strategy Builder ───────────────────────────────────────────────

st.subheader("Strategy builder — expiration P&L")

PRESETS = {
    "Long call":    [("long",  "call", 0,   1, 25)],
    "Long put":     [("long",  "put",  0,   1, 25)],
    "Bull call":    [("long",  "call", 0,   1, 25), ("short", "call", +30, 1, 25)],
    "Bear put":     [("long",  "put",  0,   1, 25), ("short", "put",  -30, 1, 25)],
    "Long straddle":[("long",  "call", 0,   1, 25), ("long",  "put",  0,   1, 25)],
    "Long strangle":[("long",  "call", +20, 1, 25), ("long",  "put",  -20, 1, 25)],
    "Iron condor":  [("short", "call", +20, 1, 25), ("long",  "call", +50, 1, 25),
                     ("short", "put",  -20, 1, 25), ("long",  "put",  -50, 1, 25)],
    "Fence":        [("long",  "put",  -20, 1, 25), ("short", "call", +20, 1, 25)],
}

atm = round(F / 10) * 10

# Session-state legs
if "legs" not in st.session_state:
    st.session_state.legs = [
        {"side": "long",  "type": "call", "strike": 460, "qty": 1, "vol": 25},
        {"side": "short", "type": "call", "strike": 490, "qty": 1, "vol": 25},
    ]

# Preset buttons
preset_cols = st.columns(len(PRESETS))
for col, (label, offsets) in zip(preset_cols, PRESETS.items()):
    if col.button(label, use_container_width=True):
        st.session_state.legs = [
            {"side": s, "type": t, "strike": atm + dk, "qty": q, "vol": v}
            for s, t, dk, q, v in offsets
        ]

# Add / clear
btn_a, btn_b = st.columns(2)
if btn_a.button("+ Add leg", use_container_width=True):
    st.session_state.legs.append({"side": "long", "type": "call",
                                   "strike": atm, "qty": 1, "vol": 25})
if btn_b.button("Clear all", use_container_width=True):
    st.session_state.legs = []

# Leg editor
legs_to_remove = []
for i, leg in enumerate(st.session_state.legs):
    lc1, lc2, lc3, lc4, lc5, lc6 = st.columns([1, 1, 1.4, 0.8, 1.2, 0.4])
    leg["side"]   = lc1.selectbox("Side",   ["long", "short"],         index=["long","short"].index(leg["side"]),   key=f"side_{i}",   label_visibility="collapsed")
    leg["type"]   = lc2.selectbox("Type",   ["call", "put"],           index=["call","put"].index(leg["type"]),     key=f"type_{i}",   label_visibility="collapsed")
    leg["strike"] = lc3.number_input("Strike",  value=float(leg["strike"]), step=1.0, key=f"strike_{i}", label_visibility="collapsed")
    leg["qty"]    = lc4.number_input("Qty",     value=int(leg["qty"]),      step=1,   key=f"qty_{i}",    label_visibility="collapsed", min_value=1)
    leg["vol"]    = lc5.number_input("IV %",    value=float(leg["vol"]),    step=0.5, key=f"vol_{i}",    label_visibility="collapsed", min_value=1.0)
    if lc6.button("×", key=f"rm_{i}"):
        legs_to_remove.append(i)

for idx in sorted(legs_to_remove, reverse=True):
    st.session_state.legs.pop(idx)

# ── Payoff computation ────────────────────────────────────────────────────────

legs = st.session_state.legs
net_cost = 0.0
for leg in legs:
    sig = leg["vol"] / 100
    if sig <= 0 or not leg["strike"]:
        continue
    p = black76(F, leg["strike"], T, sig, r)
    if p is None:
        continue
    prem = p["call"] if leg["type"] == "call" else p["put"]
    sign = 1 if leg["side"] == "long" else -1
    net_cost += sign * prem * leg["qty"]

lo_F = max(50, F * 0.6)
hi_F = F * 1.4
N    = 200
xs, ys = [], []
for i in range(N + 1):
    Fexp = lo_F + (hi_F - lo_F) * i / N
    pl   = -net_cost
    for leg in legs:
        intr = (max(0, Fexp - leg["strike"]) if leg["type"] == "call"
                else max(0, leg["strike"] - Fexp))
        sign = 1 if leg["side"] == "long" else -1
        pl  += sign * intr * leg["qty"]
    xs.append(Fexp)
    ys.append(pl)

max_p = max(ys) if ys else 0
min_p = min(ys) if ys else 0

breakevens = []
for i in range(1, len(ys)):
    y0, y1 = ys[i - 1], ys[i]
    if (y0 < 0 and y1 >= 0) or (y0 > 0 and y1 <= 0):
        be = xs[i - 1] - y0 * (xs[i] - xs[i - 1]) / (y1 - y0)
        breakevens.append(be)

# Summary metrics
m1, m2, m3, m4 = st.columns(4)
nc_label = f"{net_cost:.2f} ¢/bu" if net_cost >= 0 else f"+{-net_cost:.2f} ¢/bu"
m1.metric("Net debit / credit", nc_label if legs else "—")
m2.metric("Max profit",  f"{max_p:.2f} ¢/bu" if legs else "—")
m3.metric("Max loss",    f"{min_p:.2f} ¢/bu" if legs else "—")
m4.metric("Breakeven(s)", ", ".join(f"{b:.1f}" for b in breakevens) if breakevens else "—")

# ── Payoff chart ──────────────────────────────────────────────────────────────

if legs:
    profit_y = [y if y > 0 else None for y in ys]
    loss_y   = [y if y <= 0 else None for y in ys]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=profit_y, name="Profit",
        fill="tozeroy", fillcolor="rgba(29,158,117,0.18)",
        line=dict(color="#1D9E75", width=2),
        mode="lines", connectgaps=False,
        hovertemplate="F=%{x:.1f}¢/bu<br>P&L=+%{y:.2f}¢/bu<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=loss_y, name="Loss",
        fill="tozeroy", fillcolor="rgba(226,75,74,0.18)",
        line=dict(color="#E24B4A", width=2),
        mode="lines", connectgaps=False,
        hovertemplate="F=%{x:.1f}¢/bu<br>P&L=%{y:.2f}¢/bu<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(0,0,0,0.3)", line_width=1)
    fig.update_layout(
        xaxis_title="Futures price at expiration (¢/bu)",
        yaxis_title="P&L (¢/bu)",
        showlegend=False,
        height=340,
        margin=dict(l=0, r=0, t=10, b=10),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(245,244,239,0.5)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Add at least one leg to see the payoff diagram.")

st.divider()
st.caption(
    "Black-76 model. CBOT corn (ZC): 5,000 bu/contract, 1¢/bu = $50. "
    "P&L shown in ¢/bu. Payoff is at expiration only — no time value remaining. "
    "Listed CBOT corn options are technically American-style; Black-76 prices the "
    "European equivalent and the early-exercise premium is usually small but nonzero."
)
