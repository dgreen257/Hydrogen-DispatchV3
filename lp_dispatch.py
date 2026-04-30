"""
LP Strategic Dispatch — mini Streamlit dashboard.

Replaces the greedy merit-order in the main dashboard with a proper
Linear Programme (scipy HiGHS solver). Objective weights map directly
to LP objective coefficients, not just a sorting criterion.

Run:  streamlit run lp_dispatch.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import linprog

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LP Dispatch — H2 Corridors",
    layout="wide",
    initial_sidebar_state="expanded",
)

RESULTS_DIR = "Results"
CORRIDORS   = ["A", "B", "C", "D", "E", "EU"]

CORRIDOR_NAMES = {
    "A":  "A — N. Africa → Italy",
    "B":  "B — Iberian GW",
    "C":  "C — Global → Rotterdam",
    "D":  "D — Nordic → Hamburg",
    "E":  "E — SE Europe/Ukraine",
    "EU": "EU — Domestic",
}

REGION_COLOURS = {
    "North Africa":         "#e67e22",
    "Sub-Saharan Africa":   "#d35400",
    "Middle East":          "#c0392b",
    "EU":                   "#2980b9",
    "North America":        "#27ae60",
    "South America":        "#16a085",
    "Oceania":              "#8e44ad",
    "Nordic/Baltic":        "#2c3e50",
    "SE Europe":            "#7f8c8d",
    "Other":                "#bdc3c7",
}

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading corridor data…")
def load_raw_corridors() -> dict[str, pd.DataFrame]:
    """Read all corridor CSVs (transport costs pre-computed)."""
    dfs = {}
    for cid in CORRIDORS:
        path = os.path.join(RESULTS_DIR, f"corridor_{cid}_base.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, index_col=0)
        df["Corridor"] = cid
        dfs[cid] = df
    return dfs


@st.cache_data(show_spinner="Computing generation costs…")
def apply_gen_costs(year: int) -> dict[str, pd.DataFrame]:
    """Compute live generation costs for each corridor CSV."""
    from generation_costs import generation_costs as _gen_costs

    raw = load_raw_corridors()
    out = {}
    for cid, df in raw.items():
        df2 = _gen_costs(df.copy(), h2_demand=1000, year=year)
        if "Transport Cost per kg H2" in df2.columns:
            df2["Total Cost per kg H2"] = (
                df2["Gen. cost per kg H2"] + df2["Transport Cost per kg H2"]
            )
        out[cid] = df2
    return out


@st.cache_data(show_spinner="Loading capacity data…")
def load_caps(year: int) -> dict:
    path = "Data/combined_caps_by_year.csv"
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    df["Year"] = df["Year"].astype(int)
    yr = df[df["Year"] == year]
    return dict(zip(yr["ISO_A3"], yr["Capacity_kt"]))


@st.cache_data(show_spinner=False)
def load_security() -> pd.DataFrame:
    path = "Data/Security.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_water() -> pd.DataFrame:
    path = "Data/aqueduct-30-country-rankings.xlsx"
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_excel(path, sheet_name="results country")
    bws = df[(df["indicator_name"] == "bws") & (df["weight"] == "Tot")].copy()
    bws["score"] = pd.to_numeric(bws["score"], errors="coerce")
    bws.loc[bws["score"] < 0, "score"] = np.nan
    return bws[["iso_a3", "name_0", "score", "label"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Country-level aggregation
# ---------------------------------------------------------------------------

def build_country_df(
    year: int,
    selected_corridors: list[str],
    w_sec: float,
    w_water: float,
) -> pd.DataFrame:
    """
    Aggregate grid-point data to one row per country (cheapest corridor kept).
    Merges capacity, CPI, and water stress. Normalises scores.
    """
    corridor_dfs = apply_gen_costs(year)
    caps         = load_caps(year)
    security_df  = load_security()
    water_df     = load_water()

    TRIM = 0.10
    rows = []

    for cid in selected_corridors:
        df = corridor_dfs.get(cid)
        if df is None or df.empty:
            continue
        if "Total Cost per kg H2" not in df.columns:
            continue

        for iso, grp in df.groupby("ISO_A3"):
            grp_s = grp.sort_values("Total Cost per kg H2").reset_index(drop=True)
            n = len(grp_s)
            nt = int(np.floor(n * TRIM))
            mid = grp_s.iloc[nt: n - nt] if n - 2 * nt >= 1 else grp_s

            em_col = "Total Emissions per kg H2"
            rows.append({
                "ISO_A3":              iso,
                "Country":             grp["Country"].iloc[0] if "Country" in grp.columns else iso,
                "corridor_id":         cid,
                "H2_Region":           grp["H2_Region"].mode()[0] if "H2_Region" in grp.columns else "Other",
                "rep_cost_per_kg":     float(mid["Total Cost per kg H2"].mean()),
                "rep_emissions_per_kg": float(mid[em_col].mean()) if em_col in mid.columns else np.nan,
            })

    if not rows:
        return pd.DataFrame()

    df_all = pd.DataFrame(rows)
    # Keep cheapest corridor per country
    df_all = (
        df_all.sort_values("rep_cost_per_kg")
        .drop_duplicates("ISO_A3", keep="first")
        .reset_index(drop=True)
    )

    # Capacity
    df_all["country_cap_kt"] = df_all["ISO_A3"].map(caps)
    df_all = df_all[
        df_all["country_cap_kt"].notna()
        & (df_all["country_cap_kt"] > 0)
        & (df_all["country_cap_kt"] <= 1e6)
    ].copy()

    # Security (CPI)
    if not security_df.empty and "ISO3" in security_df.columns:
        sec = (
            security_df[["ISO3", "CPI score 2025"]]
            .rename(columns={"ISO3": "ISO_A3", "CPI score 2025": "cpi_score"})
        )
        df_all = df_all.merge(sec, on="ISO_A3", how="left")
    else:
        df_all["cpi_score"] = np.nan

    # Water stress (BWS)
    if not water_df.empty and "iso_a3" in water_df.columns:
        wat = (
            water_df[["iso_a3", "score"]]
            .rename(columns={"iso_a3": "ISO_A3", "score": "bws_score"})
        )
        df_all = df_all.merge(wat, on="ISO_A3", how="left")
    else:
        df_all["bws_score"] = np.nan

    # Normalise
    cost_min = df_all["rep_cost_per_kg"].min()
    cost_p95 = df_all["rep_cost_per_kg"].quantile(0.95)
    df_all["cost_norm"]     = ((df_all["rep_cost_per_kg"] - cost_min) / max(cost_p95 - cost_min, 1e-6)).clip(0, 1)
    df_all["security_norm"] = (df_all["cpi_score"] / 100.0).clip(0, 1)
    df_all["water_norm"]    = (df_all["bws_score"]  / 5.0 ).clip(0, 1)

    # Drop countries missing data for actively-weighted indicators
    if w_sec > 0:
        df_all = df_all[df_all["security_norm"].notna()]
    if w_water > 0:
        df_all = df_all[df_all["water_norm"].notna()]

    df_all["security_norm"] = df_all["security_norm"].fillna(0.5)
    df_all["water_norm"]    = df_all["water_norm"].fillna(0.5)

    return df_all.reset_index(drop=True)


# ---------------------------------------------------------------------------
# LP solver
# ---------------------------------------------------------------------------

def solve_lp(
    country_df: pd.DataFrame,
    demand_kt: float,
    w_cost: float,
    w_sec: float,
    w_water: float,
    w_dep: float,
    div_mode: str = "Country",
) -> tuple[pd.DataFrame, dict]:
    """
    Linear Programme:
      minimise  Σᵢ composite_i · xᵢ
      subject to
        Σᵢ xᵢ = demand_kt                       (meet demand exactly)
        0 ≤ xᵢ ≤ country_cap_kt_i               (capacity per source)
        Σᵢ∈g xᵢ ≤ max_per_group  ∀ groups g    (diversification cap)

    Returns (result_df, solver_info).
    """
    df = country_df.copy().reset_index(drop=True)
    solver_info = {"status": "no data", "message": "", "obj_value": np.nan}

    # Drop any rows with NaN in columns needed for the objective
    df["cost_norm"]     = df["cost_norm"].fillna(1.0)   # unknown cost → worst
    df["security_norm"] = df["security_norm"].fillna(0.5)
    df["water_norm"]    = df["water_norm"].fillna(0.5)
    df = df[np.isfinite(df["cost_norm"]) & np.isfinite(df["security_norm"]) & np.isfinite(df["water_norm"])]
    df = df.reset_index(drop=True)

    n = len(df)
    if n == 0 or demand_kt <= 0:
        return df, solver_info

    # --- Objective ---
    w_sum = max(float(w_cost + w_sec + w_water), 1e-6)
    c = (
        (w_cost  / w_sum) * df["cost_norm"].values +
        (w_sec   / w_sum) * (1.0 - df["security_norm"].values) +
        (w_water / w_sum) * df["water_norm"].values
    )

    # --- Bounds ---
    bounds = [(0.0, float(cap)) for cap in df["country_cap_kt"]]

    # --- Equality: meet demand ---
    A_eq = np.ones((1, n))
    b_eq = np.array([demand_kt])

    # --- Diversification inequality constraints ---
    w_dep_frac     = float(w_dep) / 100.0
    max_per_entity = demand_kt * max(0.05, (1.0 - w_dep_frac) ** 2)

    GROUP_COL = {"Country": "ISO_A3", "Region": "H2_Region", "Corridor": "corridor_id"}
    gcol      = GROUP_COL.get(div_mode, "ISO_A3")

    A_ub_rows, b_ub_rows = [], []
    for grp_val in df[gcol].unique():
        mask = (df[gcol] == grp_val).values.astype(float)
        A_ub_rows.append(mask)
        b_ub_rows.append(max_per_entity)

    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_rows)

    # --- Feasibility check ---
    total_cap = sum(min(b, demand_kt) for (_, b) in bounds)
    if total_cap < demand_kt * 0.999:
        solver_info["status"]  = "infeasible"
        solver_info["message"] = (
            f"Total available capacity ({total_cap:,.0f} kt) < demand ({demand_kt:,.0f} kt). "
            "Reduce demand or enable more corridors."
        )
        df["allocated_kt"] = 0.0
        return df, solver_info

    # --- Solve ---
    result = linprog(
        c,
        A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    solver_info["status"]  = "optimal" if result.success else result.status
    solver_info["message"] = result.message
    solver_info["obj_value"] = float(result.fun) if result.success else np.nan

    if result.success:
        df["allocated_kt"] = result.x
    else:
        df["allocated_kt"] = 0.0

    df = df[df["allocated_kt"] > 0.01].sort_values("allocated_kt", ascending=False)
    return df.reset_index(drop=True), solver_info


# ---------------------------------------------------------------------------
# Greedy solver (for comparison tab)
# ---------------------------------------------------------------------------

def solve_greedy(
    country_df: pd.DataFrame,
    demand_kt: float,
    w_cost: float,
    w_sec: float,
    w_water: float,
    w_dep: float,
    div_mode: str = "Country",
) -> pd.DataFrame:
    """Replicate the existing weighted greedy dispatch from dashboard.py."""
    df = country_df.copy()
    if df.empty or demand_kt <= 0:
        return df

    df["cost_norm"]     = df["cost_norm"].fillna(1.0)
    df["security_norm"] = df["security_norm"].fillna(0.5)
    df["water_norm"]    = df["water_norm"].fillna(0.5)

    w_sum = max(float(w_cost + w_sec + w_water), 1e-6)
    df["composite"] = (
        (w_cost  / w_sum) * df["cost_norm"] +
        (w_sec   / w_sum) * (1.0 - df["security_norm"]) +
        (w_water / w_sum) * df["water_norm"]
    )
    df = df.sort_values("composite").reset_index(drop=True)

    w_dep_frac     = float(w_dep) / 100.0
    max_per_entity = demand_kt * max(0.05, (1.0 - w_dep_frac) ** 2)

    GROUP_COL = {"Country": "ISO_A3", "Region": "H2_Region", "Corridor": "corridor_id"}
    gcol      = GROUP_COL.get(div_mode, "ISO_A3")

    allocated_rows: list[dict] = []
    remaining   = float(demand_kt)
    group_used: dict = {}

    # Pass 1 — respect concentration cap
    for _, row in df.iterrows():
        if remaining <= 1e-6:
            break
        key  = row.get(gcol, row["ISO_A3"])
        used = group_used.get(key, 0.0)
        avail = min(float(row["country_cap_kt"]), max(0.0, max_per_entity - used), remaining)
        group_used[key] = used + avail
        if avail > 1e-6:
            allocated_rows.append({**row.to_dict(), "allocated_kt": avail})
            remaining -= avail

    # Pass 2 — fill remainder without cap
    if remaining > 1e-6:
        already = {r["ISO_A3"]: r["allocated_kt"] for r in allocated_rows}
        for _, row in df.iterrows():
            if remaining <= 1e-6:
                break
            iso   = row["ISO_A3"]
            used  = already.get(iso, 0.0)
            avail = min(float(row["country_cap_kt"]) - used, remaining)
            if avail > 1e-6:
                existing = next((r for r in allocated_rows if r["ISO_A3"] == iso), None)
                if existing:
                    existing["allocated_kt"] += avail
                else:
                    allocated_rows.append({**row.to_dict(), "allocated_kt": avail})
                remaining -= avail

    return pd.DataFrame(allocated_rows).sort_values("allocated_kt", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# KPI computation
# ---------------------------------------------------------------------------

def compute_kpis(dispatch_df: pd.DataFrame, demand_kt: float) -> dict:
    empty = dict(cost=np.nan, emissions=np.nan, cpi=np.nan, bws=np.nan, hhi=np.nan, n=0)
    if dispatch_df.empty or "allocated_kt" not in dispatch_df.columns:
        return empty
    total = dispatch_df["allocated_kt"].sum()
    if total < 1e-6:
        return empty
    w = dispatch_df["allocated_kt"] / total
    shares = dispatch_df.groupby("ISO_A3")["allocated_kt"].sum() / total
    return {
        "cost":      float((dispatch_df["rep_cost_per_kg"]     * w).sum()),
        "emissions": float((dispatch_df["rep_emissions_per_kg"] * w).sum())
                     if "rep_emissions_per_kg" in dispatch_df.columns else np.nan,
        "cpi":       float((dispatch_df["cpi_score"]  * w).sum())
                     if "cpi_score"  in dispatch_df.columns else np.nan,
        "bws":       float((dispatch_df["bws_score"]  * w).sum())
                     if "bws_score"  in dispatch_df.columns else np.nan,
        "hhi":       float((shares ** 2).sum()),
        "n":         int((dispatch_df["allocated_kt"] > 0.01).sum()),
    }


# ---------------------------------------------------------------------------
# Pareto sweep
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Running Pareto sweep…")
def pareto_sweep(
    country_df_json: str,
    demand_kt: float,
    w_dep: float,
    div_mode: str,
    n_points: int = 25,
) -> pd.DataFrame:
    """
    Sweep w_cost from 100→0 while w_sec = 100 − w_cost, w_water = 0.
    Returns DataFrame with columns: w_cost, lp_cost, lp_cpi, greedy_cost, greedy_cpi.
    """
    country_df = pd.read_json(country_df_json)
    records = []
    for w_c in np.linspace(0, 100, n_points):
        w_s = 100 - w_c

        lp_df, info = solve_lp(country_df, demand_kt, w_c, w_s, 0, w_dep, div_mode)
        g_df        = solve_greedy(country_df, demand_kt, w_c, w_s, 0, w_dep, div_mode)

        lp_kpi  = compute_kpis(lp_df,  demand_kt)
        g_kpi   = compute_kpis(g_df,   demand_kt)

        records.append({
            "w_cost":      w_c,
            "lp_cost":     lp_kpi["cost"],
            "lp_cpi":      lp_kpi["cpi"],
            "greedy_cost": g_kpi["cost"],
            "greedy_cpi":  g_kpi["cpi"],
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("LP Dispatch Controls")
    st.caption("Solves a Linear Programme — not greedy merit order.")

    st.subheader("Scenario")
    year       = st.slider("Year", 2026, 2050, 2030, step=1)
    demand_kt  = st.number_input("Demand (kt H₂/yr)", 100, 50_000, 2_000, step=100)

    st.subheader("Corridors")
    selected_corridors = st.multiselect(
        "Include corridors",
        CORRIDORS,
        default=CORRIDORS,
        format_func=lambda x: CORRIDOR_NAMES.get(x, x),
    )

    st.subheader("Objective Weights")
    st.caption("These set LP objective coefficients directly — not just a sort key.")
    w_cost  = st.slider("Cost weight",             0, 100, 50, key="lp_w_cost")
    w_sec   = st.slider("Security (CPI) weight",   0, 100, 25, key="lp_w_sec")
    w_water = st.slider("Water stress weight",      0, 100, 25, key="lp_w_water")
    w_dep   = st.slider("Diversification cap",      0, 100, 25, key="lp_w_dep")
    div_mode = st.selectbox(
        "Diversification by",
        ["Country", "Region", "Corridor"],
    )
    total_w = w_cost + w_sec + w_water
    st.caption(f"Weight sum: {total_w} (normalised internally)")

    st.divider()
    st.caption("Built on scipy HiGHS LP solver. Solves in <100 ms.")


# ---------------------------------------------------------------------------
# Build country dataframe
# ---------------------------------------------------------------------------

country_df = build_country_df(year, selected_corridors, w_sec, w_water)

if country_df.empty:
    st.error("No data loaded. Check that Results/ and Data/ directories are present.")
    st.stop()

# ---------------------------------------------------------------------------
# Solve LP + greedy
# ---------------------------------------------------------------------------

lp_df, solver_info = solve_lp(country_df, demand_kt, w_cost, w_sec, w_water, w_dep, div_mode)
greedy_df           = solve_greedy(country_df, demand_kt, w_cost, w_sec, w_water, w_dep, div_mode)

lp_kpi     = compute_kpis(lp_df,     demand_kt)
greedy_kpi = compute_kpis(greedy_df, demand_kt)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("LP Strategic Dispatch")
st.caption(
    f"Year: **{year}** | Demand: **{demand_kt:,} kt H₂/yr** | "
    f"Solver: **{solver_info['status']}** | "
    f"Countries available: **{len(country_df)}**"
)

if solver_info["status"] == "infeasible":
    st.error(solver_info["message"])
elif solver_info["status"] not in ("optimal", "no data"):
    st.warning(f"Solver: {solver_info['message']}")

# ---------------------------------------------------------------------------
# KPI metrics row
# ---------------------------------------------------------------------------

col1, col2, col3, col4, col5, col6 = st.columns(6)

def _fmt(v, fmt=".2f", fallback="—"):
    return f"{v:{fmt}}" if pd.notna(v) else fallback

col1.metric("LP Cost (€/kg)",        _fmt(lp_kpi["cost"]),
            delta=_fmt(lp_kpi["cost"] - greedy_kpi["cost"], "+.3f") if pd.notna(greedy_kpi["cost"]) else None,
            delta_color="inverse")
col2.metric("LP Emissions (kg/kg)",  _fmt(lp_kpi["emissions"]))
col3.metric("LP CPI (0–100)",        _fmt(lp_kpi["cpi"], ".0f"),
            delta=_fmt(lp_kpi["cpi"] - greedy_kpi["cpi"], "+.1f") if pd.notna(greedy_kpi["cpi"]) else None)
col4.metric("LP Water (BWS 0–5)",    _fmt(lp_kpi["bws"]))
col5.metric("LP HHI",                _fmt(lp_kpi["hhi"]))
col6.metric("LP Countries",          lp_kpi["n"])

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_lp, tab_compare, tab_pareto, tab_table = st.tabs([
    "LP Allocation", "LP vs Greedy", "Pareto Frontier", "Data Table"
])

# ── Tab 1: LP Allocation ────────────────────────────────────────────────────
with tab_lp:
    if lp_df.empty:
        st.warning("LP returned no allocation.")
    else:
        lp_df["pct"] = lp_df["allocated_kt"] / demand_kt * 100

        fig_bar = px.bar(
            lp_df,
            x="Country",
            y="allocated_kt",
            color="H2_Region",
            color_discrete_map=REGION_COLOURS,
            hover_data={
                "rep_cost_per_kg":     ":.3f",
                "cpi_score":           ":.0f",
                "bws_score":           ":.2f",
                "allocated_kt":        ":.1f",
                "pct":                 ":.1f",
            },
            labels={"allocated_kt": "Allocated (kt/yr)", "Country": ""},
            title=f"LP Optimal Allocation — {demand_kt:,} kt/yr",
        )
        fig_bar.update_layout(height=420, xaxis_tickangle=-45, legend_title="Region")
        st.plotly_chart(fig_bar, use_container_width=True)

        # Cost vs share scatter
        fig_sc = px.scatter(
            lp_df,
            x="rep_cost_per_kg",
            y="pct",
            color="H2_Region",
            size="allocated_kt",
            text="ISO_A3",
            color_discrete_map=REGION_COLOURS,
            hover_name="Country",
            hover_data={"cpi_score": ":.0f", "bws_score": ":.2f"},
            labels={"rep_cost_per_kg": "Cost (€/kg H₂)", "pct": "Share of demand (%)"},
            title="Cost vs supply share — LP allocation",
        )
        fig_sc.update_traces(textposition="top center", textfont_size=10)
        fig_sc.update_layout(height=380)
        st.plotly_chart(fig_sc, use_container_width=True)


# ── Tab 2: LP vs Greedy comparison ─────────────────────────────────────────
with tab_compare:
    st.subheader("Side-by-side: LP (optimal) vs Greedy (current model)")

    c_lp, c_gr = st.columns(2)

    def _kpi_card(kpi: dict, label: str):
        st.markdown(f"**{label}**")
        st.metric("Avg cost (€/kg)",     _fmt(kpi["cost"]))
        st.metric("Avg CPI",             _fmt(kpi["cpi"], ".1f"))
        st.metric("Avg BWS",             _fmt(kpi["bws"], ".2f"))
        st.metric("HHI",                 _fmt(kpi["hhi"]))
        st.metric("Countries used",      kpi["n"])

    with c_lp:
        _kpi_card(lp_kpi, "LP (this dashboard)")
    with c_gr:
        _kpi_card(greedy_kpi, "Greedy (main dashboard)")

    st.divider()

    # Allocation comparison bar
    if not lp_df.empty and not greedy_df.empty:
        merged = (
            lp_df[["ISO_A3", "Country", "allocated_kt"]]
            .rename(columns={"allocated_kt": "LP"})
            .merge(
                greedy_df[["ISO_A3", "allocated_kt"]].rename(columns={"allocated_kt": "Greedy"}),
                on="ISO_A3", how="outer",
            )
            .fillna(0)
            .sort_values("LP", ascending=False)
        )

        fig_comp = go.Figure()
        fig_comp.add_bar(name="LP",     x=merged["Country"], y=merged["LP"],     marker_color="#2980b9")
        fig_comp.add_bar(name="Greedy", x=merged["Country"], y=merged["Greedy"], marker_color="#e67e22", opacity=0.7)
        fig_comp.update_layout(
            barmode="group",
            title="Allocation by country — LP vs Greedy",
            yaxis_title="Allocated (kt/yr)",
            xaxis_tickangle=-45,
            height=420,
            legend_title="Method",
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # Difference
        merged["diff"] = merged["LP"] - merged["Greedy"]
        merged_nonzero = merged[merged["diff"].abs() > 0.5].sort_values("diff")
        if not merged_nonzero.empty:
            fig_diff = px.bar(
                merged_nonzero,
                x="Country",
                y="diff",
                color=merged_nonzero["diff"].apply(lambda v: "LP higher" if v > 0 else "Greedy higher"),
                color_discrete_map={"LP higher": "#2980b9", "Greedy higher": "#e67e22"},
                title="Allocation difference (LP − Greedy)",
                labels={"diff": "Δ kt/yr", "Country": ""},
            )
            fig_diff.update_layout(height=360, xaxis_tickangle=-45)
            st.plotly_chart(fig_diff, use_container_width=True)
    else:
        st.info("One or both solvers returned no allocation.")


# ── Tab 3: Pareto Frontier ──────────────────────────────────────────────────
with tab_pareto:
    st.subheader("Cost–Security Pareto frontier")
    st.caption(
        "Sweeps w_cost 0→100 (w_sec = 100 − w_cost, w_water = 0). "
        "Shows the trade-off curve for LP vs the greedy approach."
    )

    if st.button("Run Pareto sweep (25 points)"):
        pareto_df = pareto_sweep(
            country_df.to_json(),
            demand_kt, w_dep, div_mode,
        )

        fig_p = go.Figure()
        fig_p.add_scatter(
            x=pareto_df["lp_cost"], y=pareto_df["lp_cpi"],
            mode="lines+markers", name="LP",
            marker=dict(color="#2980b9", size=8),
            line=dict(color="#2980b9"),
            text=[f"w_cost={w:.0f}" for w in pareto_df["w_cost"]],
        )
        fig_p.add_scatter(
            x=pareto_df["greedy_cost"], y=pareto_df["greedy_cpi"],
            mode="lines+markers", name="Greedy",
            marker=dict(color="#e67e22", size=8, symbol="square"),
            line=dict(color="#e67e22", dash="dash"),
            text=[f"w_cost={w:.0f}" for w in pareto_df["w_cost"]],
        )
        fig_p.update_layout(
            xaxis_title="Weighted avg cost (€/kg H₂) ← lower is better",
            yaxis_title="Weighted avg CPI (0–100) ↑ higher is better",
            title="Cost–Security trade-off: LP vs Greedy",
            height=460,
            legend_title="Method",
        )
        st.plotly_chart(fig_p, use_container_width=True)
        st.caption(
            "Points to the **upper-left** are Pareto-dominant (cheaper AND more secure). "
            "The LP curve should lie on or above the greedy curve — it achieves at least as good "
            "a trade-off at every weight setting."
        )


# ── Tab 4: Data table ───────────────────────────────────────────────────────
with tab_table:
    st.subheader("LP allocation detail")
    if not lp_df.empty:
        show_cols = [c for c in [
            "Country", "ISO_A3", "corridor_id", "H2_Region",
            "allocated_kt", "rep_cost_per_kg", "rep_emissions_per_kg",
            "cpi_score", "bws_score", "country_cap_kt",
        ] if c in lp_df.columns]
        st.dataframe(
            lp_df[show_cols].rename(columns={
                "rep_cost_per_kg":      "Cost (€/kg)",
                "rep_emissions_per_kg": "Emissions (kg/kg)",
                "cpi_score":            "CPI",
                "bws_score":            "BWS",
                "allocated_kt":         "Allocated (kt)",
                "country_cap_kt":       "Capacity (kt)",
            }),
            use_container_width=True,
        )

    st.subheader("All candidate countries")
    base_cols = [c for c in [
        "Country", "ISO_A3", "corridor_id", "H2_Region",
        "rep_cost_per_kg", "cpi_score", "bws_score",
        "cost_norm", "security_norm", "water_norm", "country_cap_kt",
    ] if c in country_df.columns]
    st.dataframe(country_df[base_cols], use_container_width=True)
