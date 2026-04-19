"""
dashboard.py
------------
Interactive Streamlit dashboard for the H2-Mapping corridor model.

Reads pre-computed Results/corridor_A_{year}.csv … corridor_E_{year}.csv and
renders interactive views:
  1. Supply curve  — cumulative capacity vs cost, per corridor
  2. Cost breakdown  — gen vs transport cost, median of within-cap points
  3. Source maps  — scatter globe of supply locations coloured by cost / mode / source
  4. Transport modes — pie chart + breakdown table
  5. Country caps, Flow map, Assumptions, Strategic Dispatch, H₂ Projects Pipeline

Sidebar controls (no model re-run required):
  - Year: auto-detected from available CSVs in Results/
  - Corridor toggles
  - Capacity limit toggle
  - Annual demand (kt/yr) — dynamically recomputes within_cap
  - Generation cost adjustment (%) — sensitivity analysis without re-running
  - RED III threshold slider
  - Map metric selector

Run
---
    cd "h2-mapping-main"
    /opt/anaconda3/envs/h2mapping/bin/streamlit run dashboard.py
"""

import os
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CORRIDOR_META = {
    'A':  {'name': 'North Africa → South Europe',      'colour': '#E63946'},
    'B':  {'name': 'Sub-Saharan Africa → Europe',       'colour': '#F4A261'},
    'C':  {'name': 'Global → North-West Europe',        'colour': '#2A9D8F'},
    'D':  {'name': 'North Sea & Caucasus → NW Europe',  'colour': '#457B9D'},
    'E':  {'name': 'Middle East → South Europe',        'colour': '#9B2335'},
    'EU': {'name': 'EU Domestic Production',            'colour': '#009688'},
}

EU_RED3_DEFAULT = 3.38   # kgCO₂eq / kgH₂

# Import ports: (lat,lon) key → display label
PORT_OPTIONS = {
    '37,-7':  'Huelva, Spain (37°N, 7°W)',
    '42,12':  'Civitavecchia, Italy (42°N, 12°E)',
    '48,16':  'Vienna, Austria (48°N, 16°E)',
    '52,4':   'Rotterdam, Netherlands (52°N, 4°E)',
    '54,10':  'Hamburg, Germany (54°N, 10°E)',
}

from config import DEMAND_PROFILES, demand_for_year
from corridors import CORRIDORS, EU_MEMBER_ISOS
from plot_corridor import _build_flow_map_fig, _supply_curve_data, _build_optimal_mix
# aggregate_country_supply inlined from run_corridors.py to avoid heavy dependencies
def aggregate_country_supply(df: pd.DataFrame, caps: dict,
                              trim_pct: float = 0.10) -> pd.DataFrame:
    rows = []
    for iso, grp in df.groupby('ISO_A3'):
        grp_sorted = grp.sort_values('Total Cost per kg H2').reset_index(drop=True)
        n = len(grp_sorted)
        n_trim = int(np.floor(n * trim_pct))
        if n - 2 * n_trim >= 1:
            middle = grp_sorted.iloc[n_trim: n - n_trim]
        else:
            middle = grp_sorted

        cap = caps.get(iso, np.nan) if isinstance(iso, str) else np.nan
        if pd.notna(cap) and cap > 1e6:
            cap = np.nan
        has_finite_cap = pd.notna(cap) and cap > 0

        rows.append({
            'ISO_A3':                iso,
            'Country':               grp['Country'].iloc[0] if 'Country' in grp.columns else iso,
            'rep_cost_per_kg':       middle['Total Cost per kg H2'].mean(),
            'gen_cost_per_kg':       middle['Gen. cost per kg H2'].mean(),
            'transport_cost_per_kg': middle['Transport Cost per kg H2'].mean(),
            'country_cap_kt':        cap,
            'has_finite_cap':        has_finite_cap,
            'n_points':              n,
            'n_top_points':          len(middle),
        })

    result = (pd.DataFrame(rows)
                .sort_values('rep_cost_per_kg')
                .reset_index(drop=True))

    uncapped = result[~result['has_finite_cap']]['ISO_A3'].tolist()
    if uncapped:
        print(f'  Uncapped countries (excluded from supply curve by default): {uncapped}')

    return result
from generation_costs import global_capex
from country_factors import SOLAR_CAPEX_FACTOR, WIND_CAPEX_FACTOR, WACC, WACC_COUNTRY_REN, WACC_COUNTRY_ELEC
from plotly.subplots import make_subplots

GEO_LAYOUT = dict(
    showland=True,
    landcolor='rgb(212,212,212)',
    subunitcolor='rgb(255,255,255)',
    countrycolor='rgb(255,255,255)',
    showlakes=True,
    lakecolor='rgb(255,255,255)',
    showsubunits=True,
    showcountries=True,
    projection=dict(type='natural earth'),
)

CHOROPLETH_GEO = dict(
    showframe=False,
    showcoastlines=True,
    coastlinecolor='#aaaaaa',
    showland=True,
    landcolor='#f5f5f5',
    showocean=True,
    oceancolor='#d0e8f5',
    projection_type='natural earth',
    lataxis_range=[-60, 90],
)

RESULTS_DIR = 'Results'

# ---------------------------------------------------------------------------
# Data discovery and loading (cached)
# ---------------------------------------------------------------------------

def discover_available() -> dict[str, list[int]]:
    """
    Scan Results/ and return {scenario_name: [years]} for available runs.

    Supports two modes:
    - Base files (corridor_A_base.csv etc.): all scenario/year combinations are
      available since generation costs are computed live. Years are taken from
      the demand profile anchors.
    - Legacy year/scenario files (corridor_A_2035_Intermediate.csv etc.): only
      pre-computed combinations are available.

    Base files take priority if present.
    """
    if os.path.exists(RESULTS_DIR):
        base_corridors = [fn for fn in os.listdir(RESULTS_DIR)
                          if re.match(r'corridor_[A-EU]+_base\.csv', fn)]
    else:
        base_corridors = []

    if base_corridors:
        # Base files exist — all scenarios and years are available
        all_years = sorted({
            yr
            for profile in DEMAND_PROFILES.values()
            for yr in profile.keys()
        })
        return {scenario: all_years for scenario in DEMAND_PROFILES}

    # Fallback: legacy year/scenario files
    slug_to_name = {k.replace(' ', '_'): k for k in DEMAND_PROFILES}
    available: dict[str, set] = {}
    if os.path.exists(RESULTS_DIR):
        for fn in os.listdir(RESULTS_DIR):
            m = re.match(r'corridor_[A-E]_(\d{4})_(.+)\.csv', fn)
            if m:
                year = int(m.group(1))
                name = slug_to_name.get(m.group(2), m.group(2))
                available.setdefault(name, set()).add(year)
    return {k: sorted(v) for k, v in available.items()}


@st.cache_data(show_spinner='Loading corridor data…')
def load_corridors(year: int, scenario: str) -> dict[str, pd.DataFrame]:
    """Load all available corridor CSVs for the given year and scenario.

    Prefers base files (corridor_A_base.csv) if present; falls back to
    legacy year/scenario files (corridor_A_2035_Intermediate.csv).
    """
    slug = scenario.replace(' ', '_')
    dfs = {}
    for cid in list(CORRIDOR_META.keys()):
        base_path = os.path.join(RESULTS_DIR, f'corridor_{cid}_base.csv')
        legacy_path = os.path.join(RESULTS_DIR, f'corridor_{cid}_{year}_{slug}.csv')
        path = base_path if os.path.exists(base_path) else legacy_path
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            # Natural Earth assigns ISO_A3 = '-99' to some countries (France, Norway).
            # Remap using Country name so CPI/BWS merges work correctly.
            # Unknown entries (Somaliland, Ocean) keep their original value.
            _NEG99_FIX = {'France': 'FRA', 'Norway': 'NOR'}
            if 'ISO_A3' in df.columns and 'Country' in df.columns:
                mask = df['ISO_A3'] == '-99'
                remapped = df.loc[mask, 'Country'].map(_NEG99_FIX)
                df.loc[mask, 'ISO_A3'] = remapped.fillna(df.loc[mask, 'ISO_A3'])
            df['Corridor'] = cid
            dfs[cid] = df
    return dfs


# ---------------------------------------------------------------------------
# Dynamic capacity recomputation (cached by demand + year)
# ---------------------------------------------------------------------------

def _apply_capacity_limits(df: pd.DataFrame, h2_demand: float,
                           year: int, n_global: int = None,
                           caps: dict = None) -> pd.DataFrame:
    if caps is None:
        caps = _get_caps_for_year(year)

    df = df.copy()
    df['country_cap_kt'] = df['ISO_A3'].map(caps)
    df.loc[df['country_cap_kt'] > 1e6, 'country_cap_kt'] = np.nan

    demand_per_point = h2_demand / (n_global if n_global is not None else len(df))
    df_sorted = df.sort_values('Total Cost per kg H2').copy()

    country_used: dict = {}
    within_cap_flags = []

    for _, row in df_sorted.iterrows():
        iso = row.get('ISO_A3')
        cap = caps.get(iso, np.inf) if (isinstance(iso, str) and iso != '---') else np.inf
        used = country_used.get(iso, 0.0)

        if used + demand_per_point <= cap:
            within_cap_flags.append(True)
            country_used[iso] = used + demand_per_point
        else:
            within_cap_flags.append(False)

    df_sorted['within_cap'] = within_cap_flags
    df_sorted['cap_rank'] = np.nan
    mask = df_sorted['within_cap']
    df_sorted.loc[mask, 'cap_rank'] = np.arange(1, mask.sum() + 1)
    df = df_sorted.reindex(df.index)
    return df


@st.cache_data(show_spinner='Recomputing capacity limits…')
def recompute_within_cap(df: pd.DataFrame, demand_kt: float, year: int) -> pd.DataFrame:
    """
    Re-run apply_capacity_limits with the selected demand value.
    Cached so repeated slider moves at the same value don't recompute.
    """
    return _apply_capacity_limits(df, demand_kt, year, caps=_get_caps_for_year(year))


# ---------------------------------------------------------------------------
# Live generation cost calculation (replaces pre-computed gen cost columns)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner='Calculating generation costs…')
def compute_gen_costs(df_base: pd.DataFrame, year: int, elec_type: str,
                      capex_solar: float, capex_wind: float,
                      capex_elec: float) -> pd.DataFrame:
    """
    Compute generation costs live from CAPEX slider values.

    h2_demand cancels out in the per-kg formula (constant returns to scale),
    so a fixed reference value of 1000 kt/yr is used — the per-kg outputs are
    identical regardless of demand. Demand is only used for demand-line display
    and within_cap allocation (handled separately).
    """
    from generation_costs import generation_costs as _gen_costs
    df = _gen_costs(
        df_base.copy(), h2_demand=1000, year=year, elec_type=elec_type,
        capex_solar_override=capex_solar,
        capex_wind_override=capex_wind,
        capex_elec_override=capex_elec,
    )
    if 'Transport Cost per kg H2' in df.columns:
        df['Total Cost per kg H2'] = (
            df['Gen. cost per kg H2'] + df['Transport Cost per kg H2']
        )
    return df


# ---------------------------------------------------------------------------
# Transport cost adjustment (post-hoc sensitivity)
# ---------------------------------------------------------------------------

def adjust_transport_costs(df: pd.DataFrame, adj_frac: float) -> pd.DataFrame:
    """
    Scale Transport Cost per kg H2 by adj_frac and recompute Total Cost.
    adj_frac = 1.0 → no change; 0.8 → 20% cheaper; 1.2 → 20% more expensive.
    """
    if abs(adj_frac - 1.0) < 1e-6:
        return df
    df = df.copy()
    df['Transport Cost per kg H2'] = df['Transport Cost per kg H2'] * adj_frac
    if 'Gen. cost per kg H2' in df.columns:
        df['Total Cost per kg H2'] = (
            df['Gen. cost per kg H2'] + df['Transport Cost per kg H2']
        )
    return df


# ---------------------------------------------------------------------------
# Filter application
# ---------------------------------------------------------------------------

def apply_filters(df: pd.DataFrame, cap_on: bool, demand_kt: float,
                  year: int, transport_adj_frac: float = 1.0) -> pd.DataFrame:
    """Apply transport cost adjustment and capacity filter to a corridor DataFrame."""
    # 1. Adjust transport costs (so capacity ranking uses adjusted total costs)
    out = adjust_transport_costs(df, transport_adj_frac)

    # 2. Recompute capacity limits with the current demand
    if cap_on:
        out = recompute_within_cap(out, demand_kt, year)
        out = out[out['within_cap'] == True]

    return out


# ---------------------------------------------------------------------------
# Chart functions
# ---------------------------------------------------------------------------

def fig_supply_curve(results_with_country: dict, h2_demand_kt: float,
                     year: int) -> go.Figure:
    """
    Merit-order supply curve replicating plot_corridor_supply_curves().

    Thin stepped lines = individual corridors (one step per country,
    width = country capacity cap, height = trimmed mean cost (P10–P90)).
    Thick coloured segments = optimal mix (greedy global dispatch, cheapest
    country first across all corridors, respecting capacity caps).
    """
    if not results_with_country:
        return go.Figure()

    curves = _supply_curve_data(results_with_country, h2_demand_kt)
    alloc  = _build_optimal_mix(results_with_country, h2_demand_kt)

    fig = go.Figure()

    # ── Individual corridor supply curves (thin, stepped) ──────────────────
    for cid, c in curves.items():
        if len(c['cum_supply']) == 0:
            continue
        xs = np.concatenate([[0], c['cum_supply']])
        ys = np.concatenate([[c['cost'][0]], c['cost']])

        if c['labels'] is not None:
            padded_labels = np.concatenate([[c['labels'][0]], c['labels']])
            hover = (
                '<b>%{fullData.name}</b><br>'
                'Country: %{customdata}<br>'
                'Cumulative supply: %{x:,.0f} kt H₂/yr<br>'
                'Cost: %{y:.3f} €/kg H₂<extra></extra>'
            )
        else:
            padded_labels = None
            hover = (
                '<b>%{fullData.name}</b><br>'
                'Cumulative supply: %{x:,.0f} kt H₂/yr<br>'
                'Cost: %{y:.3f} €/kg H₂<extra></extra>'
            )

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode='lines',
            line=dict(color=c['colour'], width=2, shape='hv'),
            opacity=0.55,
            name=c['label'],
            customdata=padded_labels,
            hovertemplate=hover,
        ))

    # ── Optimal mix (thick segments, coloured by corridor) ─────────────────
    if not alloc.empty:
        seen_cids: set = set()
        for cid in alloc['corridor'].unique():
            sub = alloc[alloc['corridor'] == cid].reset_index(drop=True)
            colour = sub['colour'].iloc[0]

            xs_opt, ys_opt, hover_opt = [], [], []
            for _, row in sub.iterrows():
                xs_opt  += [row['cum_kt_start'], row['cum_kt'], None]
                ys_opt  += [row['cost'], row['cost'], None]
                hover_opt += [
                    (f'<b>Optimal mix — {row["country"]} [{cid}]</b><br>'
                     f'Cost: {row["cost"]:.3f} €/kg H₂<br>'
                     f'Allocated: {row["allocated_kt"]:,.0f} kt/yr<extra></extra>')
                ] * 3

            first = cid not in seen_cids
            seen_cids.add(cid)
            cfg_name = results_with_country[cid]['corridor']['subtitle']

            fig.add_trace(go.Scatter(
                x=xs_opt, y=ys_opt,
                mode='lines',
                line=dict(color=colour, width=5),
                name=f'Optimal — {cid}: {cfg_name}',
                legendgroup=f'optimal_{cid}',
                showlegend=first,
                customdata=hover_opt,
                hovertemplate='%{customdata}',
            ))

        # Demand line
        fig.add_vline(
            x=h2_demand_kt,
            line_dash='dot', line_color='#444444', line_width=1.5,
            annotation_text=f'Demand<br>{h2_demand_kt/1000:.1f} Mt/yr',
            annotation_position='top right',
            annotation_font=dict(size=11, color='#444444'),
        )


    fig.update_layout(
        title=dict(
            text=(f'EHB Corridor Supply Curves — Country Merit Order ({year})<br>'
                  '<sup>Thin lines = individual corridors  |  '
                  'Thick segments = optimal mix (cheapest global dispatch, coloured by corridor)</sup>'),
            font=dict(size=15),
        ),
        xaxis=dict(title='Cumulative supply capacity [kt H₂/yr]', gridcolor='#e0e0e0',
                   range=[0, h2_demand_kt]),
        yaxis=dict(title='Total delivered cost [€/kg H₂]', gridcolor='#e0e0e0'),
        legend=dict(
            orientation='v', x=1.01, y=1, xanchor='left',
            bgcolor='rgba(255,255,255,0.85)', bordercolor='#cccccc', borderwidth=1,
        ),
        plot_bgcolor='white', paper_bgcolor='white',
        hovermode='closest',
        height=580,
    )
    return fig


def fig_cost_breakdown(dfs_filtered: dict, show_corridors: list[str], n_countries: int = 5) -> go.Figure:
    """Subplots: stacked bar of gen + transport cost for the cheapest N countries per corridor."""
    active = [cid for cid in show_corridors
              if dfs_filtered.get(cid) is not None and not dfs_filtered[cid].empty]
    if not active:
        return go.Figure().update_layout(title='No data')

    fig = make_subplots(
        rows=1, cols=len(active),
        subplot_titles=[f'Corridor {cid}' for cid in active],
        shared_yaxes=True,
    )

    showlegend = True
    for i, cid in enumerate(active, 1):
        df = dfs_filtered[cid]
        valid = df.dropna(subset=['Total Cost per kg H2', 'Gen. cost per kg H2',
                                  'Transport Cost per kg H2'])
        if 'Country' not in valid.columns or valid.empty:
            continue
        ctry = (valid.groupby('Country')
                     .agg(gen_cost=('Gen. cost per kg H2', 'mean'),
                          trans_cost=('Transport Cost per kg H2', 'mean'),
                          total_cost=('Total Cost per kg H2', 'mean'))
                     .sort_values('total_cost')
                     .head(n_countries)
                     .reset_index())

        fig.add_trace(go.Bar(
            name='Generation', x=ctry['Country'], y=ctry['gen_cost'],
            marker_color='#457B9D',
            hovertemplate='%{x}<br>Generation: %{y:.2f} €/kg H₂<extra></extra>',
            legendgroup='gen', showlegend=showlegend,
        ), row=1, col=i)
        fig.add_trace(go.Bar(
            name='Transport', x=ctry['Country'], y=ctry['trans_cost'],
            marker_color='#E63946',
            hovertemplate='%{x}<br>Transport: %{y:.2f} €/kg H₂<extra></extra>',
            legendgroup='trans', showlegend=showlegend,
        ), row=1, col=i)
        showlegend = False

    fig.update_layout(
        barmode='stack',
        title=f'Cost Breakdown — Cheapest {n_countries} Countries per Corridor',
        yaxis_title='Cost (€/kg H₂)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        plot_bgcolor='white',
        height=450,
    )
    fig.update_xaxes(tickangle=90)
    return fig


def fig_emissions_breakdown(dfs_filtered: dict, show_corridors: list[str],
                             red3_threshold: float) -> go.Figure:
    """Stacked bar: median gen + transport emissions per corridor, with RED III line."""
    labels, gen_emis, trans_emis = [], [], []

    for cid in show_corridors:
        df = dfs_filtered.get(cid)
        if df is None or df.empty:
            continue
        valid = df.dropna(subset=['Gen. emissions per kg H2', 'Transport Emissions per kg H2'])
        if valid.empty:
            continue
        labels.append(f'Corridor {cid}')
        gen_emis.append(valid['Gen. emissions per kg H2'].median())
        trans_emis.append(valid['Transport Emissions per kg H2'].median())

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Generation', x=labels, y=gen_emis,
        marker_color='#457B9D',
        hovertemplate='Generation: %{y:.3f} kgCO₂eq/kgH₂<extra></extra>',
    ))
    fig.add_trace(go.Bar(
        name='Transport', x=labels, y=trans_emis,
        marker_color='#E63946',
        hovertemplate='Transport: %{y:.3f} kgCO₂eq/kgH₂<extra></extra>',
    ))

    if labels:
        fig.add_hline(
            y=red3_threshold,
            line_dash='dash',
            line_color='black',
            annotation_text=f'RED III threshold ({red3_threshold:.2f} kgCO₂eq/kgH₂)',
            annotation_position='top right',
        )

    fig.update_layout(
        barmode='stack',
        title='Emissions Breakdown — Median of Within-Cap Points',
        yaxis_title='Lifecycle Emissions (kgCO₂eq / kgH₂)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        plot_bgcolor='white',
        height=400,
    )
    return fig


def fig_source_map(dfs_filtered: dict, show_corridors: list[str],
                   metric: str, height: int = 420) -> go.Figure:
    """Scatter globe coloured by the chosen metric."""
    col_map = {
        'Total Cost (€/kg H₂)':        'Total Cost per kg H2',
        'Generation Cost (€/kg H₂)':   'Gen. cost per kg H2',
        'Transport Cost (€/kg H₂)':    'Transport Cost per kg H2',
        'Total Emissions (kgCO₂eq/kg)': 'Total Emissions per kg H2',
        'Cheapest Transport Mode':      'Cheapest Medium',
        'Cheaper Energy Source':        'Cheaper source',
    }
    col = col_map.get(metric, 'Total Cost per kg H2')

    frames = []
    for cid in show_corridors:
        df = dfs_filtered.get(cid)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        return go.Figure()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=['Latitude', 'Longitude', col])

    if col in ('Cheapest Medium', 'Cheaper source'):
        fig = px.scatter_geo(
            combined, lat='Latitude', lon='Longitude',
            color=col,
            hover_data={'ISO_A3': True, 'Total Cost per kg H2': ':.2f'},
            projection='natural earth',
            title=f'Source Locations — {metric}',
        )
    else:
        p_hi = combined[col].quantile(0.97)
        fig = px.scatter_geo(
            combined, lat='Latitude', lon='Longitude',
            color=col,
            color_continuous_scale='Cividis',
            hover_data={'ISO_A3': True, 'Total Cost per kg H2': ':.2f', col: ':.3f'},
            projection='natural earth',
            title=f'Source Locations — {metric}',
            labels={col: metric},
        )
        fig.update_coloraxes(cmin=combined[col].min(), cmax=p_hi, colorbar_title=metric)
        fig.update_traces(marker_cauto=False, marker_cmin=combined[col].min(), marker_cmax=p_hi)

    fig.update_geos(**GEO_LAYOUT)
    fig.update_traces(marker_size=3)
    fig.update_layout(height=height)
    return fig


# ---------------------------------------------------------------------------
# Port-specific data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner='Loading port data…')
def load_port_results(port_key: str, year: int) -> pd.DataFrame:
    """Load port result CSV, preferring base files over legacy year files."""
    base_path = os.path.join(RESULTS_DIR, f'{port_key}_base.csv')
    legacy_path = os.path.join(RESULTS_DIR, f'{port_key}_{year}.csv')
    path = base_path if os.path.exists(base_path) else legacy_path
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, index_col=0)


def fig_port_source_map(df: pd.DataFrame, metric: str, port_label: str,
                        height: int = 380) -> go.Figure:
    """Scatter geo of source locations from a port-specific results DataFrame."""
    col_map = {
        'Total Cost (€/kg H₂)':      'Total Cost per kg H2',
        'Transport Cost (€/kg H₂)':  'Transport Cost per kg H2',
        'Cheapest Transport Mode':    'Cheapest Medium',
    }
    col = col_map.get(metric, 'Total Cost per kg H2')
    if col not in df.columns or df.empty:
        return go.Figure().update_layout(title=f'No data — {metric}', height=height)

    plot_df = df.dropna(subset=['Latitude', 'Longitude', col])
    if plot_df.empty:
        return go.Figure().update_layout(title=f'No data — {metric}', height=height)

    title = f'{metric} — Port: {port_label}'
    if col == 'Cheapest Medium':
        fig = px.scatter_geo(
            plot_df, lat='Latitude', lon='Longitude',
            color=col,
            hover_data={'ISO_A3': True, 'Total Cost per kg H2': ':.2f'},
            projection='natural earth',
            title=title,
        )
    else:
        p_hi = 3 if 'Transport' in col else plot_df[col].quantile(0.90)
        p_lo = plot_df[col].min()
        fig = px.scatter_geo(
            plot_df, lat='Latitude', lon='Longitude',
            color=col,
            color_continuous_scale='Cividis',
            range_color=[p_lo, p_hi],
            hover_data={'ISO_A3': True, col: ':.3f'},
            projection='natural earth',
            title=title,
            labels={col: metric},
        )
        fig.update_coloraxes(cauto=False, cmin=p_lo, cmax=p_hi, colorbar_title=metric)

    fig.update_geos(**GEO_LAYOUT)
    fig.update_traces(marker_size=3)
    fig.update_layout(height=height)
    return fig


# ---------------------------------------------------------------------------
# Security and water stress maps
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_security_data() -> pd.DataFrame:
    path = 'Data/Security.csv'
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_water_stress_data() -> pd.DataFrame:
    path = 'Data/aqueduct-30-country-rankings.xlsx'
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_excel(path, sheet_name='results country')
    bws = df[(df['indicator_name'] == 'bws') & (df['weight'] == 'Tot')].copy()
    bws['score'] = pd.to_numeric(bws['score'], errors='coerce')
    bws.loc[bws['score'] < 0, 'score'] = np.nan
    return bws[['iso_a3', 'name_0', 'score', 'label']].reset_index(drop=True)


def fig_security_map(height: int = 380) -> go.Figure:
    """Choropleth of Corruption Perception Index 2025 by country."""
    df = load_security_data()
    if df.empty:
        return go.Figure().update_layout(title='Security data not found', height=height)
    fig = px.choropleth(
        df,
        locations='ISO3',
        color='CPI score 2025',
        color_continuous_scale='RdYlGn',
        range_color=[0, 100],
        hover_name='Country / Territory',
        hover_data={'ISO3': True, 'CPI score 2025': True},
        title='Corruption Perception Index 2025 (higher = less corrupt)',
        labels={'CPI score 2025': 'CPI Score'},
    )
    fig.update_geos(**CHOROPLETH_GEO)
    fig.update_layout(
        paper_bgcolor='white',
        margin=dict(l=0, r=0, t=50, b=0),
        height=height,
        coloraxis_colorbar=dict(title='CPI Score', thickness=15, len=0.6),
    )
    return fig


def fig_water_stress_map(height: int = 380) -> go.Figure:
    """Choropleth of WRI Aqueduct baseline water stress score by country."""
    df = load_water_stress_data()
    if df.empty:
        return go.Figure().update_layout(title='Water stress data not found', height=height)
    fig = px.choropleth(
        df,
        locations='iso_a3',
        color='score',
        color_continuous_scale='RdYlBu_r',
        range_color=[0, 5],
        hover_name='name_0',
        hover_data={'score': ':.2f', 'label': True, 'iso_a3': False},
        title='Baseline Water Stress (WRI Aqueduct 3.0)',
        labels={'score': 'BWS Score', 'label': 'Category'},
    )
    fig.update_geos(**CHOROPLETH_GEO)
    fig.update_layout(
        paper_bgcolor='white',
        margin=dict(l=0, r=0, t=50, b=0),
        height=height,
        coloraxis_colorbar=dict(title='BWS Score', thickness=15, len=0.6),
    )
    return fig


# ---------------------------------------------------------------------------
# Strategic Multi-Criteria Dispatch — helper functions
# ---------------------------------------------------------------------------

def _aggregate_strategic_country_df(
    dfs_filtered: dict,
    show_corridors: list,
    caps_for_year: dict,
    security_df: pd.DataFrame,
    water_df: pd.DataFrame,
    cap_on: bool = True,
) -> pd.DataFrame:
    """
    Build a per-country DataFrame for strategic dispatch with cost, emissions,
    security (CPI), and water stress (BWS) scores.

    Returns one row per unique country (cheapest corridor kept where a country
    appears in multiple corridors), sorted by rep_cost_per_kg.
    """
    TRIM_PCT = 0.10
    rows = []

    for cid in show_corridors:
        df = dfs_filtered.get(cid)
        if df is None or df.empty:
            continue
        if 'Total Cost per kg H2' not in df.columns:
            continue

        for iso, grp in df.groupby('ISO_A3'):
            grp_sorted = grp.sort_values('Total Cost per kg H2').reset_index(drop=True)
            n = len(grp_sorted)
            n_trim = int(np.floor(n * TRIM_PCT))
            middle = grp_sorted.iloc[n_trim: n - n_trim] if n - 2 * n_trim >= 1 else grp_sorted

            cap = caps_for_year.get(iso, np.nan) if isinstance(iso, str) else np.nan
            if pd.notna(cap) and cap > 1e6:
                cap = np.nan
            has_finite_cap = pd.notna(cap) and cap > 0

            emiss_col = 'Total Emissions per kg H2'
            rep_emiss = float(middle[emiss_col].mean()) if emiss_col in middle.columns else np.nan

            rows.append({
                'ISO_A3':               iso,
                'Country':              grp['Country'].iloc[0] if 'Country' in grp.columns else iso,
                'corridor_id':          cid,
                'H2_Region':            grp['H2_Region'].mode()[0] if 'H2_Region' in grp.columns else 'Other',
                'rep_cost_per_kg':      float(middle['Total Cost per kg H2'].mean()),
                'rep_emissions_per_kg': rep_emiss,
                'country_cap_kt':       cap,
                'has_finite_cap':       has_finite_cap,
            })

    if not rows:
        return pd.DataFrame()

    all_df = pd.DataFrame(rows)

    # Where a country appears in multiple corridors, keep cheapest
    all_df = (
        all_df.sort_values('rep_cost_per_kg')
        .drop_duplicates('ISO_A3', keep='first')
        .reset_index(drop=True)
    )

    # Keep only countries with a finite capacity estimate (or lift all caps)
    if cap_on:
        all_df = all_df[all_df['has_finite_cap']].copy()
    else:
        all_df = all_df.copy()
        all_df['country_cap_kt'] = float('inf')

    # Merge security (CPI)
    if not security_df.empty and 'ISO3' in security_df.columns:
        sec = security_df[['ISO3', 'CPI score 2025']].rename(
            columns={'ISO3': 'ISO_A3', 'CPI score 2025': 'cpi_score'}
        )
        all_df = all_df.merge(sec, on='ISO_A3', how='left')
    else:
        all_df['cpi_score'] = np.nan

    # Merge water stress (BWS)
    if not water_df.empty and 'iso_a3' in water_df.columns:
        wat = water_df[['iso_a3', 'score']].rename(
            columns={'iso_a3': 'ISO_A3', 'score': 'bws_score'}
        )
        all_df = all_df.merge(wat, on='ISO_A3', how='left')
    else:
        all_df['bws_score'] = np.nan

    # Normalise to 0–1 (NaN left intact — exclusion handled in _build_strategic_dispatch)
    cost_min  = float(all_df['rep_cost_per_kg'].min())
    cost_max  = float(all_df['rep_cost_per_kg'].quantile(0.95))
    cost_rng  = max(cost_max - cost_min, 1e-6)
    all_df['cost_norm']     = ((all_df['rep_cost_per_kg'] - cost_min) / cost_rng).clip(0, 1)
    all_df['security_norm'] = (all_df['cpi_score'] / 100.0).clip(0, 1)
    all_df['water_norm']    = (all_df['bws_score']  / 5.0 ).clip(0, 1)

    return all_df.sort_values('rep_cost_per_kg').reset_index(drop=True)


def _build_strategic_dispatch(
    country_df: pd.DataFrame,
    demand_kt: float,
    w_cost: float,
    w_sec: float,
    w_dep: float,
    w_water: float,
    div_mode: str = 'Country',
    exempt_eu: bool = False,
) -> pd.DataFrame:
    """
    Weighted greedy dispatch.

    w_cost / w_sec / w_water feed into a per-country composite score (lower = preferred).
    w_dep controls the concentration cap: higher weight → smaller max share per entity.
    div_mode controls what counts as an entity: 'Country', 'Region', or 'Corridor'.
    """
    if country_df.empty or demand_kt <= 0:
        return pd.DataFrame()

    df = country_df.copy()

    # Exclude countries missing data for indicators that are actually weighted
    if w_sec > 0:
        df = df[df['security_norm'].notna()]
    if w_water > 0:
        df = df[df['water_norm'].notna()]
    # Fill any residual NaN norms (weight=0 case) so composite stays finite
    df['security_norm'] = df['security_norm'].fillna(0.5)
    df['water_norm']    = df['water_norm'].fillna(0.5)

    # Composite score (lower is more preferred)
    w_sum = float(w_cost + w_sec + w_water)
    if w_sum < 1e-6:
        w_sum = 1.0
    df['composite'] = (
        (w_cost  / w_sum) * df['cost_norm'] +
        (w_sec   / w_sum) * (1.0 - df['security_norm']) +
        (w_water / w_sum) * df['water_norm']
    )

    df = df.sort_values('composite').reset_index(drop=True)

    # Concentration cap derived from diversification weight (quadratic for more sensitivity)
    w_dep_frac = float(w_dep) / 100.0
    max_per_entity_kt = demand_kt * max(0.05, (1.0 - w_dep_frac) ** 2)

    _GROUP_COL = {'Country': 'ISO_A3', 'Region': 'H2_Region', 'Corridor': 'corridor_id'}
    group_col = _GROUP_COL.get(div_mode, 'ISO_A3')

    allocated_rows = []
    remaining = float(demand_kt)
    group_used: dict = {}

    # First pass: respect group concentration cap
    for _, row in df.iterrows():
        if remaining <= 1e-6:
            break
        is_eu_domestic = exempt_eu and row.get('corridor_id') == 'EU'
        if is_eu_domestic:
            avail = min(float(row['country_cap_kt']), remaining)
        else:
            key = row.get(group_col, row['ISO_A3'])
            used = group_used.get(key, 0.0)
            group_avail = max(0.0, max_per_entity_kt - used)
            avail = min(float(row['country_cap_kt']), group_avail)
            group_used[key] = used + min(remaining, avail)
        alloc = min(remaining, avail)
        if alloc > 1e-6:
            allocated_rows.append({**row.to_dict(), 'allocated_kt': alloc})
            remaining -= alloc

    # Second pass: if demand still unmet, fill without concentration cap
    if remaining > 1e-6:
        already = {r['ISO_A3']: r['allocated_kt'] for r in allocated_rows}
        for _, row in df.iterrows():
            if remaining <= 1e-6:
                break
            iso = row['ISO_A3']
            used = already.get(iso, 0.0)
            avail = float(row['country_cap_kt']) - used
            alloc = min(remaining, avail)
            if alloc > 1e-6:
                existing = next((r for r in allocated_rows if r['ISO_A3'] == iso), None)
                if existing:
                    existing['allocated_kt'] += alloc
                else:
                    allocated_rows.append({**row.to_dict(), 'allocated_kt': alloc})
                remaining -= alloc

    return pd.DataFrame(allocated_rows).reset_index(drop=True)


def _compute_strategic_kpis(
    dispatch_df: pd.DataFrame,
    demand_kt: float,
    ref_emissions_grey: float = 9.0,
    div_mode: str = 'Country',
) -> dict:
    """Compute KPIs from a strategic dispatch result."""
    _empty = {
        'delivered_cost': np.nan, 'delivered_emissions': np.nan,
        'weighted_security': np.nan, 'hhi': np.nan,
        'weighted_water': np.nan, 'total_carbon_avoided': np.nan,
        'n_countries': 0,
        'radar_cost': 0.0, 'radar_security': 0.0, 'radar_diversification': 0.0,
        'radar_water': 0.0, 'radar_carbon': 0.0,
    }
    if dispatch_df.empty or dispatch_df['allocated_kt'].sum() < 1e-6:
        return _empty

    total   = float(dispatch_df['allocated_kt'].sum())
    weights = dispatch_df['allocated_kt'] / total

    delivered_cost      = float((dispatch_df['rep_cost_per_kg'] * weights).sum())
    delivered_emissions = (
        float((dispatch_df['rep_emissions_per_kg'] * weights).sum())
        if 'rep_emissions_per_kg' in dispatch_df.columns else np.nan
    )
    weighted_security = float((dispatch_df['cpi_score'] * weights).sum())
    weighted_water    = float((dispatch_df['bws_score']  * weights).sum())
    n_countries       = int((dispatch_df['allocated_kt'] > 1e-3).sum())

    # HHI computed at the grouping level matching div_mode
    _hhi_col = {'Country': 'ISO_A3', 'Region': 'H2_Region', 'Corridor': 'corridor_id'}.get(div_mode, 'ISO_A3')
    if _hhi_col in dispatch_df.columns:
        _group_shares = dispatch_df.groupby(_hhi_col)['allocated_kt'].sum() / total
        hhi = float((_group_shares ** 2).sum())
    else:
        hhi = float((weights ** 2).sum())

    # Total carbon avoided [kt CO₂] = grey baseline emissions × demand
    # Assumes green H₂ is zero-emission; depends only on demand and the grey reference,
    # never on portfolio composition or objective weightings — gives 3 values per year.
    total_carbon_avoided = float(ref_emissions_grey * demand_kt)

    # Radar normalisation (0–1, higher = better)
    # Cost: 3.00 €/kg → 1.0 (best), 4.50 €/kg → 0.0 (worst)
    radar_cost            = float(np.clip(1.0 - (delivered_cost - 3.00) / 1.5, 0.0, 1.0))
    radar_security        = float(np.clip(weighted_security / 100.0, 0.0, 1.0))
    radar_diversification = float(np.clip(1.0 - hhi, 0.0, 1.0))
    radar_water           = float(np.clip(1.0 - weighted_water / 5.0, 0.0, 1.0))

    # Carbon: total carbon avoided normalised against 200 Mt CO₂ fixed cap
    # 0 → 0 Mt avoided, 1.0 → ≥200 Mt avoided (same scale across all scenarios)
    _CARBON_CAP_KT = 200_000.0  # 200 Mt CO₂
    radar_carbon = float(np.clip(total_carbon_avoided / _CARBON_CAP_KT, 0.0, 1.0))

    return {
        'delivered_cost':        delivered_cost,
        'delivered_emissions':   delivered_emissions,
        'weighted_security':     weighted_security,
        'hhi':                   hhi,
        'weighted_water':        weighted_water,
        'total_carbon_avoided':  total_carbon_avoided,
        'n_countries':           n_countries,
        'div_mode':              div_mode,
        'radar_cost':            radar_cost,
        'radar_security':        radar_security,
        'radar_diversification': radar_diversification,
        'radar_water':           radar_water,
        'radar_carbon':          radar_carbon,
    }


def fig_strategic_radar(kpis: dict, height: int = 420) -> go.Figure:
    """Radar / spider chart for strategic dispatch KPIs (all axes 0–1, higher = better)."""
    dc  = kpis.get('delivered_cost', np.nan)
    sec = kpis.get('weighted_security', np.nan)
    hhi = kpis.get('hhi', np.nan)
    bws = kpis.get('weighted_water', np.nan)
    tca = kpis.get('total_carbon_avoided', np.nan)  # kt CO₂

    def _fmt(val, fmt, fallback='N/A'):
        return fmt.format(val) if pd.notna(val) else fallback

    div_mode = kpis.get('div_mode', 'Country')
    div_val = (1.0 - hhi) if pd.notna(hhi) else np.nan
    tca_mt  = tca / 1000 if pd.notna(tca) else np.nan  # convert to Mt CO₂ for display

    categories = [
        f"Cost Score<br>({_fmt(dc, '{:.2f} €/kg')})",
        f"Security<br>(CPI: {_fmt(sec, '{:.0f}')})",
        f"Diversification ({div_mode})<br>(1−HHI: {_fmt(div_val, '{:.2f}')})",
        f"Water Access<br>(BWS: {_fmt(bws, '{:.2f}')})",
        f"Carbon Avoided<br>({_fmt(tca_mt, '{:.2f} MtCO₂')})",
    ]

    values = [
        kpis.get('radar_cost', 0.0),
        kpis.get('radar_security', 0.0),
        kpis.get('radar_diversification', 0.0),
        kpis.get('radar_water', 0.0),
        kpis.get('radar_carbon', 0.0),
    ]
    # Close the polygon
    categories_closed = categories + [categories[0]]
    values_closed     = values + [values[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(42, 157, 143, 0.25)',
        line=dict(color='#2A9D8F', width=2),
        name='Strategic Mix',
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickvals=[0.25, 0.5, 0.75, 1.0],
                tickfont=dict(size=9),
                gridcolor='#e0e0e0',
            ),
            angularaxis=dict(tickfont=dict(size=11)),
            bgcolor='white',
        ),
        paper_bgcolor='white',
        showlegend=False,
        height=height,
        margin=dict(l=80, r=80, t=50, b=50),
        title=dict(text='Supply Mix Performance', x=0.5, font=dict(size=14)),
    )
    return fig


def fig_strategic_source_map(
    dispatch_df: pd.DataFrame,
    demand_kt: float,
    height: int = 420,
) -> go.Figure:
    """Choropleth of source countries coloured by % of demand supplied."""
    if dispatch_df.empty:
        return go.Figure().update_layout(title='No dispatch data', height=height)

    df = dispatch_df.copy()
    df['pct_demand']   = (df['allocated_kt'] / demand_kt * 100).round(2)
    df['allocated_Mt'] = (df['allocated_kt'] / 1000).round(3)

    hover_data: dict = {'pct_demand': ':.1f', 'allocated_Mt': ':.3f'}
    if 'rep_cost_per_kg' in df.columns:
        hover_data['rep_cost_per_kg'] = ':.3f'
    if 'cpi_score' in df.columns:
        hover_data['cpi_score'] = ':.0f'
    if 'bws_score' in df.columns:
        hover_data['bws_score'] = ':.2f'

    max_pct = max(float(df['pct_demand'].max()), 1.0)

    fig = px.choropleth(
        df,
        locations='ISO_A3',
        color='pct_demand',
        color_continuous_scale='Blues',
        range_color=[0, max_pct],
        hover_name='Country',
        hover_data=hover_data,
        labels={
            'pct_demand':      '% of demand',
            'allocated_Mt':    'Mt H₂/yr',
            'rep_cost_per_kg': '€/kg',
            'cpi_score':       'CPI',
            'bws_score':       'BWS',
        },
        title='Source Countries — Strategic Dispatch',
    )
    fig.update_geos(**CHOROPLETH_GEO)
    fig.update_layout(
        paper_bgcolor='white',
        margin=dict(l=0, r=0, t=40, b=0),
        height=height,
        coloraxis_colorbar=dict(title='% of demand', thickness=15, len=0.6),
    )
    return fig


def _generate_strategic_pdf(
    scenario: str,
    year: int,
    w_cost: int,
    w_sec: int,
    w_dep: int,
    w_water: int,
    ref_cost_grey: float,
    ref_emiss_grey: float,
    strat_cap_on: bool,
    kpis: dict,
    dispatch_df: pd.DataFrame,
    demand_kt: float,
    fig_radar: go.Figure,
    fig_map: go.Figure,
) -> bytes:
    """Build a landscape A4 PDF report for the strategic dispatch tab."""
    from io import BytesIO
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image, HRFlowable, PageBreak,
    )

    buf = BytesIO()
    page_w, _ = landscape(A4)
    margin = 1.5 * cm
    usable_w = page_w - 2 * margin

    doc = SimpleDocTemplate(
        buf,
        pagesize=landscape(A4),
        leftMargin=margin, rightMargin=margin,
        topMargin=margin, bottomMargin=margin,
    )

    styles = getSampleStyleSheet()
    style_title = ParagraphStyle(
        'rpt_title', parent=styles['Heading1'],
        fontSize=18, spaceAfter=4, textColor=colors.HexColor('#1a3a5c'),
    )
    style_h2 = ParagraphStyle(
        'rpt_h2', parent=styles['Heading2'],
        fontSize=11, spaceBefore=4, spaceAfter=3,
        textColor=colors.HexColor('#1a3a5c'),
    )
    style_body = ParagraphStyle(
        'rpt_body', parent=styles['Normal'], fontSize=9, spaceAfter=2,
    )
    style_small = ParagraphStyle(
        'rpt_small', parent=styles['Normal'], fontSize=7.5,
        textColor=colors.HexColor('#666666'),
    )

    story = []

    # ── Title ──
    story.append(Paragraph('Strategic Dispatch Report', style_title))
    story.append(Paragraph(
        f'Scenario: <b>{scenario}</b> &nbsp;&nbsp;|&nbsp;&nbsp; Year: <b>{year}</b>',
        style_body,
    ))
    story.append(Spacer(1, 0.2 * cm))
    story.append(HRFlowable(width='100%', thickness=1.5, color=colors.HexColor('#2A9D8F')))
    story.append(Spacer(1, 0.25 * cm))

    # ── Settings table ──
    total_w = w_cost + w_sec + w_dep + w_water
    def _norm(w):
        return f'{100 * w / total_w:.0f}%' if total_w > 0 else '—'

    settings_data = [
        ['Weight', 'Raw / Normalised', 'Reference Parameter', 'Value'],
        ['Cost',            f'{w_cost} ({_norm(w_cost)})',   'Grey H₂ cost',      f'{ref_cost_grey:.2f} €/kg'],
        ['Security (CPI)',  f'{w_sec} ({_norm(w_sec)})',     'Grey H₂ emissions', f'{ref_emiss_grey:.1f} kgCO₂/kg'],
        ['Diversification', f'{w_dep} ({_norm(w_dep)})',     'Capacity limits',   'On' if strat_cap_on else 'Off'],
        ['Water Access',    f'{w_water} ({_norm(w_water)})', '',                  ''],
    ]
    s_tbl = Table(settings_data, colWidths=[4.0*cm, 4.2*cm, 5.2*cm, 4.0*cm])
    s_tbl.setStyle(TableStyle([
        ('BACKGROUND',     (0, 0), (-1, 0),  colors.HexColor('#1a3a5c')),
        ('TEXTCOLOR',      (0, 0), (-1, 0),  colors.white),
        ('FONTNAME',       (0, 0), (-1, 0),  'Helvetica-Bold'),
        ('FONTSIZE',       (0, 0), (-1, -1), 8.5),
        ('GRID',           (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7f7f7')]),
        ('FONTNAME',       (0, 1), (0, -1),  'Helvetica-Bold'),
        ('FONTNAME',       (2, 1), (2, -1),  'Helvetica-Bold'),
        ('VALIGN',         (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING',     (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING',  (0, 0), (-1, -1), 3),
    ]))
    story.append(s_tbl)
    story.append(Spacer(1, 0.3 * cm))
    story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#cccccc')))
    story.append(Spacer(1, 0.25 * cm))

    # ── KPI metrics ──
    story.append(Paragraph('Portfolio Performance', style_h2))

    dc      = kpis.get('delivered_cost', np.nan)
    sec_val = kpis.get('weighted_security', np.nan)
    hhi_val = kpis.get('hhi', np.nan)
    div_val = (1.0 - hhi_val) if pd.notna(hhi_val) else np.nan
    bws_val = kpis.get('weighted_water', np.nan)
    ca_val  = kpis.get('total_carbon_avoided', np.nan)  # kt CO₂
    n_c     = kpis.get('n_countries', 0)
    tot_alloc_kt = dispatch_df['allocated_kt'].sum() if not dispatch_df.empty else 0

    kpi_headers = ['Delivered Cost', 'Security (CPI)', 'Diversification (1−HHI)', 'Water Stress (BWS)', 'Total Carbon Avoided']
    kpi_values  = [
        f'{dc:.3f} €/kg'              if pd.notna(dc)      else '—',
        f'{sec_val:.0f} / 100'        if pd.notna(sec_val) else '—',
        f'{div_val:.3f}'              if pd.notna(div_val) else '—',
        f'{bws_val:.2f} / 5.0'        if pd.notna(bws_val) else '—',
        f'{ca_val / 1000:.2f} Mt CO₂' if pd.notna(ca_val)  else 'N/A',
    ]
    kpi_cw  = usable_w / 5
    kpi_tbl = Table([kpi_headers, kpi_values], colWidths=[kpi_cw] * 5)
    kpi_tbl.setStyle(TableStyle([
        ('BACKGROUND',     (0, 0), (-1, 0),  colors.HexColor('#2A9D8F')),
        ('TEXTCOLOR',      (0, 0), (-1, 0),  colors.white),
        ('FONTNAME',       (0, 0), (-1, 0),  'Helvetica-Bold'),
        ('FONTSIZE',       (0, 0), (-1, 0),  8),
        ('ALIGN',          (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME',       (0, 1), (-1, 1),  'Helvetica-Bold'),
        ('FONTSIZE',       (0, 1), (-1, 1),  12),
        ('GRID',           (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ('BACKGROUND',     (0, 1), (-1, 1),  colors.HexColor('#e8f7f5')),
        ('VALIGN',         (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING',     (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING',  (0, 0), (-1, -1), 5),
    ]))
    story.append(kpi_tbl)
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph(
        f'Sourcing from <b>{n_c} countries</b> &nbsp;·&nbsp; '
        f'Allocated: <b>{tot_alloc_kt / 1000:.2f} Mt H₂/yr</b> &nbsp;·&nbsp; '
        f'Demand: {demand_kt / 1000:.2f} Mt H₂/yr',
        style_small,
    ))
    story.append(Spacer(1, 0.3 * cm))
    story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#cccccc')))
    story.append(Spacer(1, 0.2 * cm))

    # ── Charts (radar + map side by side) ──
    chart_w = (usable_w - 0.4 * cm) / 2
    chart_h = chart_w * 0.68

    from io import BytesIO as _BytesIO
    radar_img_bytes = fig_radar.to_image(format='png', width=700, height=480, scale=2)
    map_img_bytes   = fig_map.to_image(format='png', width=700, height=480, scale=2)

    chart_tbl = Table(
        [[Image(_BytesIO(radar_img_bytes), width=chart_w, height=chart_h),
          Image(_BytesIO(map_img_bytes),   width=chart_w, height=chart_h)]],
        colWidths=[chart_w, chart_w],
    )
    chart_tbl.setStyle(TableStyle([
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING',   (0, 0), (-1, -1), 0),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 0),
        ('TOPPADDING',    (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('INNERGRID',     (0, 0), (-1, -1), 0, colors.white),
        ('BOX',           (0, 0), (-1, -1), 0, colors.white),
    ]))
    story.append(chart_tbl)

    # ── Dispatch detail table (second page) ──
    if not dispatch_df.empty:
        story.append(PageBreak())
        story.append(Paragraph('Dispatch Detail', style_h2))
        story.append(Paragraph(
            f'Scenario: <b>{scenario}</b> &nbsp;|&nbsp; Year: <b>{year}</b>',
            style_body,
        ))
        story.append(Spacer(1, 0.2 * cm))

        _disp = dispatch_df.copy()
        _disp['allocated_Mt'] = (_disp['allocated_kt'] / 1000).round(3)
        _disp['pct_demand']   = (_disp['allocated_kt'] / demand_kt * 100).round(2)

        col_order = {
            'Country':              'Country',
            'ISO_A3':               'ISO',
            'corridor_id':          'Corridor',
            'allocated_Mt':         'Alloc (Mt/yr)',
            'pct_demand':           '% Demand',
            'rep_cost_per_kg':      'Cost (€/kg)',
            'rep_emissions_per_kg': 'Emissions (kg)',
            'cpi_score':            'CPI',
            'bws_score':            'BWS',
        }
        present  = [c for c in col_order if c in _disp.columns]
        headers_d = [col_order[c] for c in present]

        def _fmt(val):
            if pd.isna(val): return '—'
            if isinstance(val, float): return f'{val:.2f}'
            return str(val)

        rows_d = [[_fmt(row[c]) for c in present] for _, row in _disp.iterrows()]
        cw_d   = usable_w / len(present)
        d_tbl  = Table([headers_d] + rows_d, colWidths=[cw_d] * len(present), repeatRows=1)
        d_tbl.setStyle(TableStyle([
            ('BACKGROUND',     (0, 0), (-1, 0),  colors.HexColor('#1a3a5c')),
            ('TEXTCOLOR',      (0, 0), (-1, 0),  colors.white),
            ('FONTNAME',       (0, 0), (-1, 0),  'Helvetica-Bold'),
            ('FONTSIZE',       (0, 0), (-1, -1), 8),
            ('GRID',           (0, 0), (-1, -1), 0.3, colors.lightgrey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            ('ALIGN',          (1, 0), (-1, -1), 'CENTER'),
            ('VALIGN',         (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING',     (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING',  (0, 0), (-1, -1), 3),
        ]))
        story.append(d_tbl)

    # ── Footer ──
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(
        f'Generated {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")} · '
        'Green Hydrogen EHB Corridor Analysis',
        style_small,
    ))

    doc.build(story)
    return buf.getvalue()


# ---------------------------------------------------------------------------

@st.cache_data(show_spinner='Loading capacity data…')
def _load_caps_df() -> pd.DataFrame:
    """Load combined_caps_by_year.csv, returning empty DataFrame if missing."""
    path = 'Data/combined_caps_by_year.csv'
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df['Year'] = df['Year'].astype(int)
    return df


@st.cache_data(show_spinner=False)
def _get_caps_for_year(year: int) -> dict:
    """Return {ISO_A3: capacity_kt} for the given year from the caps CSV."""
    caps_df = _load_caps_df()
    if caps_df.empty:
        return {}
    rows = caps_df[caps_df['Year'] == year]
    if rows.empty:
        return {}
    return dict(zip(rows['ISO_A3'], rows['Capacity_kt']))


def fig_capacity_map(caps_df: pd.DataFrame, year: int) -> go.Figure:
    """Choropleth of green H₂ export capacity by country for a given year."""
    df_yr = caps_df[caps_df['Year'] == year].copy()
    if df_yr.empty:
        return go.Figure()

    df_yr['log_cap'] = np.log10(df_yr['Capacity_kt'].clip(lower=1))
    log_max = float(np.log10(caps_df['Capacity_kt'].max()) * 1.05)

    tick_vals = [0, 1, 2, 3, 4, 5]
    tick_text  = ['1', '10', '100', '1k', '10k', '100k']

    fig = px.choropleth(
        df_yr,
        locations='ISO_A3',
        color='log_cap',
        range_color=[0, log_max],
        color_continuous_scale='Greens',
        hover_name='ISO_A3',
        hover_data={'Capacity_kt': ':,.0f', 'Source': True, 'log_cap': False},
        labels={
            'log_cap':      'kt H₂/yr (log)',
            'Capacity_kt':  'kt H₂/yr',
            'Source':       'Capacity source',
        },
        title=f'Green H₂ export capacity by country — {year}',
    )
    fig.update_geos(
        showframe=False,
        showcoastlines=True, coastlinecolor='#aaaaaa',
        showland=True,  landcolor='#f5f5f5',
        showocean=True, oceancolor='#d0e8f5',
        projection_type='natural earth',
        lataxis_range=[-60, 90],
    )
    fig.update_layout(
        paper_bgcolor='white',
        font=dict(size=12),
        margin=dict(l=0, r=0, t=60, b=0),
        height=500,
        coloraxis_colorbar=dict(
            title='kt H₂/yr',
            thickness=15,
            len=0.6,
            tickvals=tick_vals,
            ticktext=tick_text,
        ),
    )
    return fig


def fig_transport_mode_pie(dfs_filtered: dict, show_corridors: list[str]) -> go.Figure:
    """Pie chart: proportion of source locations using each transport mode."""
    frames = []
    for cid in show_corridors:
        df = dfs_filtered.get(cid)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return go.Figure()

    combined = pd.concat(frames, ignore_index=True)
    if 'Cheapest Medium' not in combined.columns:
        return go.Figure()

    counts = combined['Cheapest Medium'].value_counts()
    colour_map = {
        'NH3':    '#0072B2',
        'LOHC':   '#E69F00',
        'H2 Gas': '#009E73',
        'H2 Liq': '#CC79A7',
    }
    colours = [colour_map.get(m, '#888') for m in counts.index]

    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        marker=dict(colors=colours),
        hole=0.35,
    ))
    fig.update_layout(title='Transport Mode Share (within-cap source points)', height=350)
    return fig


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _cost_at_half_demand(df: pd.DataFrame, demand_kt: float) -> float:
    """
    Weighted-average delivered cost when sourcing the cheapest demand/2 kt from
    this corridor (merit-order approach).

    Groups within-cap grid points by country, sorts country blocks by mean cost
    (cheapest first), and accumulates capacity until demand/2 is filled.
    Returns the volume-weighted average cost over those locations.
    If the corridor cannot supply demand/2, returns the weighted average over
    all available supply.
    """
    half = demand_kt / 2.0
    if half <= 0 or 'country_cap_kt' not in df.columns or 'Total Cost per kg H2' not in df.columns:
        return np.nan
    valid = df.dropna(subset=['Total Cost per kg H2', 'country_cap_kt'])
    if valid.empty:
        return np.nan
    country_grp = (
        valid.groupby('Country')
             .agg(cap_kt=('country_cap_kt', 'first'),
                  mean_cost=('Total Cost per kg H2', 'mean'))
             .dropna()
             .sort_values('mean_cost')
    )
    if country_grp.empty:
        return np.nan
    cum, wtd_sum = 0.0, 0.0
    for _, row in country_grp.iterrows():
        if cum >= half:
            break
        allocated = min(row['cap_kt'], half - cum)
        wtd_sum += allocated * row['mean_cost']
        cum += allocated
    return wtd_sum / cum if cum > 0 else np.nan


def build_summary(dfs_filtered: dict, show_corridors: list[str],
                  red3_threshold: float, demand_kt: float = 0.0) -> pd.DataFrame:
    rows = []
    for cid in show_corridors:
        df = dfs_filtered.get(cid)
        if df is None or df.empty:
            continue
        meta       = CORRIDOR_META.get(cid, {})
        valid_cost = df.dropna(subset=['Total Cost per kg H2'])
        valid_emis = df.dropna(subset=['Total Emissions per kg H2'])

        pct_below = (
            (valid_emis['Total Emissions per kg H2'] < red3_threshold).mean() * 100
            if not valid_emis.empty else np.nan
        )
        dom_mode = (
            df['Cheapest Medium'].value_counts().idxmax()
            if 'Cheapest Medium' in df.columns and not df.empty else '—'
        )
        half_dem_cost = (
            _cost_at_half_demand(df, demand_kt) if demand_kt > 0 else np.nan
        )

        rows.append({
            'Corridor':                   f'Corridor {cid}',
            'Points (within cap)':        len(df),
            'Half-demand cost (€/kg)':    round(half_dem_cost, 2) if pd.notna(half_dem_cost) else np.nan,
            'Min cost (€/kg)':            round(valid_cost['Total Cost per kg H2'].min(), 2)
                                          if not valid_cost.empty else np.nan,
            'Median cost (€/kg)':         round(valid_cost['Total Cost per kg H2'].median(), 2)
                                          if not valid_cost.empty else np.nan,
            'Med. emissions':             round(valid_emis['Total Emissions per kg H2'].median(), 3)
                                          if not valid_emis.empty else np.nan,
            '% below RED III':            round(pct_below, 1),
            'Dom. mode':                  dom_mode,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Flow map (reuses plot_corridor._build_flow_map_fig)
# ---------------------------------------------------------------------------

def fig_flow_map(dfs_filtered: dict, show_corridors: list[str],
                 h2_demand_kt: float, year: int,
                 n_countries: int = 15) -> go.Figure:
    """Reconstruct results dict from loaded DataFrames and build the flow map."""
    results = {}
    for cid in show_corridors:
        df = dfs_filtered.get(cid)
        if df is None or df.empty:
            continue
        cfg = CORRIDORS.get(cid)
        if cfg is None:
            continue
        results[cid] = {'corridor': cfg, 'df': df, 'n_global': None}
    if not results:
        return go.Figure().update_layout(title='No corridor data available for flow map')
    return _build_flow_map_fig(results, n_countries=n_countries,
                               h2_demand=h2_demand_kt, year=year)


# ---------------------------------------------------------------------------
# Assumption / diagnostic chart functions
# ---------------------------------------------------------------------------

_REGION_ORDER_DIAG = [
    'EU', 'Non-EU Europe', 'North Africa', 'Sub-Saharan Africa',
    'Middle East', 'Central Asia', 'South Asia', 'East Asia',
    'Southeast Asia', 'Oceania', 'North America', 'Latin America', 'Russia', 'Other',
]


def fig_capex_assumptions(selected_year: int, elec_type: str = 'alkaline') -> go.Figure:
    """Line chart of global baseline CAPEX trajectories 2025–2050 with selected year highlighted."""
    years = list(range(2025, 2051))
    params = [global_capex(y, elec_type) for y in years]
    wind_vals  = [p['capex_wind']       for p in params]
    solar_vals = [p['capex_solar']      for p in params]
    elec_vals  = [p['capex_elec']       for p in params]
    if elec_type == 'alkaline':
        eff_vals = [min(65.0 + 0.5 * max(0, y - 2026), 65.0 + 0.5 * 24) for y in years]
    else:
        eff_vals   = [p['efficiency'] * 100 for p in params]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Onshore wind CAPEX (EUR/kW)',
            'Solar PV CAPEX (EUR/kWp)',
            f'{elec_type.title()} electrolyser CAPEX (EUR/kW)',
            f'{elec_type.title()} electrolyser efficiency (LHV %)',
        ),
        horizontal_spacing=0.12, vertical_spacing=0.20,
    )

    panels = [
        ((1, 1), wind_vals,  '#457B9D'),
        ((1, 2), solar_vals, '#E69F00'),
        ((2, 1), elec_vals,  '#2A9D8F'),
        ((2, 2), eff_vals,   '#E63946'),
    ]
    for (row, col), vals, colour in panels:
        fig.add_trace(go.Scatter(
            x=years, y=vals, mode='lines',
            line=dict(width=2, color=colour), showlegend=False,
        ), row=row, col=col)
        # Highlight selected year
        if selected_year in years:
            idx = years.index(selected_year)
            fmt = f'{vals[idx]:.1f}%' if (row, col) == (2, 2) else f'{vals[idx]:.0f}'
            fig.add_trace(go.Scatter(
                x=[selected_year], y=[vals[idx]],
                mode='markers+text',
                marker=dict(size=10, color=colour, symbol='diamond'),
                text=[fmt], textposition='top center',
                showlegend=False,
            ), row=row, col=col)
        fig.add_vline(x=2030, line_dash='dot', line_color='#aaaaaa', row=row, col=col)

    fig.update_xaxes(title_text='Year')
    fig.update_layout(
        title=(f'<b>Global baseline CAPEX — {selected_year} highlighted</b><br>'
               '<sup>Pre-regional-adjustment baselines. '
               'Actual cost per grid point = baseline × region factor, annualised at region WACC.</sup>'),
        plot_bgcolor='white', paper_bgcolor='white', font=dict(size=11),
        height=480,
    )
    return fig


def fig_regional_factors_dash() -> go.Figure:
    """Bar chart of solar/wind CAPEX multipliers per H2_Region."""
    regions = _REGION_ORDER_DIAG
    solar_f = [SOLAR_CAPEX_FACTOR.get(r, 1.0) for r in regions]
    wind_f  = [WIND_CAPEX_FACTOR.get(r, 1.0)  for r in regions]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Solar CAPEX factor (× baseline)', 'Wind CAPEX factor (× baseline)'),
        horizontal_spacing=0.10,
    )
    for col, vals, colour in [
        (1, solar_f, '#E69F00'),
        (2, wind_f,  '#457B9D'),
    ]:
        fig.add_trace(go.Bar(
            x=regions, y=vals, marker_color=colour, opacity=0.85, showlegend=False,
        ), row=1, col=col)
        fig.add_hline(y=1.0, line_dash='dot', line_color='grey', row=1, col=col)

    fig.update_xaxes(tickangle=45)
    fig.update_layout(
        title='<b>Regional CAPEX multipliers</b>',
        plot_bgcolor='white', paper_bgcolor='white', font=dict(size=11),
        height=380,
    )
    return fig


def fig_wacc_maps() -> go.Figure:
    """Side-by-side choropleth maps of WACC by country for each technology."""
    ren_iso  = list(WACC_COUNTRY_REN.keys())
    ren_vals = [v * 100 for v in WACC_COUNTRY_REN.values()]
    elec_iso  = list(WACC_COUNTRY_ELEC.keys())
    elec_vals = [v * 100 for v in WACC_COUNTRY_ELEC.values()]

    shared_range = [0, max(max(ren_vals, default=0), max(elec_vals, default=0))]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Renewable WACC — Solar & Wind (%)', 'Electrolyser WACC (%)'),
        specs=[[{'type': 'choropleth'}, {'type': 'choropleth'}]],
        horizontal_spacing=0.02,
    )

    for col, iso_list, vals, title in [
        (1, ren_iso,  ren_vals,  'Ren. WACC (%)'),
        (2, elec_iso, elec_vals, 'Elec. WACC (%)'),
    ]:
        fig.add_trace(go.Choropleth(
            locations=iso_list,
            z=vals,
            locationmode='ISO-3',
            colorscale='RdYlGn_r',
            zmin=shared_range[0],
            zmax=shared_range[1],
            colorbar=dict(title=title, len=0.7, thickness=12,
                          x=0.48 if col == 1 else 1.0),
            hovertemplate='<b>%{location}</b><br>WACC: %{z:.1f}%<extra></extra>',
            showscale=True,
        ), row=1, col=col)

    fig.update_geos(
        showland=True, landcolor='#f5f5f5',
        showocean=True, oceancolor='#d0e8f5',
        showframe=False, showcoastlines=True, coastlinecolor='#aaaaaa',
        projection_type='natural earth',
        lataxis_range=[-60, 90],
    )
    fig.update_layout(
        title='<b>WACC by country and technology</b>',
        paper_bgcolor='white', font=dict(size=11),
        height=420,
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


def fig_gen_cost_by_region(dfs_filtered: dict, show_corridors: list) -> go.Figure:
    """Stacked bar of mean generation cost components per region from loaded corridor data."""
    frames = []
    for cid in show_corridors:
        df = dfs_filtered.get(cid)
        if df is not None and not df.empty and 'H2_Region' in df.columns:
            frames.append(df)
    if not frames:
        return go.Figure()

    combined = pd.concat(frames, ignore_index=True)
    regions = [r for r in _REGION_ORDER_DIAG if r in combined['H2_Region'].unique()]
    has_components = all(c in combined.columns for c in
                         ['Yearly Cost Solar', 'Yearly Cost Wind', 'Yearly Cost Electrolyser',
                          'Yearly gen. cost', 'Gen. cost per kg H2'])

    if has_components:
        valid = combined[combined['Gen. cost per kg H2'] > 0]
        demand_kg = (valid['Yearly gen. cost'] / valid['Gen. cost per kg H2']).median()
        combined = combined.copy()
        combined['_elec_kg']  = combined[['Yearly Cost Solar', 'Yearly Cost Wind']].min(axis=1) / demand_kg
        combined['_water_kg'] = 0.07
        combined['_elec_capex_kg'] = (combined['Yearly Cost Electrolyser'] / demand_kg - 0.07).clip(lower=0)

        fig = go.Figure()
        for comp, colour, label in [
            ('_elec_kg',       '#009E73', 'Electricity (renewable LCOE)'),
            ('_elec_capex_kg', '#0072B2', 'Electrolyser CAPEX + OPEX'),
            ('_water_kg',      '#56B4E9', 'Water'),
        ]:
            vals = [combined[combined['H2_Region'] == r][comp].mean() for r in regions]
            fig.add_trace(go.Bar(name=label, x=regions, y=vals, marker_color=colour, opacity=0.85))
        fig.update_layout(barmode='stack',
                          title='<b>Generation cost breakdown by region</b> (mean across selected corridors)')
    else:
        vals = [combined[combined['H2_Region'] == r]['Gen. cost per kg H2'].mean() for r in regions]
        fig = go.Figure(go.Bar(x=regions, y=vals, marker_color='#457B9D', opacity=0.85))
        fig.update_layout(title='<b>Mean generation cost by region</b> (selected corridors)')

    fig.update_layout(
        xaxis=dict(tickangle=45),
        yaxis_title='EUR / kg H₂',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        plot_bgcolor='white', paper_bgcolor='white', font=dict(size=11),
        height=420,
    )
    return fig


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title='European Union Green Hydrogen Supply Dashboard',
        layout='wide',
    )

    st.title('European Union Green Hydrogen Supply Dashboard')
    st.caption(
        'Transport costs pre-computed once; generation costs calculated live from CAPEX sliders. '
        'Adjust Solar, Wind, and Electrolyser CAPEX to explore cost sensitivity in real time.'
    )

    # ── Discover available (scenario, year) combinations ─────────────────────
    available = discover_available()

    if not available:
        st.error(
            f'No corridor CSVs found in `{RESULTS_DIR}/`. '
            'Run `run_corridors.py` first to generate results '
            '(files are saved as `corridor_A_2035_Intermediate.csv` etc.).'
        )
        st.stop()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header('Controls')

        # Scenario selector — determines which pre-computed files are loaded
        st.subheader('Scenario')
        available_scenarios = sorted(available.keys(),
                                     key=lambda s: list(DEMAND_PROFILES.keys()).index(s)
                                     if s in DEMAND_PROFILES else 99)
        default_scenario = 'Intermediate' if 'Intermediate' in available_scenarios else available_scenarios[0]
        selected_scenario = st.radio(
            'Demand scenario',
            options=available_scenarios,
            index=available_scenarios.index(default_scenario),
            help='Load pre-computed results for this scenario.',
        )

        st.divider()

        # Year selector — filtered to years available for the selected scenario
        st.subheader('Year')
        years_for_scenario = available.get(selected_scenario, [])
        selected_year = st.selectbox(
            'Model year',
            options=years_for_scenario,
            index=len(years_for_scenario) - 1,
        )

        h2_demand_kt = demand_for_year(selected_scenario, selected_year)
        h2_demand_mt = h2_demand_kt / 1_000
        st.metric(
            label=f'Demand ({selected_scenario}, {selected_year})',
            value=f'{h2_demand_mt:.1f} Mt H₂/yr',
            help='From the scenario demand profile for the selected year.',
        )

        # Show full profile for context
        with st.expander('Full scenario profile', expanded=False):
            profile_rows = [
                {'Year': yr, **{s: f'{v/1000:.1f} Mt' for s, v in
                  {sc: DEMAND_PROFILES[sc][yr] for sc in DEMAND_PROFILES}.items()}}
                for yr in sorted(DEMAND_PROFILES['Ambitious'].keys())
            ]
            st.dataframe(pd.DataFrame(profile_rows), hide_index=True)

        st.divider()

        # ── Load data for selected scenario + year ───────────────────────────
        all_dfs = load_corridors(selected_year, selected_scenario)
        available_corridors = sorted(all_dfs.keys())
        show_corridors = available_corridors

        cap_on = True

        st.subheader('CAPEX Sensitivity')

        _elec_type = 'alkaline'

        _defaults = global_capex(selected_year, _elec_type)

        _solar_key = f'solar_capex_{selected_year}_{_elec_type}'
        _wind_key  = f'wind_capex_{selected_year}_{_elec_type}'
        _elec_key  = f'elec_capex_{selected_year}_{_elec_type}'
        _trans_key = 'transport_adj_pct'

        if st.button('Reset to defaults', use_container_width=True):
            st.session_state[_solar_key] = int(_defaults['capex_solar'])
            st.session_state[_wind_key]  = int(_defaults['capex_wind'])
            st.session_state[_elec_key]  = int(_defaults['capex_elec'])
            st.session_state[_trans_key] = 0
            st.rerun()

        solar_capex = st.slider(
            'Solar CAPEX (€/kWp)',
            min_value=100, max_value=1200,
            value=int(_defaults['capex_solar']), step=10,
            key=f'solar_capex_{selected_year}_{_elec_type}',
            help=f'Global baseline. Year {selected_year} default: {_defaults["capex_solar"]:.0f} €/kWp. '
                 'Regional multipliers still applied.',
        )
        wind_capex = st.slider(
            'Wind CAPEX (€/kW)',
            min_value=400, max_value=2000,
            value=int(_defaults['capex_wind']), step=25,
            key=f'wind_capex_{selected_year}_{_elec_type}',
            help=f'Global baseline (onshore). Year {selected_year} default: {_defaults["capex_wind"]:.0f} €/kW.',
        )
        elec_capex = st.slider(
            'Electrolyser CAPEX (€/kW)',
            min_value=100, max_value=1500,
            value=int(_defaults['capex_elec']), step=25,
            key=f'elec_capex_{selected_year}_{_elec_type}',
            help=f'Year {selected_year} default: {_defaults["capex_elec"]:.0f} €/kW.',
        )
        transport_adj_pct = st.slider(
            'Transport cost adjustment (%)',
            min_value=-50, max_value=50,
            value=0, step=5, format='%d%%',
            key=_trans_key,
            help='Scales all transport costs for sensitivity analysis.',
        )
        transport_adj_frac = 1.0 + transport_adj_pct / 100.0

        st.divider()

        if transport_adj_pct != 0:
            st.info(f'Transport cost adjusted {transport_adj_pct:+d}%.')
        else:
            st.caption('Generation costs calculated live from CAPEX sliders. '
                       'Run `python run_corridors.py --base-only` to refresh base transport data.')

    if not all_dfs:
        st.error(f'No corridor base files found in `{RESULTS_DIR}/`.')
        st.stop()

    # ── Compute generation costs live from CAPEX sliders ─────────────────────
    dfs_with_gen = {}
    for cid, df_base in all_dfs.items():
        dfs_with_gen[cid] = compute_gen_costs(
            df_base, selected_year, _elec_type, solar_capex, wind_capex, elec_capex
        )

    # ── Apply transport adjustment + capacity filters ─────────────────────────
    dfs_filtered = {}
    for cid, df in dfs_with_gen.items():
        dfs_filtered[cid] = apply_filters(df, cap_on, h2_demand_kt, selected_year, transport_adj_frac)

    if not show_corridors:
        st.warning('Select at least one corridor in the sidebar.')
        st.stop()

    # ── Build country-level supply data for merit-order chart & optimal mix ──
    caps_for_year = _get_caps_for_year(selected_year)
    results_with_country = {}
    for cid in show_corridors:
        df_c = dfs_filtered.get(cid)
        if df_c is None or df_c.empty:
            continue
        cfg = CORRIDORS.get(cid)
        if cfg is None:
            continue
        df_country = aggregate_country_supply(df_c, caps_for_year)
        results_with_country[cid] = {
            'corridor': cfg, 'df': df_c, 'df_country': df_country, 'n_global': None
        }

    # Optimal mix weighted-average cost (for main summary metric)
    _alloc = _build_optimal_mix(results_with_country, h2_demand_kt)
    opt_avg_cost = np.nan
    if not _alloc.empty and _alloc['allocated_kt'].sum() > 0:
        opt_avg_cost = float(
            (_alloc['cost'] * _alloc['allocated_kt']).sum()
            / _alloc['allocated_kt'].sum()
        )

    # ── Summary metrics ──────────────────────────────────────────────────────
    st.subheader('Summary')
    summary_df = build_summary(dfs_filtered, show_corridors, EU_RED3_DEFAULT, h2_demand_kt)

    # Primary summary: optimal supply mix average cost
    if pd.notna(opt_avg_cost):
        st.metric(
            label='Optimal Supply Mix — Weighted Avg. Delivered Cost',
            value=f'{opt_avg_cost:.3f} €/kg H₂',
            help=(
                f'Volume-weighted average cost of the cheapest global merit-order dispatch. '
                f'Greedy allocation of {h2_demand_mt:.1f} Mt H₂/yr across all selected '
                'corridors — cheapest countries dispatched first, respecting national '
                'capacity caps. This is the minimum achievable average import cost.'
            ),
        )

    # Per-corridor delivered cost at half-demand
    if not summary_df.empty:
        cols = st.columns(len(show_corridors))
        for col_st, (_, row) in zip(cols, summary_df.iterrows()):
            with col_st:
                cid_label  = row['Corridor'].split(':')[0].strip()
                half_cost  = row.get('Half-demand cost (€/kg)', np.nan)
                cost_val   = half_cost if pd.notna(half_cost) else row.get('Median cost (€/kg)', np.nan)
                label_text = 'Half-demand cost' if pd.notna(half_cost) else 'Median cost'
                st.metric(
                    label=row['Corridor'],
                    value=f"{cost_val:.2f} €/kg" if pd.notna(cost_val) else '—',
                    help=(
                        f"{label_text}: weighted-average delivered cost when sourcing "
                        f"the cheapest {h2_demand_mt/2:.1f} Mt H₂/yr from Corridor {cid_label} "
                        "(merit-order, cheapest countries first)."
                        if pd.notna(half_cost) else
                        f"Median total cost for Corridor {cid_label}."
                    ),
                )

        with st.expander('Full summary table', expanded=False):
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Main charts ──────────────────────────────────────────────────────────
    tab_supply, tab_strategic, tab_maps, tab_caps, tab_flow, tab_assumptions, tab_projects = st.tabs([
        'Supply Curve',
        'Strategic Dispatch',
        'Source Maps',
        'Country Caps',
        'Flow Map',
        'Assumptions',
        'H₂ Projects Pipeline',
    ])

    with tab_supply:
        st.plotly_chart(
            fig_supply_curve(results_with_country, h2_demand_kt, selected_year),
            use_container_width=True,
        )
        st.caption(
            'Steps ordered cheapest-first per corridor; width = national capacity cap, '
            'height = trimmed mean cost (P10–P90, excludes cheapest and most expensive 10% of grid points). '
            'Thick lines = optimal global dispatch. Countries without a capacity estimate are excluded.'
        )

    with tab_maps:
        # ── Port selector (affects Total Cost, Transport Cost, Transport Mode maps) ──
        _port_keys  = list(PORT_OPTIONS.keys())
        _port_labels = list(PORT_OPTIONS.values())
        _selected_port_label = st.selectbox(
            'Import port (affects Total Cost, Transport Cost, and Transport Mode maps)',
            options=_port_labels,
            index=0,
            key='tab4_port_select',
        )
        _selected_port_key = _port_keys[_port_labels.index(_selected_port_label)]

        _port_df_raw = load_port_results(_selected_port_key, selected_year)
        if not _port_df_raw.empty:
            # Compute generation costs live (same CAPEX sliders as corridor data)
            _port_df = compute_gen_costs(
                _port_df_raw, selected_year, _elec_type, solar_capex, wind_capex, elec_capex
            )
            _port_df = adjust_transport_costs(_port_df, transport_adj_frac)
            if cap_on and 'within_cap' in _port_df.columns:
                _port_df = _port_df[_port_df['within_cap'] == True].copy()
        else:
            _port_df = _port_df_raw.copy()

        st.divider()

        # Row 1: Total Cost (port) — full width
        if _port_df.empty:
            st.warning(f'No port data for {_selected_port_label}, {selected_year}.')
        else:
            st.plotly_chart(
                fig_port_source_map(_port_df, 'Total Cost (€/kg H₂)',
                                    _selected_port_label, height=380),
                use_container_width=True,
            )

        # Row 2: Generation Cost (corridor) | Transport Cost (port)
        _r2c1, _r2c2 = st.columns(2)
        with _r2c1:
            st.plotly_chart(
                fig_source_map(dfs_with_gen, show_corridors,
                               'Generation Cost (€/kg H₂)', height=380),
                use_container_width=True,
            )
        with _r2c2:
            if not _port_df.empty:
                st.plotly_chart(
                    fig_port_source_map(_port_df, 'Transport Cost (€/kg H₂)',
                                        _selected_port_label, height=380),
                    use_container_width=True,
                )

        # Row 3: Cheapest Transport Mode (port) | Energy Source (corridor)
        _r3c1, _r3c2 = st.columns(2)
        with _r3c1:
            if not _port_df.empty:
                st.plotly_chart(
                    fig_port_source_map(_port_df, 'Cheapest Transport Mode',
                                        _selected_port_label, height=380),
                    use_container_width=True,
                )
        with _r3c2:
            st.plotly_chart(
                fig_source_map(dfs_with_gen, show_corridors,
                               'Cheaper Energy Source', height=380),
                use_container_width=True,
            )

        # Row 4: Security | Water Stress
        _r4c1, _r4c2 = st.columns(2)
        with _r4c1:
            st.plotly_chart(fig_security_map(height=380), use_container_width=True)
        with _r4c2:
            st.plotly_chart(fig_water_stress_map(height=380), use_container_width=True)

    with tab_caps:
        caps_df = _load_caps_df()
        if caps_df.empty:
            st.warning(
                '`Data/combined_caps_by_year.csv` not found. '
                'Run `run_corridors.py` (or `python plot_capacity.py`) to generate it.'
            )
        else:
            cap_years = sorted(caps_df['Year'].unique())
            default_cap_year = selected_year if selected_year in cap_years else cap_years[-1]
            cap_year = st.slider(
                'Year',
                min_value=int(cap_years[0]),
                max_value=int(cap_years[-1]),
                value=int(default_cap_year),
                step=1,
                key='cap_year_slider',
            )
            st.plotly_chart(fig_capacity_map(caps_df, cap_year), use_container_width=True)
            st.caption(
                'Capacity = theoretical potential × readiness(year). '
                'Readiness anchored to IEA 2030 pipeline, ramping to 100% by 2050. Log scale.'
            )
            with st.expander(f'Top 20 countries by capacity — {cap_year}', expanded=False):
                top20 = (
                    caps_df[caps_df['Year'] == cap_year]
                    .sort_values('Capacity_kt', ascending=False)
                    .head(20)[['ISO_A3', 'Capacity_kt', 'Source']]
                    .reset_index(drop=True)
                )
                top20.columns = ['Country (ISO A3)', 'Capacity (kt H₂/yr)', 'Source']
                st.dataframe(top20, hide_index=True, use_container_width=True)

    with tab_flow:
        n_countries_flow = st.slider(
            'Top N source countries per corridor',
            min_value=5, max_value=30, value=15, step=1,
            key='flow_n_countries',
        )
        st.plotly_chart(
            fig_flow_map(dfs_filtered, show_corridors, h2_demand_kt,
                         selected_year, n_countries=n_countries_flow),
            use_container_width=True,
        )
        st.caption(
            'Flow lines from within-cap supply countries to corridor EU entry points (stars). '
            'Dot colour = transport medium; dot size ∝ allocated volume.'
        )

    with tab_assumptions:
        st.subheader('Technology Cost Assumptions')
        st.caption(
            'Global baseline CAPEX before regional adjustment. '
            'Actual cost = baseline × region factor, annualised at region WACC. '
            '2030 marks the learning-rate inflection.'
        )

        elec_type_diag = 'alkaline'
        st.plotly_chart(
            fig_capex_assumptions(selected_year, elec_type_diag),
            use_container_width=True,
        )

        st.divider()
        st.subheader('Regional Adjustment Factors')
        st.caption(
            'Solar/wind CAPEX multiplied by region factors; WACC applied to annualisation (CRF).'
        )
        st.plotly_chart(fig_regional_factors_dash(), use_container_width=True)

        st.divider()
        st.subheader('WACC by Country and Technology')
        st.caption(
            'Weighted average cost of capital (%) used to annualise CAPEX for each technology. '
            'Renewable WACC applies to solar and wind; electrolyser WACC carries an additional '
            'technology risk premium (~2%).'
        )
        st.plotly_chart(fig_wacc_maps(), use_container_width=True)

        st.divider()
        st.subheader('Generation Cost by Region (current selection)')
        st.caption(
            'Mean gen. cost per region across within-cap grid points, split by component.'
        )
        st.plotly_chart(
            fig_gen_cost_by_region(dfs_filtered, show_corridors),
            use_container_width=True,
        )

    with tab_strategic:
        st.header('Strategic Multi-Criteria Dispatch')
        st.caption(
            'Adjust the objective weights to explore trade-offs between cost, energy security, '
            'supply diversification, and water stress. The dispatch model re-ranks source '
            'countries according to a weighted composite score and applies a concentration cap '
            'based on the diversification weight.'
        )

        strat_cap_on = st.toggle(
            'Apply capacity limits',
            value=cap_on,
            key='strat_cap_on',
            help='When off, all countries are eligible regardless of national production targets. '
                 'Overrides the global capacity-limits setting for this tab only.',
        )

        # ── Layout: left = controls, right = charts ──
        left_col, right_col = st.columns([1, 2])
        with left_col:
            st.subheader('Objective Weights')
            st.caption('Weights are auto-normalised — they need not sum to 100.')
            w_cost  = st.slider('Cost',                    0, 100, 25, key='strat_w_cost')
            w_sec   = st.slider('Security (CPI)',           0, 100, 25, key='strat_w_sec')
            w_water = st.slider('Water Access',             0, 100, 25, key='strat_w_water')
            div_mode = st.selectbox(
                'Diversification by',
                options=['Country', 'Region', 'Corridor'],
                index=0,
                key='strat_div_mode',
                help='Apply the diversification cap to individual countries, H₂ production regions, or supply corridors.',
            )
            w_dep   = st.slider('Diversification',          0, 100, 25, key='strat_w_dep',
                                help='Controls maximum share any single country/region/corridor can supply. '
                                     'At 100 no single entity supplies more than 5% of demand.')
            _dep_cap_pct = max(5.0, (1.0 - w_dep / 100.0) ** 2 * 100)
            st.caption(f'Max per {div_mode.lower()}: **{_dep_cap_pct:.0f}%** of demand')
            exempt_eu = st.checkbox(
                'Exempt EU domestic production',
                value=True,
                key='strat_exempt_eu',
                help='EU domestic supply is not subject to the diversification cap — '
                     'it represents home production, not dependency on a foreign supplier.',
            )
            total_w = w_cost + w_sec + w_dep + w_water
            st.caption(f'Sum of weights: {total_w}')

        # Carbon Reference inputs rendered below after charts; read current values from session state
        ref_cost_grey  = float(st.session_state.get('strat_ref_cost',  1.50))
        ref_emiss_grey = float(st.session_state.get('strat_ref_emiss', 9.0))

        # ── Run weighted dispatch ──
        if strat_cap_on != cap_on:
            dfs_strat = {
                cid: apply_filters(df, strat_cap_on, h2_demand_kt, selected_year, transport_adj_frac)
                for cid, df in dfs_with_gen.items()
                if cid in show_corridors
            }
        else:
            dfs_strat = dfs_filtered

        _strat_sec  = load_security_data()
        _strat_wat  = load_water_stress_data()
        _strat_ctry = _aggregate_strategic_country_df(
            dfs_strat, show_corridors, caps_for_year, _strat_sec, _strat_wat,
            cap_on=strat_cap_on,
        )
        _strat_disp = _build_strategic_dispatch(
            _strat_ctry, h2_demand_kt, w_cost, w_sec, w_dep, w_water,
            div_mode=div_mode, exempt_eu=exempt_eu,
        )
        _strat_kpis = _compute_strategic_kpis(
            _strat_disp, h2_demand_kt, float(ref_emiss_grey), div_mode=div_mode,
        )

        # ── Charts in right column: map on top, radar below ──
        _fig_strat_radar = fig_strategic_radar(_strat_kpis)
        _fig_strat_map   = fig_strategic_source_map(_strat_disp, h2_demand_kt)
        with right_col:
            st.plotly_chart(_fig_strat_map,   use_container_width=True)
            st.plotly_chart(_fig_strat_radar, use_container_width=True)

        # ── Summary list + Carbon Reference at bottom of left column ──
        dc      = _strat_kpis.get('delivered_cost', np.nan)
        sec_val = _strat_kpis.get('weighted_security', np.nan)
        hhi_val = _strat_kpis.get('hhi', np.nan)
        div_val = (1.0 - hhi_val) if pd.notna(hhi_val) else np.nan
        bws_val = _strat_kpis.get('weighted_water', np.nan)
        ca_val  = _strat_kpis.get('total_carbon_avoided', np.nan)  # kt CO₂
        de_val  = _strat_kpis.get('delivered_emissions', np.nan)   # kgCO₂/kgH₂
        n_c       = _strat_kpis.get('n_countries', 0)
        tot_alloc = _strat_disp['allocated_kt'].sum() if not _strat_disp.empty else 0
        ca_str = f'{ca_val / 1000:.2f} Mt CO₂/yr' if pd.notna(ca_val) else 'N/A'
        # Cost of carbon avoided (€/tCO₂): extra cost of green vs grey per tonne CO₂ displaced
        _cca_denom = ref_emiss_grey - (de_val if pd.notna(de_val) else 0.0)
        if pd.notna(dc) and _cca_denom > 1e-6:
            cca_val = (dc - ref_cost_grey) / _cca_denom * 1000  # €/tCO₂
            cca_str = f'{cca_val:.0f} €/tCO₂'
        else:
            cca_str = 'N/A'
        with left_col:
            st.subheader('Dispatch Summary')
            st.markdown(
                f'- **Countries:** {n_c}\n'
                f'- **Total allocated:** {tot_alloc/1000:.1f} Mt H₂/yr'
                f' (demand: {h2_demand_kt/1000:.1f} Mt H₂/yr)\n'
                f'- **Delivered cost:** {"—" if pd.isna(dc) else f"{dc:.3f} €/kg"}\n'
                f'- **Security (CPI):** {"—" if pd.isna(sec_val) else f"{sec_val:.0f} / 100"}\n'
                f'- **Diversification (1−HHI):** {"—" if pd.isna(div_val) else f"{div_val:.3f}"}\n'
                f'- **Water stress (BWS):** {"—" if pd.isna(bws_val) else f"{bws_val:.2f} / 5.0"}\n'
                f'- **Total carbon avoided:** {ca_str}\n'
                f'- **Cost of carbon avoided:** {cca_str}'
            )
            st.divider()
            st.subheader('Carbon Reference')
            st.caption('Grey H₂ baseline for carbon-avoided calculation.')
            st.number_input(
                'Grey H₂ cost (€/kg)', value=1.50, step=0.10,
                min_value=0.5, max_value=5.0, key='strat_ref_cost',
                help='Typical SMR grey hydrogen: €1.0–2.0/kg.',
            )
            st.number_input(
                'Grey H₂ emissions (kgCO₂/kgH₂)', value=9.0, step=0.5,
                min_value=1.0, max_value=15.0, key='strat_ref_emiss',
                help='SMR without CCS: ~9 kgCO₂/kgH₂.',
            )

        # ── Dispatch detail table ──
        if not _strat_disp.empty:
            with st.expander('Dispatch detail table', expanded=False):
                _display_cols = ['Country', 'ISO_A3', 'corridor_id', 'allocated_kt',
                                 'rep_cost_per_kg', 'rep_emissions_per_kg', 'cpi_score', 'bws_score']
                _display_cols = [c for c in _display_cols if c in _strat_disp.columns]
                _disp_show = _strat_disp[_display_cols].copy()
                _disp_show['allocated_Mt'] = (_disp_show['allocated_kt'] / 1000).round(3)
                _disp_show['pct_demand']   = (_disp_show['allocated_kt'] / h2_demand_kt * 100).round(2)
                st.dataframe(
                    _disp_show.rename(columns={
                        'Country':              'Country',
                        'ISO_A3':               'ISO',
                        'corridor_id':          'Corridor',
                        'allocated_kt':         'Allocated (kt/yr)',
                        'allocated_Mt':         'Allocated (Mt/yr)',
                        'pct_demand':           '% of Demand',
                        'rep_cost_per_kg':      'Cost (€/kg)',
                        'rep_emissions_per_kg': 'Emissions (kgCO₂/kg)',
                        'cpi_score':            'CPI Score',
                        'bws_score':            'BWS Score',
                    }),
                    use_container_width=True,
                    hide_index=True,
                )


    with tab_projects:
        st.header('Global Electrolytic H₂ Projects Pipeline')
        st.caption(
            'Source: IEA Hydrogen Production Projects database. '
            'Capacity shown is announced/planned kt H₂/yr for electrolysis projects only. '
            'Colour scale is logarithmic — each step is a 10× increase.'
        )

        _proj_df = load_h2_projects()

        if _proj_df.empty:
            st.error('`Data/Hydrogen Production Projects.xlsx` not found.')
        else:
            # ── Filters ──
            _all_statuses = [s for s in _H2_STATUS_ORDER if s in _proj_df['Status'].unique()]
            _default_statuses = [s for s in _all_statuses
                                 if s in ('Operational', 'FID/Construction', 'DEMO',
                                          'Feasibility study', 'Concept')]
            _sel_statuses = st.multiselect(
                'Project status filter',
                options=_all_statuses,
                default=_default_statuses,
                key='proj_status_filter',
                help='Include projects with these statuses. '
                     'Deselect "Concept" to focus on more mature projects.',
            )

            # ── Top-level metrics ──
            _sub_all = _proj_df[_proj_df['Status'].isin(_sel_statuses)].dropna(
                subset=['Capacity_kt_H2_per_y']
            ) if _sel_statuses else _proj_df.dropna(subset=['Capacity_kt_H2_per_y'])

            _ded = _sub_all[_sub_all['Technology_electricity'] == 'Dedicated renewable']
            _grid = _sub_all[_sub_all['Technology_electricity'] == 'Grid']
            _grd_ren = _sub_all[_sub_all['Technology_electricity'] == 'Grid+Renewables']

            _m1, _m2, _m3, _m4 = st.columns(4)
            with _m1:
                st.metric('Total capacity (all sources)',
                          f'{_sub_all["Capacity_kt_H2_per_y"].sum()/1000:.1f} Mt H₂/yr',
                          help='All electrolysis projects matching the status filter.')
            with _m2:
                st.metric('Dedicated renewables',
                          f'{_ded["Capacity_kt_H2_per_y"].sum()/1000:.1f} Mt H₂/yr',
                          help='Projects using only dedicated (off-grid) renewable electricity.')
            with _m3:
                st.metric('Grid electricity',
                          f'{_grid["Capacity_kt_H2_per_y"].sum()/1000:.1f} Mt H₂/yr',
                          help='Projects using grid electricity.')
            with _m4:
                st.metric('Grid + Renewables',
                          f'{_grd_ren["Capacity_kt_H2_per_y"].sum()/1000:.1f} Mt H₂/yr',
                          help='Projects combining grid and dedicated renewables.')

            st.divider()

            # ── The two choropleth maps ──
            _map_col1, _map_col2 = st.columns(2)
            with _map_col1:
                st.subheader('Dedicated Renewables')
                st.plotly_chart(
                    fig_h2_pipeline_map(_proj_df, 'Dedicated renewable',
                                        _sel_statuses, height=420),
                    use_container_width=True,
                )
            with _map_col2:
                st.subheader('Grid Electricity')
                st.plotly_chart(
                    fig_h2_pipeline_map(_proj_df, 'Grid',
                                        _sel_statuses, height=420),
                    use_container_width=True,
                )

            st.divider()

            # ── Stacked bar: top countries across all sources ──
            st.subheader('Top countries — capacity by electricity source')
            _top_n = st.slider('Top N countries', min_value=10, max_value=40,
                                value=20, step=5, key='proj_top_n')
            st.plotly_chart(
                fig_h2_pipeline_bar(_proj_df, _sel_statuses, top_n=_top_n, height=500),
                use_container_width=True,
            )

            # ── Raw data table ──
            with st.expander('Project data table', expanded=False):
                _disp_cols = ['Project name', 'Country', 'Status', 'Technology',
                               'Technology_electricity', 'Date online',
                               'Capacity_kt_H2_per_y', 'Location']
                _disp_cols = [c for c in _disp_cols if c in _proj_df.columns]
                _tbl = _proj_df[_proj_df['Status'].isin(_sel_statuses)][_disp_cols].copy() \
                    if _sel_statuses else _proj_df[_disp_cols].copy()
                _tbl = _tbl.sort_values('Capacity_kt_H2_per_y', ascending=False)
                _tbl.columns = [c.replace('Capacity_kt_H2_per_y', 'Capacity (kt H₂/yr)')
                                  .replace('Technology_electricity', 'Electricity source')
                                 for c in _tbl.columns]
                st.dataframe(_tbl, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# H₂ Projects Pipeline helpers
# ---------------------------------------------------------------------------

_H2_PROJ_COLS = [
    'Ref', 'Project name', 'Country', 'Date online', 'Decomission date', 'Status',
    'Technology', 'Technology_details', 'Technology_electricity',
    'Technology_electricity_details', 'Product',
    'EndUse_Refining', 'EndUse_Ammonia', 'EndUse_Methanol', 'EndUse_Iron&Steel',
    'EndUse_Other Ind', 'EndUse_Mobility', 'EndUse_Power', 'EndUse_Grid inj.',
    'EndUse_CHP', 'EndUse_Domestic heat', 'EndUse_Biofuels', 'EndUse_Synfuels',
    'EndUse_CH4 grid inj.', 'EndUse_CH4 mobility',
    'Announced Size', 'Capacity_MWel', 'Capacity_Nm3_H2_per_h',
    'Capacity_kt_H2_per_y', 'Capacity_t_CO2_per_y',
    'IEA_capacity_Nm3_H2_per_h', 'References', 'Location', 'Latitude', 'Longitude',
]

_H2_ELEC_TECHS = {'PEM', 'ALK', 'SOEC', 'Other Electrolysis', 'AEM', 'Electrolysis'}

_H2_STATUS_ORDER = [
    'Operational', 'FID/Construction', 'DEMO', 'Feasibility study', 'Concept',
    'Various', 'Decommisioned',
]

_H2_ELEC_SOURCE_COLOURS = {
    'Dedicated renewable': '#2A9D8F',
    'Grid':                '#E63946',
    'Grid+Renewables':     '#F4A261',
    'Nuclear':             '#9B2335',
    'Other/unknown':       '#aaaaaa',
}


@st.cache_data(show_spinner='Loading H₂ projects data…')
def load_h2_projects() -> pd.DataFrame:
    """Load IEA Hydrogen Production Projects database from the Excel file."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data', 'Hydrogen Production Projects.xlsx')
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_excel(path, sheet_name=0, skiprows=2)
    df.columns = _H2_PROJ_COLS
    # Normalise electricity source capitalisation
    df['Technology_electricity'] = df['Technology_electricity'].str.strip()
    df['Technology_electricity'] = df['Technology_electricity'].replace(
        {'Other/Unknown': 'Other/unknown'}
    )
    # Keep only electrolysis projects with a country
    df = df[df['Technology'].isin(_H2_ELEC_TECHS) & df['Country'].notna()].copy()
    df['Capacity_kt_H2_per_y'] = pd.to_numeric(df['Capacity_kt_H2_per_y'], errors='coerce')
    return df.reset_index(drop=True)


def fig_h2_pipeline_map(
    proj_df: pd.DataFrame,
    elec_source: str,
    statuses: list[str],
    height: int = 420,
) -> go.Figure:
    """
    Choropleth of total electrolytic H₂ production capacity (kt H₂/yr)
    per country, filtered by electricity source and project status.
    """
    if proj_df.empty:
        return go.Figure().update_layout(title='No project data found', height=height)

    # Filter
    mask = proj_df['Technology_electricity'] == elec_source
    if statuses:
        mask &= proj_df['Status'].isin(statuses)
    sub = proj_df[mask].dropna(subset=['Capacity_kt_H2_per_y', 'Country'])

    if sub.empty:
        return go.Figure().update_layout(
            title=f'No data — {elec_source} ({", ".join(statuses)})', height=height
        )

    by_country = (
        sub.groupby('Country')['Capacity_kt_H2_per_y']
        .sum()
        .reset_index()
        .rename(columns={'Country': 'ISO_A3', 'Capacity_kt_H2_per_y': 'capacity_kt'})
    )
    # Count projects per country for hover
    proj_count = (
        sub.groupby('Country').size()
        .reset_index(name='n_projects')
        .rename(columns={'Country': 'ISO_A3'})
    )
    by_country = by_country.merge(proj_count, on='ISO_A3', how='left')
    by_country['capacity_Mt'] = (by_country['capacity_kt'] / 1000).round(3)

    # Log scale for colour (many orders of magnitude across countries)
    by_country['log_cap'] = np.log10(by_country['capacity_kt'].clip(lower=0.01))
    log_max = float(np.ceil(by_country['log_cap'].max()))

    # Build readable tick labels on log axis
    tick_vals, tick_text = [], []
    for exp in range(0, int(log_max) + 1):
        tick_vals.append(exp)
        tick_text.append(f'{10**exp:,.0f}' if exp < 4 else f'{10**exp/1000:,.0f}k')

    colour = _H2_ELEC_SOURCE_COLOURS.get(elec_source, '#457B9D')
    # Build a white→colour scale
    colour_scale = [[0, '#f0f0f0'], [1, colour]]

    status_label = ', '.join(statuses) if statuses else 'all statuses'
    title = (
        f'<b>{elec_source}</b> — electrolytic H₂ capacity by country<br>'
        f'<sup>{status_label} | kt H₂/yr (log scale)</sup>'
    )

    fig = px.choropleth(
        by_country,
        locations='ISO_A3',
        color='log_cap',
        color_continuous_scale=colour_scale,
        range_color=[0, log_max],
        hover_name='ISO_A3',
        hover_data={
            'capacity_kt': ':,.1f',
            'capacity_Mt': ':.3f',
            'n_projects':  True,
            'log_cap':     False,
        },
        labels={
            'capacity_kt': 'kt H₂/yr',
            'capacity_Mt': 'Mt H₂/yr',
            'n_projects':  '# projects',
        },
        title=title,
    )
    fig.update_geos(**CHOROPLETH_GEO)
    fig.update_layout(
        paper_bgcolor='white',
        margin=dict(l=0, r=0, t=60, b=0),
        height=height,
        coloraxis_colorbar=dict(
            title='kt H₂/yr',
            thickness=15,
            len=0.6,
            tickvals=tick_vals,
            ticktext=tick_text,
        ),
    )
    return fig


def fig_h2_pipeline_bar(
    proj_df: pd.DataFrame,
    statuses: list[str],
    top_n: int = 20,
    height: int = 420,
) -> go.Figure:
    """
    Stacked horizontal bar of top N countries by total electrolytic H₂ capacity,
    split by electricity source.
    """
    if proj_df.empty:
        return go.Figure()

    sub = proj_df.copy()
    if statuses:
        sub = sub[sub['Status'].isin(statuses)]
    sub = sub.dropna(subset=['Capacity_kt_H2_per_y', 'Country'])

    # Aggregate per country × electricity source
    grp = (
        sub.groupby(['Country', 'Technology_electricity'])['Capacity_kt_H2_per_y']
        .sum()
        .reset_index()
    )
    total_by_country = grp.groupby('Country')['Capacity_kt_H2_per_y'].sum()
    top_countries = total_by_country.nlargest(top_n).index.tolist()
    grp = grp[grp['Country'].isin(top_countries)]

    # Pivot for stacked bar
    pivot = grp.pivot(index='Country', columns='Technology_electricity',
                      values='Capacity_kt_H2_per_y').fillna(0)
    pivot = pivot.loc[top_countries[::-1]]  # reverse so largest at top

    fig = go.Figure()
    for src in pivot.columns:
        colour = _H2_ELEC_SOURCE_COLOURS.get(src, '#aaaaaa')
        fig.add_trace(go.Bar(
            name=src,
            x=pivot[src],
            y=pivot.index,
            orientation='h',
            marker_color=colour,
            hovertemplate='%{y}: %{x:,.1f} kt H₂/yr<extra>' + src + '</extra>',
        ))

    fig.update_layout(
        barmode='stack',
        title=f'Top {top_n} countries — electrolytic H₂ capacity by electricity source',
        xaxis_title='Capacity (kt H₂/yr)',
        legend=dict(orientation='h', yanchor='bottom', y=1.01),
        plot_bgcolor='white', paper_bgcolor='white',
        height=height,
        margin=dict(l=80, r=20, t=80, b=40),
    )
    return fig


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    main()
