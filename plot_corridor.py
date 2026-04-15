"""
plot_corridor.py
----------------
Visualisation functions for the 5-corridor EHB comparison.

Functions
---------
  plot_corridor_supply_curves(results)
      Merit-order supply curves (within-cap) — one stepped line per corridor.
      X: cumulative supply [kt H2/yr], Y: total delivered cost [EUR/kg H2].
      Dashed horizontal line at EHB 10 Mt/yr import target.
      This is the primary thesis chart.

  plot_corridor_cost_breakdown(results)
      Grouped stacked bar chart: generation cost + transport cost (medians of
      within-cap points) per corridor, with corridor name annotations.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from corridors import CORRIDORS

# EHB import target
EHB_TARGET_KT = 10_000.0


# ---------------------------------------------------------------------------
# Helper: build sorted supply table from results dict
# ---------------------------------------------------------------------------

def _supply_curve_data(results: dict, h2_demand: float = 15_000,
                       n_global: int = None) -> dict:
    """
    For each corridor, return sorted arrays of (cumulative_kt, cost_per_kg)
    using one representative value per country.

    Country-level approach (preferred):
      Uses df_country from results (built by aggregate_country_supply()).
      Each country = one step.  X-width = country cap [kt H2/yr].
      Cost = trimmed mean of grid-point costs (P10–P90, drops cheapest and
             most expensive 10% to remove outliers).
      Countries with no finite cap are excluded from the chart.

    Grid-point fallback (legacy):
      Used only when df_country is not present in results (e.g. old CSVs).
      Each grid point = h2_demand / n_global kt.
    """
    # Resolve n_global for the legacy fallback
    if n_global is None:
        for res in results.values():
            if isinstance(res, dict) and res.get('n_global'):
                n_global = res['n_global']
                break

    curves = {}
    for cid, res in results.items():
        cfg        = res['corridor']
        df_country = res.get('df_country')

        if df_country is not None:
            # ── Country-level supply curve ────────────────────────────────
            capped = (df_country[df_country['has_finite_cap']]
                      .sort_values('rep_cost_per_kg')
                      .reset_index(drop=True))
            if capped.empty:
                continue

            capped['cum_supply_kt'] = capped['country_cap_kt'].cumsum()

            curves[cid] = {
                'cum_supply':        capped['cum_supply_kt'].values,
                'cost':              capped['rep_cost_per_kg'].values,
                'labels':            capped['ISO_A3'].values,
                'colour':            cfg['colour'],
                'label':             f"Corridor {cid}: {cfg['subtitle']}",
                'uncapped_countries': (df_country[~df_country['has_finite_cap']]
                                       ['ISO_A3'].tolist()),
            }

        else:
            # ── Legacy grid-point fallback ────────────────────────────────
            df = res['df']
            within = df[df['within_cap'] == True].copy()
            if within.empty:
                continue

            within = within.sort_values('Total Cost per kg H2').reset_index(drop=True)
            n_rows = n_global if n_global is not None else len(df)
            demand_per_pt = h2_demand / n_rows
            within['cum_supply_kt'] = (within.index + 1) * demand_per_pt

            curves[cid] = {
                'cum_supply':        within['cum_supply_kt'].values,
                'cost':              within['Total Cost per kg H2'].values,
                'labels':            None,
                'colour':            cfg['colour'],
                'label':             f"Corridor {cid}: {cfg['subtitle']}",
                'uncapped_countries': [],
            }

    return curves


# ---------------------------------------------------------------------------
# Helper: greedy merit-order dispatch across all corridors
# ---------------------------------------------------------------------------

def _build_optimal_mix(results: dict, demand_kt: float) -> pd.DataFrame:
    """
    Greedy merit-order dispatch: take the cheapest available country across
    all corridors combined, respecting each country's capacity cap.

    Where a country appears in multiple corridors, keep only the corridor
    that delivers it at the lowest cost.

    Returns a DataFrame (sorted cheapest-first, allocated up to demand_kt):
        corridor, colour, label, country, cost, cap_kt,
        allocated_kt, cum_kt_start, cum_kt
    """
    rows = []
    for cid, res in results.items():
        cfg        = res['corridor']
        df_country = res.get('df_country')
        if df_country is None:
            continue
        for _, row in df_country[df_country['has_finite_cap']].iterrows():
            rows.append({
                'corridor': cid,
                'colour':   cfg['colour'],
                'label':    row['ISO_A3'],
                'country':  row.get('Country', row['ISO_A3']),
                'cost':     row['rep_cost_per_kg'],
                'cap_kt':   row['country_cap_kt'],
            })

    if not rows:
        return pd.DataFrame()

    pool = (pd.DataFrame(rows)
              .sort_values('cost')
              .drop_duplicates(subset=['label'], keep='first')
              .sort_values('cost')
              .reset_index(drop=True))

    remaining = demand_kt
    allocated = []
    for _, row in pool.iterrows():
        if remaining <= 0:
            break
        volume = min(remaining, float(row['cap_kt']))
        allocated.append({**row.to_dict(), 'allocated_kt': volume})
        remaining -= volume

    if not allocated:
        return pd.DataFrame()

    alloc = pd.DataFrame(allocated)
    alloc['cum_kt']       = alloc['allocated_kt'].cumsum()
    alloc['cum_kt_start'] = alloc['cum_kt'] - alloc['allocated_kt']
    return alloc


# ---------------------------------------------------------------------------
# Chart 1: Merit-order supply curves
# ---------------------------------------------------------------------------

def plot_corridor_supply_curves(results: dict, h2_demand: float = 15_000,
                                n_global: int = None, year: int = 2040):
    """
    Stepped merit-order supply curves for all corridors, plus an optimal-mix
    curve showing the cheapest way to meet h2_demand across all corridors.

    Individual corridor curves (thin lines):
      Each step = one country, ordered cheapest-first within that corridor.
      X-width of each step = country capacity cap.

    Optimal mix curve (thick segments, same corridor colours):
      Greedy dispatch across all corridors — cheapest country globally first,
      respecting capacity caps.  Stops at h2_demand.  Each segment is coloured
      by the corridor that supplies it.

    Vertical dashed line marks h2_demand.
    """
    curves = _supply_curve_data(results, h2_demand, n_global=n_global)
    if not curves:
        print('No corridor results to plot.')
        return

    fig = go.Figure()

    # ── Individual corridor supply curves (thin) ─────────────────────────────
    for cid, c in curves.items():
        xs = np.concatenate([[0], c['cum_supply']])
        ys = np.concatenate([[c['cost'][0]], c['cost']])

        if c['labels'] is not None:
            padded_labels = np.concatenate([[c['labels'][0]], c['labels']])
            hovertemplate = (
                '<b>%{fullData.name}</b><br>'
                'Country: %{customdata}<br>'
                'Cumulative supply: %{x:,.0f} kt H\u2082/yr<br>'
                'Cost: %{y:.3f} EUR/kg H\u2082<extra></extra>'
            )
        else:
            padded_labels = None
            hovertemplate = (
                '<b>%{fullData.name}</b><br>'
                'Cumulative supply: %{x:,.0f} kt H\u2082/yr<br>'
                'Cost: %{y:.3f} EUR/kg H\u2082<extra></extra>'
            )

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode='lines',
            line=dict(color=c['colour'], width=2, shape='hv'),
            opacity=0.55,
            name=c['label'],
            legendgroup=f'corridor_{cid}',
            customdata=padded_labels,
            hovertemplate=hovertemplate,
        ))

    # ── Optimal mix curve (thick segments, coloured by corridor) ─────────────
    alloc = _build_optimal_mix(results, h2_demand)
    if not alloc.empty:
        # One trace per corridor so each gets its colour and legend entry
        seen_cids: set = set()
        for cid in alloc['corridor'].unique():
            sub = alloc[alloc['corridor'] == cid].reset_index(drop=True)
            colour = sub['colour'].iloc[0]

            # Build NaN-separated horizontal segments (one per country step)
            xs, ys, hover = [], [], []
            for _, row in sub.iterrows():
                xs  += [row['cum_kt_start'], row['cum_kt'], None]
                ys  += [row['cost'], row['cost'], None]
                hover += [
                    (f'<b>Optimal mix — {row["country"]} [{cid}]</b><br>'
                     f'Cost: {row["cost"]:.3f} EUR/kg H\u2082<br>'
                     f'Allocated: {row["allocated_kt"]:,.0f} kt/yr<extra></extra>'),
                ] * 3

            first = cid not in seen_cids
            seen_cids.add(cid)

            cfg_name = results[cid]['corridor']['subtitle']
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode='lines',
                line=dict(color=colour, width=5),
                name=f'Optimal mix — {cid}: {cfg_name}',
                legendgroup=f'optimal_{cid}',
                showlegend=first,
                customdata=hover,
                hovertemplate='%{customdata}',
            ))

        # Vertical dashed line at demand target
        fig.add_vline(
            x=h2_demand,
            line_dash='dot',
            line_color='#444444',
            line_width=1.5,
            annotation_text=f'Demand<br>{h2_demand/1000:.1f} Mt/yr',
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
        xaxis=dict(
            title='Cumulative supply capacity [kt H\u2082/yr]',
            gridcolor='#e0e0e0',
        ),
        yaxis=dict(
            title='Total delivered cost [EUR/kg H\u2082]',
            gridcolor='#e0e0e0',
        ),
        legend=dict(
            orientation='v',
            x=1.01, y=1,
            xanchor='left',
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='#cccccc',
            borderwidth=1,
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        margin=dict(l=70, r=300, t=100, b=70),
        height=580,
        width=1100,
    )

    fig.show()


# ---------------------------------------------------------------------------
# Chart 2: Cost breakdown comparison (stacked bar, gen + transport)
# ---------------------------------------------------------------------------

def plot_corridor_cost_breakdown(results: dict, year: int = 2040):
    """
    Stacked bar chart showing median generation cost + median transport cost
    for within-cap grid points, one bar group per corridor.

    Also shows:
      - Error bars (P25–P75 range of total cost)
      - Annotation with best-source country
    """
    rows = []
    for cid, res in results.items():
        cfg    = res['corridor']
        df     = res['df']
        within = df[df['within_cap'] == True].dropna(subset=['Total Cost per kg H2'])
        if within.empty:
            continue

        med_gen   = within['Gen. cost per kg H2'].median()
        med_trans = within['Transport Cost per kg H2'].median()
        p25_total = within['Total Cost per kg H2'].quantile(0.25)
        p75_total = within['Total Cost per kg H2'].quantile(0.75)
        min_total = within['Total Cost per kg H2'].min()
        best_src  = (within.loc[within['Total Cost per kg H2'].idxmin(), 'ISO_A3']
                     if 'ISO_A3' in within.columns else '---')

        rows.append({
            'cid':       cid,
            'label':     f'Corridor {cid}',
            'subtitle':  cfg['subtitle'],
            'med_gen':   med_gen,
            'med_trans': med_trans,
            'med_total': med_gen + med_trans,
            'p25':       p25_total,
            'p75':       p75_total,
            'min':       min_total,
            'best_src':  best_src,
            'colour':    cfg['colour'],
        })

    if not rows:
        print('No corridor results to plot.')
        return

    comp = pd.DataFrame(rows).sort_values('med_total')

    labels    = comp['label'].tolist()
    med_gen   = comp['med_gen'].tolist()
    med_trans = comp['med_trans'].tolist()
    med_total = comp['med_total'].tolist()
    colours   = comp['colour'].tolist()

    fig = go.Figure()

    # Generation cost bars (bottom)
    fig.add_trace(go.Bar(
        name='Generation cost',
        x=labels,
        y=med_gen,
        marker_color='#66bb6a',
        text=[f'{v:.2f}' for v in med_gen],
        textposition='inside',
        insidetextanchor='middle',
        hovertemplate='<b>%{x}</b><br>Gen cost: %{y:.3f} EUR/kg<extra></extra>',
    ))

    # Transport cost bars (top)
    fig.add_trace(go.Bar(
        name='Transport cost',
        x=labels,
        y=med_trans,
        marker_color='#90a4ae',
        text=[f'{v:.2f}' for v in med_trans],
        textposition='inside',
        insidetextanchor='middle',
        hovertemplate='<b>%{x}</b><br>Transport cost: %{y:.3f} EUR/kg<extra></extra>',
    ))

    # P25–P75 error bar overlay (invisible bar, error bars visible)
    fig.add_trace(go.Scatter(
        x=labels,
        y=med_total,
        mode='markers',
        marker=dict(color='black', size=6, symbol='diamond'),
        error_y=dict(
            type='data',
            symmetric=False,
            array=[row['p75'] - row['med_total'] for _, row in comp.iterrows()],
            arrayminus=[row['med_total'] - row['p25'] for _, row in comp.iterrows()],
            color='black',
            thickness=1.5,
            width=6,
        ),
        name='P25–P75 range',
        hovertemplate=(
            '<b>%{x}</b><br>'
            'Median total: %{y:.3f} EUR/kg<br>'
            'P25–P75 range shown<extra></extra>'
        ),
    ))

    # Best-source annotations above each bar
    for _, row in comp.iterrows():
        fig.add_annotation(
            x=row['label'],
            y=row['p75'] + 0.05,
            text=f"min: {row['min']:.2f}<br>({row['best_src']})",
            showarrow=False,
            font=dict(size=9, color='#333333'),
            align='center',
        )

    # Subtitle annotations below x-axis labels
    for _, row in comp.iterrows():
        fig.add_annotation(
            x=row['label'],
            y=-0.14,
            xref='x', yref='paper',
            text=f"<i>{row['subtitle']}</i>",
            showarrow=False,
            font=dict(size=8, color='#666666'),
            align='center',
        )

    fig.update_layout(
        barmode='stack',
        title=dict(
            text=f'EHB Corridor Cost Breakdown — Median of Within-Cap Supply Points ({year})',
            font=dict(size=16),
        ),
        xaxis=dict(
            title='',
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title='Total delivered cost [EUR/kg H\u2082]',
            gridcolor='#e0e0e0',
        ),
        legend=dict(
            orientation='h',
            x=0.5, y=1.02,
            xanchor='center',
            bgcolor='rgba(255,255,255,0.85)',
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=70, r=50, t=100, b=120),
        height=520,
        width=820,
    )

    fig.show()


# ---------------------------------------------------------------------------
# Chart 3: Flow map — geodesic lines from cheapest sources to destinations
# ---------------------------------------------------------------------------

# Transport medium colour palette (colourblind-safe Wong palette)
MEDIUM_COLOURS = {
    'NH3':    '#0072B2',   # blue
    'LOHC':   '#E69F00',   # orange
    'H2 Gas': '#009E73',   # green
    'H2 Liq': '#CC79A7',   # pink
}
MEDIUM_LABELS = {
    'NH3':    'NH₃ (ship / truck)',
    'LOHC':   'LOHC (ship)',
    'H2 Gas': 'H₂ gas (pipeline)',
    'H2 Liq': 'LH₂ (ship)',
}

# Large countries split into East/West regions to show intra-country variation
REGIONAL_SPLITS = {
    'USA': [(-130, -100, 'USA-W'), (-100, -60, 'USA-E')],
    'CAN': [(-140, -90,  'CAN-W'), (-90,  -50, 'CAN-E')],
}


def _region_id(iso: str, lon: float) -> str:
    """Return a sub-region label for USA/Canada, ISO_A3 for all others."""
    if iso in REGIONAL_SPLITS:
        for lon_min, lon_max, label in REGIONAL_SPLITS[iso]:
            if lon_min <= lon < lon_max:
                return label
    return iso


def _aggregate_to_regions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse grid points to one row per country (or sub-region for USA/Canada).
    Uses mean costs and centroid lat/lon; modal transport medium; max capacity.
    """
    df = df.copy()
    df['_region'] = [_region_id(iso, lon)
                     for iso, lon in zip(df['ISO_A3'], df['Longitude'])]

    # Drop rows with NaN costs so they don't pollute group means
    df = df.dropna(subset=['Total Cost per kg H2', 'Transport Cost per kg H2'])

    records = []
    for region, g in df.groupby('_region'):
        cap = g['country_cap_kt'].dropna().max() if 'country_cap_kt' in g.columns else float('nan')
        records.append({
            'ISO_A3':                   g['ISO_A3'].iloc[0],
            'Country':                  g['Country'].iloc[0],
            '_region':                  region,
            'Latitude':                 g['Latitude'].mean(),
            'Longitude':                g['Longitude'].mean(),
            'Gen. cost per kg H2':      g['Gen. cost per kg H2'].mean(),
            'Transport Cost per kg H2': g['Transport Cost per kg H2'].mean(),
            'Total Cost per kg H2':     g['Total Cost per kg H2'].mean(),
            'Cheapest Medium':          g['Cheapest Medium'].value_counts().idxmax(),
            'country_cap_kt':           cap,
        })
    return pd.DataFrame(records).dropna(subset=['Total Cost per kg H2'])


def _build_flow_map_fig(results: dict, n_countries: int = 15,
                        h2_demand: float = None, year: int = 2040) -> go.Figure:
    """
    Build and return the flow map Plotly figure without showing it.
    Called by plot_corridor_flow_map (show in browser) and Streamlit dashboards.
    """
    # Resolve n_global from results dict
    n_global = None
    for res in results.values():
        if isinstance(res, dict) and res.get('n_global'):
            n_global = res['n_global']
            break

    fig = go.Figure()

    def _fmt(v):
        return f'{v:.2f}' if pd.notna(v) else 'N/A'

    def _fmt_cap(v):
        if pd.isna(v) or v == 0:
            return 'N/A'
        return f'{v:,.0f} kt/yr'

    def _fmt_vol(v):
        if pd.isna(v) or v <= 0:
            return 'N/A'
        return f'{v:,.0f} kt/yr'

    # demand_per_point: consistent with apply_capacity_limits()
    demand_per_pt = None
    if h2_demand is not None and n_global is not None:
        demand_per_pt = h2_demand / n_global

    # ── 1. Flow lines (one trace per corridor, NaN-separated polylines) ──────
    for cid, res in results.items():
        cfg    = res['corridor']
        df     = res['df']
        colour = cfg['colour']
        if cfg.get('destination') is None:
            continue   # domestic corridor — no flow lines
        dest_lat, dest_lon = cfg['destination']

        # Within-cap rows only — show merit-order dispatched sources
        within = df[df['within_cap'] == True].copy() if 'within_cap' in df.columns else df.copy()
        valid = within.dropna(subset=['Total Cost per kg H2', 'Transport Cost per kg H2'])
        if valid.empty:
            continue

        agg = _aggregate_to_regions(valid)
        top = agg.nsmallest(n_countries, 'Total Cost per kg H2')

        lats, lons = [], []
        for _, row in top.iterrows():
            lats += [row['Latitude'], dest_lat, None]
            lons += [row['Longitude'], dest_lon, None]

        fig.add_trace(go.Scattergeo(
            lat=lats,
            lon=lons,
            mode='lines',
            line=dict(color=colour, width=1.8),
            opacity=0.55,
            name=f"Corridor {cid} — {cfg['subtitle']}",
            legendgroup=f'corridor_{cid}',
            showlegend=True,
            hoverinfo='skip',
        ))

    # ── 2. Source scatter (coloured by transport medium) ─────────────────────
    shown_media: set = set()
    for cid, res in results.items():
        cfg    = res['corridor']
        df     = res['df']
        if cfg.get('destination') is None:
            continue   # domestic corridor — no flow map entry
        dest_lat, dest_lon = cfg['destination']

        within = df[df['within_cap'] == True].copy() if 'within_cap' in df.columns else df.copy()
        valid = within.dropna(subset=['Total Cost per kg H2', 'Transport Cost per kg H2'])
        if valid.empty:
            continue

        # Aggregate to countries; count within-cap rows per region for volume
        valid = valid.copy()
        valid['_region'] = [_region_id(iso, lon)
                            for iso, lon in zip(valid['ISO_A3'], valid['Longitude'])]

        records = []
        for region, g in valid.groupby('_region'):
            n_pts = len(g)
            alloc = n_pts * demand_per_pt if demand_per_pt is not None else float('nan')
            cap   = g['country_cap_kt'].dropna().max() if 'country_cap_kt' in g.columns else float('nan')
            records.append({
                'ISO_A3':                   g['ISO_A3'].iloc[0],
                'Country':                  g['Country'].iloc[0],
                '_region':                  region,
                'Latitude':                 g['Latitude'].mean(),
                'Longitude':                g['Longitude'].mean(),
                'Gen. cost per kg H2':      g['Gen. cost per kg H2'].mean(),
                'Transport Cost per kg H2': g['Transport Cost per kg H2'].mean(),
                'Total Cost per kg H2':     g['Total Cost per kg H2'].mean(),
                'Cheapest Medium':          g['Cheapest Medium'].value_counts().idxmax(),
                'country_cap_kt':           cap,
                'allocated_kt':             alloc,
                'n_points':                 n_pts,
            })
        agg = pd.DataFrame(records).dropna(subset=['Total Cost per kg H2'])
        if agg.empty:
            continue

        top = agg.nsmallest(n_countries, 'Total Cost per kg H2').copy().reset_index(drop=True)

        # Size dots proportionally to allocated_kt (7–18 px range)
        max_alloc = top['allocated_kt'].max()
        if pd.notna(max_alloc) and max_alloc > 0:
            top['_size'] = 7 + 11 * (top['allocated_kt'] / max_alloc).clip(0, 1)
        else:
            # Fallback: size by cheapness rank
            top['_rank'] = range(1, len(top) + 1)
            top['_size'] = 13 - (top['_rank'] - 1) * (7 / max(len(top) - 1, 1))
        top['_size'] = top['_size'].clip(lower=6)

        for medium, grp in top.groupby('Cheapest Medium'):
            med_colour  = MEDIUM_COLOURS.get(medium, '#888888')
            show_legend = medium not in shown_media
            shown_media.add(medium)

            hover = [
                f'<b>Corridor {cid} — {row["_region"]}</b><br>'
                f'Allocated supply: {_fmt_vol(row["allocated_kt"])}<br>'
                f'({row["n_points"]} grid points × {demand_per_pt:.1f} kt/pt)<br>'
                f'Avg total cost: {_fmt(row["Total Cost per kg H2"])} EUR/kg<br>'
                f'Avg gen. cost: {_fmt(row["Gen. cost per kg H2"])} EUR/kg<br>'
                f'Avg transport: {_fmt(row["Transport Cost per kg H2"])} EUR/kg<br>'
                f'Modal medium: {medium}<br>'
                f'Country cap: {_fmt_cap(row["country_cap_kt"])}'
                if demand_per_pt is not None else
                f'<b>Corridor {cid} — {row["_region"]}</b><br>'
                f'Avg total cost: {_fmt(row["Total Cost per kg H2"])} EUR/kg<br>'
                f'Avg gen. cost: {_fmt(row["Gen. cost per kg H2"])} EUR/kg<br>'
                f'Avg transport: {_fmt(row["Transport Cost per kg H2"])} EUR/kg<br>'
                f'Modal medium: {medium}<br>'
                f'Country cap: {_fmt_cap(row["country_cap_kt"])}'
                for _, row in grp.iterrows()
            ]

            fig.add_trace(go.Scattergeo(
                lat=grp['Latitude'],
                lon=grp['Longitude'],
                mode='markers',
                marker=dict(
                    size=grp['_size'].tolist(),
                    color=med_colour,
                    opacity=0.92,
                    line=dict(width=1.0, color='white'),
                ),
                name=MEDIUM_LABELS.get(medium, medium),
                legendgroup=f'medium_{medium}',
                showlegend=show_legend,
                hovertemplate='%{customdata}<extra></extra>',
                customdata=hover,
            ))

    # ── 3. Destination stars ─────────────────────────────────────────────────
    for cid, res in results.items():
        cfg = res['corridor']
        if cfg.get('destination') is None:
            continue   # domestic corridor — no destination star
        dest_lat, dest_lon = cfg['destination']

        fig.add_trace(go.Scattergeo(
            lat=[dest_lat],
            lon=[dest_lon],
            mode='markers+text',
            marker=dict(
                size=16,
                color=cfg['colour'],
                symbol='star',
                line=dict(width=1.5, color='white'),
            ),
            text=[f' {cid}'],
            textfont=dict(size=11, color=cfg['colour']),
            textposition='middle right',
            name=f'Destination {cid}',
            legendgroup=f'corridor_{cid}',
            showlegend=False,
            hovertemplate=(
                f'<b>Corridor {cid} — {cfg["subtitle"]}</b><br>'
                f'Entry point: ({dest_lat:.2f}°N, {dest_lon:.2f}°E)'
                '<extra></extra>'
            ),
        ))

    # ── Layout ───────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=(
                f'EHB Import Corridors — Within-Cap Supply Sources (top {n_countries} per corridor) ({year})<br>'
                '<sup>Lines = corridors  |  Dot colour = modal transport medium  |  '
                'Star = EU entry point  |  Dot size ∝ allocated volume  |  Hover for quantity &amp; cost</sup>'
            ),
            font=dict(size=15),
            x=0.5, xanchor='center',
        ),
        geo=dict(
            showland=True,
            landcolor='#f5f2ee',
            showocean=True,
            oceancolor='#d4e8f5',
            showcoastlines=True,
            coastlinecolor='#aaaaaa',
            showcountries=True,
            countrycolor='#dddddd',
            showframe=False,
            projection_type='natural earth',
        ),
        legend=dict(
            orientation='v',
            x=1.01, y=1,
            xanchor='left',
            bgcolor='rgba(255,255,255,0.92)',
            bordercolor='#cccccc',
            borderwidth=1,
            font=dict(size=11),
            tracegroupgap=8,
        ),
        paper_bgcolor='white',
        margin=dict(l=0, r=250, t=90, b=0),
        height=580,
        width=1150,
    )

    return fig


def plot_corridor_flow_map(results: dict, n_countries: int = 15,
                           h2_demand: float = None, year: int = 2040):
    """
    World map with geodesic flow lines from within-cap supply countries
    to each corridor's EU entry point.  Opens in browser.
    """
    _build_flow_map_fig(results, n_countries, h2_demand, year).show()


# ---------------------------------------------------------------------------
# Chart 4: Optimal supply mix — greedy merit-order dispatch across all corridors
# ---------------------------------------------------------------------------

def plot_optimal_supply_mix(results: dict, demand_kt: float = 15_000):
    """
    Greedy merit-order dispatch: fill total demand from the cheapest available
    source across all 5 corridors combined, respecting each country's capacity cap.

    Algorithm
    ---------
    1. Aggregate every corridor's valid-cost rows to country/region level.
    2. Where a country appears in multiple corridors, keep only the corridor
       that gives the cheapest delivered cost.
    3. Sort cheapest-first; allocate min(remaining_demand, cap_kt) until met.
    4. Plot the resulting supply mix on a world map + a merit-order supply curve.

    Parameters
    ----------
    results   : dict  {cid: {'corridor': cfg, 'df': DataFrame}}
    demand_kt : float  Total demand to meet [kt H₂/yr]  (default 15 000 kt = 15 Mt)
    """

    # ── 1. Collect & aggregate all corridors ─────────────────────────────────
    all_rows = []
    for cid, res in results.items():
        cfg   = res['corridor']
        df    = res['df']
        valid = df.dropna(subset=['Total Cost per kg H2', 'Transport Cost per kg H2']).copy()
        if valid.empty:
            continue
        agg = _aggregate_to_regions(valid)
        dest = cfg.get('destination')
        agg['_corridor']  = cid
        agg['_corr_col']  = cfg['colour']
        agg['_dest_lat']  = dest[0] if dest else None
        agg['_dest_lon']  = dest[1] if dest else None
        agg['_subtitle']  = cfg['subtitle']
        all_rows.append(agg)

    if not all_rows:
        print('No corridor results to dispatch.')
        return

    combined = pd.concat(all_rows, ignore_index=True)

    # ── 2. One entry per region: keep cheapest corridor ───────────────────────
    best = (combined
            .sort_values('Total Cost per kg H2')
            .drop_duplicates(subset=['_region'], keep='first')
            .sort_values('Total Cost per kg H2')
            .reset_index(drop=True))

    # ── 3. Greedy dispatch ────────────────────────────────────────────────────
    remaining  = demand_kt
    allocated  = []
    for _, row in best.iterrows():
        cap = row['country_cap_kt']
        if pd.isna(cap) or cap <= 0:
            continue
        volume = min(remaining, cap)
        allocated.append({**row.to_dict(), 'allocated_kt': volume})
        remaining -= volume
        if remaining <= 0:
            break

    if not allocated:
        print('No sources with valid capacity caps — regenerate corridor CSVs first.')
        return

    alloc = pd.DataFrame(allocated).reset_index(drop=True)
    supplied    = alloc['allocated_kt'].sum()
    unmet       = max(0.0, demand_kt - supplied)
    wavg_cost   = (alloc['Total Cost per kg H2'] * alloc['allocated_kt']).sum() / supplied

    # Print summary
    print(f'\n  ── Optimal supply mix  (demand = {demand_kt:,.0f} kt H₂/yr) ──')
    print(f'  Demand met:  {supplied:,.0f} / {demand_kt:,.0f} kt  '
          f'({100*supplied/demand_kt:.1f}%)')
    if unmet > 0:
        print(f'  Unmet demand: {unmet:,.0f} kt  (increase demand or add more data)')
    print(f'  Weighted avg delivered cost: {wavg_cost:.3f} EUR/kg H₂')
    print(f'  {len(alloc)} source regions used\n')

    by_corr = alloc.groupby('_corridor')['allocated_kt'].sum().sort_values(ascending=False)
    for cid, vol in by_corr.items():
        print(f'    Corridor {cid}: {vol:>8,.0f} kt  ({100*vol/supplied:.1f}%)')

    # ── 4. Map ────────────────────────────────────────────────────────────────
    def _fmt(v):
        return f'{v:.2f}' if pd.notna(v) else 'N/A'

    def _fmt_cap(v):
        return f'{v:,.0f} kt/yr' if pd.notna(v) and v > 0 else 'N/A'

    # Dot size: scale by allocated_kt, range 7–22 px
    max_vol = alloc['allocated_kt'].max()
    alloc['_dot_size'] = 7 + 15 * (alloc['allocated_kt'] / max_vol)

    fig = go.Figure()

    # Flow lines (coloured by corridor) — skip domestic (no destination)
    for cid in alloc['_corridor'].unique():
        sub  = alloc[alloc['_corridor'] == cid]
        col  = sub['_corr_col'].iloc[0]
        dlat = sub['_dest_lat'].iloc[0]
        dlon = sub['_dest_lon'].iloc[0]
        sub_label = sub['_subtitle'].iloc[0]

        if pd.isna(dlat) or pd.isna(dlon):
            continue   # domestic corridor — no flow lines

        lats, lons = [], []
        for _, row in sub.iterrows():
            lats += [row['Latitude'], dlat, None]
            lons += [row['Longitude'], dlon, None]

        fig.add_trace(go.Scattergeo(
            lat=lats, lon=lons,
            mode='lines',
            line=dict(color=col, width=1.8),
            opacity=0.50,
            name=f'Corridor {cid} — {sub_label}',
            legendgroup=f'corridor_{cid}',
            showlegend=True,
            hoverinfo='skip',
        ))

    # Source dots (coloured by transport medium)
    shown_media: set = set()
    for medium, grp in alloc.groupby('Cheapest Medium'):
        med_colour  = MEDIUM_COLOURS.get(medium, '#888888')
        show_legend = medium not in shown_media
        shown_media.add(medium)

        hover = [
            f'<b>{row["_region"]}</b>  [Corridor {row["_corridor"]}]<br>'
            f'Allocated: {row["allocated_kt"]:,.0f} kt/yr<br>'
            f'Total cost: {_fmt(row["Total Cost per kg H2"])} EUR/kg<br>'
            f'Gen. cost: {_fmt(row["Gen. cost per kg H2"])} EUR/kg<br>'
            f'Transport: {_fmt(row["Transport Cost per kg H2"])} EUR/kg<br>'
            f'Medium: {medium}<br>'
            f'Capacity cap: {_fmt_cap(row["country_cap_kt"])}'
            for _, row in grp.iterrows()
        ]

        fig.add_trace(go.Scattergeo(
            lat=grp['Latitude'],
            lon=grp['Longitude'],
            mode='markers',
            marker=dict(
                size=grp['_dot_size'].tolist(),
                color=med_colour,
                opacity=0.92,
                line=dict(width=1.0, color='white'),
            ),
            name=MEDIUM_LABELS.get(medium, medium),
            legendgroup=f'medium_{medium}',
            showlegend=show_legend,
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover,
        ))

    # Destination stars
    dest_seen: set = set()
    for _, row in alloc.iterrows():
        cid = row['_corridor']
        if cid in dest_seen:
            continue
        dest_seen.add(cid)
        if pd.isna(row['_dest_lat']) or pd.isna(row['_dest_lon']):
            continue   # domestic corridor — no destination star
        fig.add_trace(go.Scattergeo(
            lat=[row['_dest_lat']],
            lon=[row['_dest_lon']],
            mode='markers+text',
            marker=dict(size=16, color=row['_corr_col'], symbol='star',
                        line=dict(width=1.5, color='white')),
            text=[f' {cid}'],
            textfont=dict(size=11, color=row['_corr_col']),
            textposition='middle right',
            name=f'Destination {cid}',
            legendgroup=f'corridor_{cid}',
            showlegend=False,
            hoverinfo='skip',
        ))

    pct_met = 100 * supplied / demand_kt
    fig.update_layout(
        title=dict(
            text=(
                f'Optimal Supply Mix — {demand_kt/1000:.0f} Mt H₂/yr demand  '
                f'({pct_met:.0f}% met,  wavg cost {wavg_cost:.2f} EUR/kg)<br>'
                '<sup>Dot size ∝ allocated volume  |  Dot colour = transport medium  |  '
                'Line colour = corridor  |  Hover for detail</sup>'
            ),
            font=dict(size=14),
            x=0.5, xanchor='center',
        ),
        geo=dict(
            showland=True, landcolor='#f5f2ee',
            showocean=True, oceancolor='#d4e8f5',
            showcoastlines=True, coastlinecolor='#aaaaaa',
            showcountries=True, countrycolor='#dddddd',
            showframe=False, projection_type='natural earth',
        ),
        legend=dict(
            orientation='v', x=1.01, y=1, xanchor='left',
            bgcolor='rgba(255,255,255,0.92)',
            bordercolor='#cccccc', borderwidth=1,
            font=dict(size=11), tracegroupgap=8,
        ),
        paper_bgcolor='white',
        margin=dict(l=0, r=250, t=100, b=0),
        height=580, width=1150,
    )
    fig.show()

    # ── 5. Merit-order supply curve ───────────────────────────────────────────
    alloc['cum_kt'] = alloc['allocated_kt'].cumsum()
    alloc['cum_kt_start'] = alloc['cum_kt'] - alloc['allocated_kt']

    fig2 = go.Figure()

    for _, row in alloc.iterrows():
        col = MEDIUM_COLOURS.get(row['Cheapest Medium'], '#888888')
        fig2.add_trace(go.Scatter(
            x=[row['cum_kt_start'], row['cum_kt']],
            y=[row['Total Cost per kg H2'], row['Total Cost per kg H2']],
            mode='lines',
            line=dict(color=col, width=6),
            showlegend=False,
            hovertemplate=(
                f'<b>{row["_region"]}</b> [Corridor {row["_corridor"]}]<br>'
                f'Cost: {row["Total Cost per kg H2"]:.3f} EUR/kg<br>'
                f'Volume: {row["allocated_kt"]:,.0f} kt<extra></extra>'
            ),
        ))

    # Demand line
    fig2.add_shape(
        type='line', x0=demand_kt, x1=demand_kt, y0=0, y1=1,
        xref='x', yref='paper',
        line=dict(color='black', dash='dash', width=1.5),
    )
    fig2.add_annotation(
        x=demand_kt, y=0.97, xref='x', yref='paper',
        text=f'Demand<br>{demand_kt/1000:.0f} Mt/yr',
        showarrow=True, arrowhead=2, ax=40, ay=0,
        font=dict(size=11),
    )

    # Medium legend entries
    for medium, col in MEDIUM_COLOURS.items():
        if medium in alloc['Cheapest Medium'].values:
            fig2.add_trace(go.Scatter(
                x=[None], y=[None], mode='lines',
                line=dict(color=col, width=6),
                name=MEDIUM_LABELS.get(medium, medium),
            ))

    fig2.update_layout(
        title=dict(
            text=(f'Merit-Order Supply Curve — Optimal Mix  '
                  f'(wavg {wavg_cost:.2f} EUR/kg, {len(alloc)} sources)<br>'
                  '<sup>Each segment = one country/region, ordered cheapest-first</sup>'),
            font=dict(size=15), x=0.5, xanchor='center',
        ),
        xaxis=dict(title='Cumulative supply [kt H₂/yr]', gridcolor='#e0e0e0'),
        yaxis=dict(title='Delivered cost [EUR/kg H₂]', gridcolor='#e0e0e0'),
        legend=dict(
            orientation='v', x=1.01, y=1, xanchor='left',
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='#cccccc', borderwidth=1,
        ),
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=70, r=220, t=80, b=70),
        height=480, width=1000,
    )
    fig2.show()


# ---------------------------------------------------------------------------
# Chart 5: Detailed cost component breakdown (stacked bar, per-corridor best)
# ---------------------------------------------------------------------------

def plot_cost_component_breakdown(results: dict, h2_demand: float = 15_000,
                                   year: int = 2040, elec_type: str = 'alkaline'):
    """
    Stacked bar chart showing every cost sub-component for the cheapest
    within-cap point in each corridor.

    Components (bottom → top):
        Electricity (solar or wind, whichever is cheaper at that location)
        Electrolyser stack CAPEX (annualised at location-specific WACC)
        Electrical BOP + compression vessel CAPEX (annualised)
        Electrolyser O&M  (2 % of total CAPEX/yr)
        Water  (genuine OPEX, EUR/kg)
        Transport  (cheapest medium at that point)

    All CAPEX items are correctly annualised via the CRF.  Compression
    electricity (4 kWh/kg) flows through the renewable sizing block, so
    it is captured in the Electricity bar — not double-counted here.
    """
    from country_factors import WACC as WACC_MAP
    from generation_costs import annualise

    year_diff                = max(0, min(30, year - 2020))
    full_load_hours          = 4000
    other_capex_elec         = 41.6                         # Electrical BOP [EUR/kW]
    compression_capex_per_kw = 35.0 * (0.985 ** year_diff) # Compression vessel [EUR/kW]
    elec_opex                = 0.02
    water_cost               = 0.07
    comp_elec                = 4

    if elec_type == 'alkaline':
        capex_h2_global  = 650  * (0.97  ** year_diff)
        lifetime_hours   = 80000 + 1333 * year_diff
        electrolyser_eff = 0.640 + 0.0057 * year_diff
    elif elec_type == 'SOEC':
        capex_h2_global  = 2500 * (0.95  ** year_diff)
        lifetime_hours   = 20000 + 2667 * year_diff
        electrolyser_eff = 0.80 + 0.0013 * year_diff
    else:  # PEM
        capex_h2_global  = 900  * (0.97  ** year_diff)
        lifetime_hours   = 60000 + 2000 * year_diff
        electrolyser_eff = 0.68 + 0.0020 * year_diff

    electrolyser_lifetime = lifetime_hours / full_load_hours
    kg_per_yr             = h2_demand * 1e6
    elec_demand_mw        = (h2_demand * 1000 / full_load_hours * 33.3 / electrolyser_eff
                              + comp_elec * h2_demand * 1000 / full_load_hours)

    rows = []
    for cid, res in results.items():
        cfg    = res['corridor']
        df     = res['df']
        within = df[df['within_cap'] == True].dropna(subset=['Total Cost per kg H2'])
        if within.empty:
            continue

        row = within.loc[within['Total Cost per kg H2'].idxmin()]

        wacc         = WACC_MAP.get(row.get('H2_Region', 'Other'), 0.09)
        yearly_solar = row['Yearly Cost Solar']
        yearly_wind  = row['Yearly Cost Wind']
        elec_per_kg  = min(yearly_solar, yearly_wind) / kg_per_yr
        elec_source  = 'Solar' if yearly_solar <= yearly_wind else 'Wind'

        # Separate stack CAPEX from BOP+compression CAPEX for the chart
        stack_capex_eur        = capex_h2_global * elec_demand_mw * 1000
        bop_comp_capex_eur     = (other_capex_elec + compression_capex_per_kw) * elec_demand_mw * 1000
        total_capex_h2         = stack_capex_eur + bop_comp_capex_eur

        stack_capex_per_kg     = annualise(stack_capex_eur,    wacc, electrolyser_lifetime) / kg_per_yr
        bop_comp_per_kg        = annualise(bop_comp_capex_eur, wacc, electrolyser_lifetime) / kg_per_yr
        stack_om_per_kg        = elec_opex * total_capex_h2 / kg_per_yr

        rows.append({
            'label':       f'Corridor {cid}',
            'country':     row.get('Country', '?'),
            'medium':      row.get('Cheapest Medium', '?'),
            'elec_source': elec_source,
            'colour':      cfg['colour'],
            'electricity': elec_per_kg,
            'stack_capex': stack_capex_per_kg,
            'bop_comp':    bop_comp_per_kg,
            'stack_om':    stack_om_per_kg,
            'water':       water_cost,
            'transport':   row['Transport Cost per kg H2'],
        })

    if not rows:
        print('No corridor data for component breakdown.')
        return

    comp = pd.DataFrame(rows)

    # Colour palette for components (consistent across calls)
    COMP_COLOURS = {
        'electricity': '#FFD166',   # yellow
        'stack_capex': '#06D6A0',   # teal
        'bop_comp':    '#073B4C',   # dark navy
        'stack_om':    '#118AB2',   # blue
        'water':       '#8ECAE6',   # light blue
        'transport':   '#EF476F',   # red-pink
    }
    COMP_LABELS = {
        'electricity': 'Electricity (renewable)',
        'stack_capex': 'Electrolyser stack CAPEX (annualised)',
        'bop_comp':    'BOP + compression CAPEX (annualised)',
        'stack_om':    'O&M (2% CAPEX/yr)',
        'water':       'Water',
        'transport':   'Transport',
    }
    COMPONENTS = ['electricity', 'stack_capex', 'bop_comp', 'stack_om', 'water', 'transport']

    fig = go.Figure()

    # One trace per component (stacked)
    for comp_key in COMPONENTS:
        fig.add_trace(go.Bar(
            name=COMP_LABELS[comp_key],
            x=comp['label'],
            y=comp[comp_key],
            marker_color=COMP_COLOURS[comp_key],
            hovertemplate=(
                f'<b>{COMP_LABELS[comp_key]}</b><br>'
                '%{y:.3f} EUR/kg<extra></extra>'
            ),
        ))

    # Annotations: country + medium above each bar
    totals = comp[COMPONENTS].sum(axis=1)
    for i, row in comp.iterrows():
        total = totals.iloc[i]
        fig.add_annotation(
            x=row['label'],
            y=total + 0.05,
            text=(f"<b>{row['country']}</b><br>"
                  f"{row['elec_source']} → {row['medium']}"),
            showarrow=False,
            font=dict(size=10),
            align='center',
            yanchor='bottom',
        )

    fig.update_layout(
        barmode='stack',
        title=dict(
            text=(f'H\u2082 Cost Component Breakdown — Cheapest Point per Corridor<br>'
                  f'<sup>{year}, {elec_type} electrolyser, {h2_demand:,} kt/yr demand</sup>'),
            font=dict(size=16), x=0.5, xanchor='center',
        ),
        xaxis=dict(title='Corridor'),
        yaxis=dict(
            title='Cost [EUR/kg H\u2082]',
            gridcolor='#e0e0e0',
            zeroline=True,
        ),
        legend=dict(
            orientation='v', x=1.01, y=1, xanchor='left',
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='#cccccc', borderwidth=1,
            traceorder='normal',
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=70, r=260, t=100, b=70),
        height=520,
        width=900,
    )

    fig.show()


# ---------------------------------------------------------------------------
# Emissions component breakdown chart
# ---------------------------------------------------------------------------

def plot_emissions_component_breakdown(results: dict, year: int = 2040,
                                       elec_type: str = 'alkaline',
                                       heat_source: str = 'renewable'):
    """
    Stacked bar chart showing generation and transport emissions per kg H2
    across all within-cap source locations in each corridor.

    Each bar shows the median generation + median transport emissions across
    all within-cap points. Individual points are overlaid as a scatter to show
    the spread across the corridor.

    Horizontal dashed line at EU RED III threshold (3.38 kgCO2eq/kgH2).

    Uses pre-computed 'Gen. emissions per kg H2' and 'Transport Emissions per
    kg H2' columns set by emissions.py — one value per source location.
    """
    from emissions import EU_RED3_KGC02_KGH2
    from corridors import CORRIDORS as _COR

    COLOURS = {
        'gen':       '#FFD166',   # yellow — generation
        'transport': '#0072B2',   # blue   — transport
    }
    LABELS = {
        'gen':       'Generation emissions (renewable electricity + electrolyser mfg)',
        'transport': 'Transport emissions (conditioning + shipping + reconversion)',
    }

    rows   = []
    scatter_x, scatter_y, scatter_colours = [], [], []

    for cid, res in results.items():
        cfg    = _COR.get(cid, res['corridor']) or res['corridor'] or {}
        df     = res['df']
        within = df[df['within_cap'] == True].dropna(
            subset=['Gen. emissions per kg H2', 'Transport Emissions per kg H2']
        )
        if within.empty:
            continue

        gen_col   = within['Gen. emissions per kg H2']
        trans_col = within['Transport Emissions per kg H2']
        total_col = gen_col + trans_col

        label = f'Corridor {cid}'
        rows.append({
            'label':     label,
            'colour':    cfg.get('colour', '#888888'),
            'gen':       gen_col.median(),
            'transport': trans_col.median(),
            'n_points':  len(within),
            'pct_below': (total_col < EU_RED3_KGC02_KGH2).mean() * 100,
        })

        # Individual points for scatter overlay (sample up to 300 for readability)
        sample = within.sample(min(300, len(within)), random_state=42)
        scatter_x.extend([label] * len(sample))
        scatter_y.extend((sample['Gen. emissions per kg H2'] +
                          sample['Transport Emissions per kg H2']).tolist())
        scatter_colours.extend([cfg.get('colour', '#888888')] * len(sample))

    if not rows:
        print('No corridor data for emissions component breakdown.')
        return

    comp_df = pd.DataFrame(rows)
    fig     = go.Figure()

    # Stacked bars: generation + transport (median per corridor)
    for key in ('gen', 'transport'):
        fig.add_trace(go.Bar(
            name=LABELS[key],
            x=comp_df['label'],
            y=comp_df[key],
            marker_color=COLOURS[key],
            hovertemplate=(
                f'<b>{LABELS[key]}</b><br>'
                'Median: %{y:.3f} kgCO\u2082eq/kgH\u2082<extra></extra>'
            ),
        ))

    # Scatter overlay: all individual within-cap points
    fig.add_trace(go.Scatter(
        x=scatter_x,
        y=scatter_y,
        mode='markers',
        name='Individual source locations',
        marker=dict(
            color='rgba(80,80,80,0.25)',
            size=4,
            symbol='circle',
        ),
        hovertemplate='%{y:.3f} kgCO\u2082eq/kgH\u2082<extra></extra>',
    ))

    # EU RED III threshold line
    fig.add_hline(
        y=EU_RED3_KGC02_KGH2,
        line_dash='dash',
        line_color='red',
        line_width=2,
        annotation_text=f'EU RED III threshold ({EU_RED3_KGC02_KGH2} kgCO\u2082eq/kgH\u2082)',
        annotation_position='top right',
        annotation_font=dict(color='red', size=11),
    )

    # Annotations above each bar: n locations + % below RED III
    for _, r in comp_df.iterrows():
        total = r['gen'] + r['transport']
        fig.add_annotation(
            x=r['label'],
            y=total + 0.05,
            text=f"n={r['n_points']:,}<br>{r['pct_below']:.0f}% below RED III",
            showarrow=False,
            font=dict(size=9),
            align='center',
            yanchor='bottom',
        )

    heat_label = 'renewable electricity' if heat_source == 'renewable' else 'natural gas'
    fig.update_layout(
        barmode='stack',
        title=dict(
            text=(f'H\u2082 Lifecycle Emissions — All Within-Cap Locations per Corridor<br>'
                  f'<sup>Median bars + individual points | {year}, {elec_type} electrolyser, '
                  f'reconversion: {heat_label}</sup>'),
            font=dict(size=16), x=0.5, xanchor='center',
        ),
        xaxis=dict(title='Corridor'),
        yaxis=dict(
            title='Emissions [kgCO\u2082eq/kgH\u2082]',
            gridcolor='#e0e0e0',
            zeroline=True,
        ),
        legend=dict(
            orientation='v', x=1.01, y=1, xanchor='left',
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='#cccccc', borderwidth=1,
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=70, r=320, t=110, b=70),
        height=540,
        width=980,
    )

    fig.show()


# ---------------------------------------------------------------------------
# Normalised emissions distribution (density histogram)
# ---------------------------------------------------------------------------

def plot_emissions_distribution(results: dict, year: int = 2040,
                                elec_type: str = 'alkaline',
                                heat_source: str = 'renewable'):
    """
    Normalised probability-density histogram of total lifecycle emissions per
    corridor, so corridors with very different point counts (e.g. the global
    Corridor C vs. the narrow Corridor D) are visually comparable.

    Each corridor is drawn as a semi-transparent filled histogram with
    histnorm='probability density'.  A vertical dashed line marks the EU
    RED III threshold.  Median per corridor is shown as a vertical tick.
    """
    from emissions import EU_RED3_KGC02_KGH2
    from corridors import CORRIDORS as _COR

    fig = go.Figure()

    for cid, res in results.items():
        cfg    = _COR.get(cid, res['corridor']) or res['corridor'] or {}
        df     = res['df']
        within = df[df['within_cap'] == True].dropna(
            subset=['Gen. emissions per kg H2', 'Transport Emissions per kg H2']
        )
        if within.empty:
            continue

        total   = (within['Gen. emissions per kg H2'] +
                   within['Transport Emissions per kg H2'])
        colour  = cfg.get('colour', '#888888')
        label   = f"Corridor {cid} — {cfg.get('subtitle', '')} (n={len(total):,})"
        median  = total.median()

        # Normalised histogram (probability density — area sums to 1)
        fig.add_trace(go.Histogram(
            x=total,
            name=label,
            histnorm='probability density',
            nbinsx=40,
            marker_color=colour,
            opacity=0.45,
            hovertemplate=(
                f'<b>Corridor {cid}</b><br>'
                'Emissions: %{x:.2f} kgCO\u2082eq/kgH\u2082<br>'
                'Density: %{y:.3f}<extra></extra>'
            ),
        ))

        # Median tick line per corridor
        fig.add_vline(
            x=median,
            line_dash='dot',
            line_color=colour,
            line_width=1.5,
            annotation_text=f'{cid} median<br>{median:.2f}',
            annotation_position='top',
            annotation_font=dict(color=colour, size=9),
        )

    # EU RED III threshold
    fig.add_vline(
        x=EU_RED3_KGC02_KGH2,
        line_dash='dash',
        line_color='red',
        line_width=2,
        annotation_text=f'EU RED III ({EU_RED3_KGC02_KGH2} kgCO\u2082eq/kgH\u2082)',
        annotation_position='top left',
        annotation_font=dict(color='red', size=11),
    )

    heat_label = 'renewable electricity' if heat_source == 'renewable' else 'natural gas'
    fig.update_layout(
        barmode='overlay',
        title=dict(
            text=(f'H\u2082 Lifecycle Emissions — Normalised Distribution per Corridor<br>'
                  f'<sup>Probability density | {year}, {elec_type} electrolyser, '
                  f'reconversion: {heat_label}</sup>'),
            font=dict(size=16), x=0.5, xanchor='center',
        ),
        xaxis=dict(
            title='Total lifecycle emissions [kgCO\u2082eq/kgH\u2082]',
            gridcolor='#e0e0e0',
        ),
        yaxis=dict(
            title='Probability density',
            gridcolor='#e0e0e0',
        ),
        legend=dict(
            orientation='v', x=1.01, y=1, xanchor='left',
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='#cccccc', borderwidth=1,
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=70, r=320, t=110, b=70),
        height=520,
        width=1000,
    )

    fig.show()


# ---------------------------------------------------------------------------
# Entry point (standalone test)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    # Allow loading saved corridor CSVs for quick re-plotting without re-running
    # Usage: python plot_corridor.py [year]  e.g. python plot_corridor.py 2030
    _year = int(sys.argv[1]) if len(sys.argv) > 1 else 2040
    results_from_csv = {}
    for cid, cfg in CORRIDORS.items():
        path = f'Results/corridor_{cid}_{_year}.csv'
        try:
            df = pd.read_csv(path, index_col=0)
            results_from_csv[cid] = {'corridor': cfg, 'df': df}
            print(f'  Loaded {path}  ({len(df)} rows)')
        except FileNotFoundError:
            print(f'  Not found: {path}  (run run_corridors.py first)')

    if results_from_csv:
        plot_corridor_supply_curves(results_from_csv)
        plot_corridor_cost_breakdown(results_from_csv)
        plot_cost_component_breakdown(results_from_csv)
        plot_emissions_component_breakdown(results_from_csv, heat_source='renewable')
        plot_emissions_distribution(results_from_csv, year=_year, heat_source='renewable')
        plot_corridor_flow_map(results_from_csv, n_countries=15)
        plot_optimal_supply_mix(results_from_csv, demand_kt=15_000)
    else:
        print('No corridor CSV files found. Run run_corridors.py first.')
