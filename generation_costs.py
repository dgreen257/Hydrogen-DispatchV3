import math

import pandas as pd
from country_factors import SOLAR_CAPEX_FACTOR, WIND_CAPEX_FACTOR, WACC, WACC_COUNTRY_REN, WACC_COUNTRY_ELEC

# Offshore wind CAPEX multiplier relative to the onshore global baseline.
# Fixed-bottom offshore installation is ~2.5× onshore due to foundations,
# marine logistics and subsea cable costs.
# Sources: IRENA Offshore Wind Outlook 2019; BloombergNEF LCOE 2024.
OFFSHORE_CAPEX_MULT = 2.5


def annualise(capex, interest, lifetime):
    """Annualises a CapEx investment using the capital recovery factor formula.
    capex, interest, and lifetime can be scalars or pandas Series (vectorised)."""
    return capex * (interest / (1 - (1 + interest) ** (-lifetime)))


def global_capex(year: int, elec_type: str = 'alkaline') -> dict:
    """Return global (pre-location-adjustment) baseline CAPEX and efficiency for a given year.
    Mirrors the cost-curve logic in generation_costs() — import this for dashboard/diagnostic plots."""
    year_diff = max(0, min(24, year - 2026))
    if year <= 2030:
        capex_wind  = 1200 * (0.9775 ** year_diff)
        capex_solar = 700  * (0.9500 ** year_diff)
    else:
        capex_wind  = 1200 * (0.9775 ** 4) * (0.9950 ** (year - 2030))
        capex_solar = 700  * (0.9500 ** 4) * (0.9800 ** (year - 2030))
    if elec_type == 'alkaline':
        capex_elec = (850 * (0.917 ** year_diff) if year <= 2030
                      else 850 * (0.917 ** 4) * (0.976 ** (year - 2030)))
        efficiency = 0.640 + 0.0057 * year_diff
    elif elec_type == 'SOEC':
        capex_elec = 2200 * (0.95 ** year_diff)
        efficiency = 0.808 + 0.0013 * year_diff
    else:  # PEM
        capex_elec = 1000 * (0.97 ** year_diff)
        efficiency = 0.692 + 0.0020 * year_diff
    return {
        'capex_wind':  round(capex_wind,  1),
        'capex_solar': round(capex_solar, 1),
        'capex_elec':  round(capex_elec,  1),
        'efficiency':  round(efficiency,  4),
    }


def generation_costs(df_ren, h2_demand, year=2020, elec_type='alkaline',
                     full_load_hours=6000, location_adjusted=True,
                     capex_solar_override=None,
                     capex_wind_override=None,
                     capex_elec_override=None):
    """
    Calculates H2 generation cost (yearly and per kg) for every grid point.

    Parameters
    ----------
    df_ren          : pd.DataFrame — main renewables dataframe (must have H2_Region column)
    h2_demand       : float        — annual H2 demand [kt/yr]
    year            : int          — target year (2020–2050)
    elec_type       : str          — electrolyser technology: 'alkaline', 'SOEC', or 'PEM'
    full_load_hours : int          — electrolyser operating hours per year [hrs/yr]
    location_adjusted : bool       — if True, apply per-region capex and WACC factors
                                     from country_factors.py; if False, use flat global values

    Cost components
    ---------------
    - Solar or wind electricity (cheapest per location)
    - Electrolyser capex + opex
    - Balance-of-plant, compression, water

    Location adjustment (when location_adjusted=True)
    --------------------------------------------------
    - Solar/wind capex multiplied by region-specific installation cost factor
    - Annualisation uses region-specific WACC instead of a flat global rate
    - Electrolyser WACC is also region-adjusted (same project, same financing)
    """

    # ------------------------------------------------------------------
    # Fixed parameters
    # ------------------------------------------------------------------
    opex_factor_solar = 0.045   # Solar O&M as fraction of CapEx [/yr]
    opex_wind         = 8       # Wind variable O&M [EUR/MWh]
    wind_efficiency   = 0.40    # Betz efficiency factor
    blade             = 50      # Turbine blade length [m]
    turbine_size      = 2       # Turbine nameplate capacity [MW]
    elec_opex         = 0.02    # Electrolyser O&M as fraction of CapEx [/yr]
    comp_elec         = 4       # Compression electricity [kWh/kg H2] — operating energy; already flows
                                # through elec_demand → renewable sizing (NOT double-counted below)
    other_capex_elec  = 41.6    # Electrical BOP capex [EUR/kW]: power electronics, transformers,
                                # switchgear, cabling — annualised via total_capex_h2
    water_cost        = 0.07    # Water OPEX [EUR/kg H2] — genuinely per-kg operating cost

    # ------------------------------------------------------------------
    # Year index (0 = 2026, 24 = 2050)
    # ------------------------------------------------------------------
    year_diff = max(0, min(24, year - 2026))

    # ------------------------------------------------------------------
    # Global (pre-location-adjustment) technology cost curves
    # Sources: IRENA Renewable Power Generation Costs 2023;
    #          IEA Global Hydrogen Review 2023; BNEF 2024
    # ------------------------------------------------------------------

    # Wind: 2.25 %/yr decline to 2030, then 0.5 %/yr thereafter
    # Baseline: 1200 EUR/kW (2026 global weighted average onshore, IRENA/IEA)
    # At 2030 (year_diff=4): ~1097 EUR/kW
    if year <= 2030:
        capex_wind_global = 1200 * (0.9775 ** year_diff)
    else:
        capex_wind_global = 1200 * (0.9775 ** 4) * (0.9950 ** (year - 2030))

    # Solar: ~5 %/yr decline to 2030, then ~2 %/yr thereafter
    # Baseline: 700 EUR/kWp (2026 global weighted average utility-scale, IRENA/IEA)
    # At 2030 (year_diff=4): ~569 EUR/kWp
    if year <= 2030:
        capex_solar_global = 700 * (0.9500 ** year_diff)
    else:
        capex_solar_global = 700 * (0.9500 ** 4) * (0.9800 ** (year - 2030))

    # ------------------------------------------------------------------
    # Electrolyser parameters by technology
    # Sources: IEA Global Hydrogen Review 2023; IRENA 2023; BNEF 2024
    # ------------------------------------------------------------------
    # Compression CAPEX [EUR/kW of electrolyser capacity].
    # Covers compressor vessel + motor to raise H2 from ~30 bar (electrolyser
    # outlet) to ~70–200 bar (pipeline / storage).  Compression ELECTRICITY
    # (4 kWh/kg) is handled separately via comp_elec → elec_demand → renewable
    # sizing above, so there is no double-counting here.
    # This is added to total_capex_h2 and correctly annualised via the CRF.
    # Base: 32 EUR/kW (2026, IRENA/IEA); declining 1.5 %/yr as compressor
    # manufacturing matures.  Literature range: 30–60 EUR/kW.
    compression_capex_per_kw = 32.0 * (0.985 ** year_diff)

    if elec_type == 'alkaline':
        if year <= 2030:
            capex_h2_global  = 850 * (0.917 ** year_diff)            # EUR/kW; 850→~601 by 2030 (~8.3%/yr)
        else:
            capex_h2_global  = 850 * (0.917 ** 4) * (0.976 ** (year - 2030))  # ~601→~471 by 2040 (~2.4%/yr)
        lifetime_hours       = 88000 + 1333 * year_diff     # hrs; ~88k hrs in 2026 → ~120k by 2050
        electrolyser_eff     = 0.640 + 0.0057 * year_diff   # LHV; 64% in 2026 → ~72% in 2040
    elif elec_type == 'SOEC':
        capex_h2_global      = 2200 * (0.95 ** year_diff)   # EUR/kW; 2026 early-commercial baseline
        lifetime_hours       = 36000 + 2667 * year_diff     # hrs; improving rapidly
        electrolyser_eff     = 0.808 + 0.0013 * year_diff   # LHV; highest efficiency
    else:  # PEM
        capex_h2_global      = 1000 * (0.97 ** year_diff)   # EUR/kW; ~alkaline in 2026
        lifetime_hours       = 72000 + 2000 * year_diff     # hrs
        electrolyser_eff     = 0.692 + 0.0020 * year_diff   # LHV; comparable to alkaline

    electrolyser_lifetime = lifetime_hours / full_load_hours  # [years]

    # ------------------------------------------------------------------
    # CAPEX overrides (from dashboard sliders — replace year-based defaults)
    # ------------------------------------------------------------------
    if capex_solar_override is not None:
        capex_solar_global = capex_solar_override
    if capex_wind_override is not None:
        capex_wind_global = capex_wind_override
    if capex_elec_override is not None:
        capex_h2_global = capex_elec_override

    # ------------------------------------------------------------------
    # Electricity demand sizing
    # ------------------------------------------------------------------
    h2_demand_hourly   = h2_demand * 1000 / full_load_hours          # [t/hr]
    elec_demand        = (h2_demand_hourly * 33.3 / electrolyser_eff   # [MW] continuous
                          + comp_elec * h2_demand_hourly)
    elec_demand_yearly = (h2_demand * 1000 * 33.3 / electrolyser_eff   # [MWh/yr]
                          + comp_elec * h2_demand * 1000)

    # ------------------------------------------------------------------
    # Per-row location factors (region-specific capex multiplier + country-level WACC)
    # ------------------------------------------------------------------
    if location_adjusted and 'H2_Region' in df_ren.columns:
        solar_factor  = df_ren['H2_Region'].map(SOLAR_CAPEX_FACTOR).fillna(1.0)
        wind_factor   = df_ren['H2_Region'].map(WIND_CAPEX_FACTOR).fillna(1.0)
        # Regional WACC as fallback for the ~17 countries not in the CSV datasets
        regional_wacc = df_ren['H2_Region'].map(WACC).fillna(0.08)
        if 'ISO_A3' in df_ren.columns:
            # Country-level, technology-specific WACC from CSV datasets
            wacc_ren  = df_ren['ISO_A3'].map(WACC_COUNTRY_REN).fillna(regional_wacc)
            wacc_elec = df_ren['ISO_A3'].map(WACC_COUNTRY_ELEC).fillna(regional_wacc + 0.02)
        else:
            wacc_ren  = regional_wacc
            wacc_elec = regional_wacc + 0.02
    else:
        solar_factor = 1.0
        wind_factor  = 1.0
        wacc_ren     = 0.08   # flat global fallback
        wacc_elec    = 0.10   # +2% technology risk premium

    capex_solar = capex_solar_global * solar_factor   # [EUR/kWp] per row

    # Offshore override: replace region-adjusted onshore CAPEX with a flat
    # global offshore multiplier for rows where Is_Offshore == True.
    if 'Is_Offshore' in df_ren.columns:
        offshore_mask = df_ren['Is_Offshore'].fillna(False).infer_objects(copy=False).astype(bool)
        if offshore_mask.any():
            capex_wind = pd.Series(
                float(capex_wind_global) * wind_factor,
                index=df_ren.index, dtype=float,
            )
            capex_wind[offshore_mask] = float(capex_wind_global) * OFFSHORE_CAPEX_MULT
        else:
            capex_wind = capex_wind_global * wind_factor
    else:
        capex_wind = capex_wind_global * wind_factor  # [EUR/kW]  per row

    # ------------------------------------------------------------------
    # Solar costs
    # ------------------------------------------------------------------
    # Solar Energy Potential column: kWh/kWp/yr (location-specific yield)
    df_ren['Solar Array Size']  = elec_demand_yearly * 1000 / df_ren['Solar Energy Potential']  # [kWp]
    df_ren['Solar CapEx']       = df_ren['Solar Array Size'] * capex_solar                       # [EUR]
    df_ren['Yearly Cost Solar'] = (annualise(df_ren['Solar CapEx'], wacc_ren, 25)
                                   + opex_factor_solar * df_ren['Solar CapEx'])                  # [EUR/yr]
    df_ren['Elec Cost Solar']   = df_ren['Yearly Cost Solar'] / elec_demand_yearly              # [EUR/MWh]

    # ------------------------------------------------------------------
    # Wind costs
    # ------------------------------------------------------------------
    # Wind Power Density column: W/m² (location-specific average)
    # Cap at a physically plausible maximum: 800 W/m² ≈ 11.4 m/s mean wind speed.
    # Values above this threshold are data artefacts in the source grid (e.g. some
    # coastal cells inherit anomalously high values from the underlying wind atlas).
    WPD_CAP = 800.0
    wpd_capped = df_ren['Wind Power Density'].clip(upper=WPD_CAP)

    capex_turbine = turbine_size * capex_wind * 1000                                                              # [EUR/turbine]
    df_ren['Wind Turbine Power'] = (wpd_capped * wind_efficiency
                                    * (blade ** 2) * math.pi / 1e6)                              # [MW/turbine]
    df_ren['No. of Turbines']    = elec_demand / df_ren['Wind Turbine Power']
    df_ren['Wind CapEx']         = df_ren['No. of Turbines'] * capex_turbine                     # [EUR]
    df_ren['Yearly Cost Wind']   = (annualise(df_ren['Wind CapEx'], wacc_ren, 20)
                                    + opex_wind * elec_demand_yearly)                            # [EUR/yr]
    df_ren['Elec Cost Wind']     = df_ren['Yearly Cost Wind'] / elec_demand_yearly              # [EUR/MWh]

    # Cheapest electricity source per location
    df_ren['Cheaper source'] = ['Solar' if x < y else 'Wind'
                                for x, y in zip(df_ren['Yearly Cost Solar'], df_ren['Yearly Cost Wind'])]

    # ------------------------------------------------------------------
    # Electrolyser costs (same technology everywhere, WACC is location-adjusted)
    # ------------------------------------------------------------------
    # Stack + electrical BOP + compression vessel — all correctly annualised via CRF
    total_capex_h2 = (capex_h2_global + other_capex_elec + compression_capex_per_kw) * elec_demand * 1000   # [EUR]
    df_ren['Yearly Cost Electrolyser'] = (annualise(total_capex_h2, wacc_elec, electrolyser_lifetime)
                                          + elec_opex * total_capex_h2
                                          + water_cost * h2_demand * 1e6)                        # [EUR/yr]

    # ------------------------------------------------------------------
    # Total generation cost
    # ------------------------------------------------------------------
    df_ren['Yearly gen. cost'] = (df_ren[['Yearly Cost Solar', 'Yearly Cost Wind']].min(axis=1)
                                  + df_ren['Yearly Cost Electrolyser'])                          # [EUR/yr]
    df_ren['Gen. cost per kg H2'] = df_ren['Yearly gen. cost'] / (h2_demand * 1e6)             # [EUR/kg H2]

    return df_ren
