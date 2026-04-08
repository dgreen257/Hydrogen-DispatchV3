"""
country_factors.py

Location-based cost adjustment factors applied in generation_costs.py.

Two factors are applied per region:

1. CAPEX MULTIPLIER (solar and wind separately)
   Reflects differences in installation costs: labour, logistics, supply chain
   maturity, permitting complexity, and grid connection costs.
   Baseline = 1.0 (global reference). Sources: IRENA Renewable Power Generation
   Costs 2023; IEA World Energy Outlook 2023; BloombergNEF H2 cost data.

2. WACC (weighted average cost of capital / discount rate)
   Country-risk-adjusted cost of capital. Applied via CRF to annualise CAPEX.
   Two technology-specific values are used:
     - wacc_ren  (renewable): from yearly-costsofcapital-global-onshore-wind.csv
     - wacc_elec (electrolyser): from yearly-costsofcapital-global-electrolysis.csv
       (~2% higher due to technology risk premium on less-proven electrolysis)
   Source: Damodaran country risk + technology risk methodology (177 countries, 2024).
   The WACC regional dict below serves as a fallback for the ~17 unmatched countries.
"""
import csv as _csv
import os as _os

# ---------------------------------------------------------------------------
# Solar capex multipliers by H2_Region
# Reference: 1.0 = global utility-scale baseline (~350 EUR/kWp in 2024)
# ---------------------------------------------------------------------------
SOLAR_CAPEX_FACTOR = {
    "EU":                 1.30,   # High labour, permitting, BOS costs
    "Non-EU Europe":      1.20,   # Somewhat lower than EU but still high
    "North Africa":       0.85,   # Low labour, excellent logistics from EU proximity
    "Sub-Saharan Africa": 1.05,   # Low labour offset by logistics/import costs
    "Middle East":        0.85,   # Low labour, mature large-project experience
    "Central Asia":       1.00,   # Moderate — limited project pipeline
    "South Asia":         0.75,   # India especially competitive; low labour
    "East Asia":          0.85,   # China manufacturing base proximity
    "Southeast Asia":     0.90,   # Growing market, moderate costs
    "Oceania":            1.10,   # Australia: high wages, remote sites
    "North America":      1.00,   # US reference market
    "Latin America":      0.90,   # Chile competitive; Brazil moderate
    "Russia":             1.10,   # Logistics and supply chain constraints
    "Other":              1.00,   # Default fallback
}

# ---------------------------------------------------------------------------
# Wind capex multipliers by H2_Region
# Reference: 1.0 = global onshore wind baseline (~1050 EUR/kW in 2024)
# Note: these are onshore figures. Offshore would be ~1.8-2.5x onshore.
# ---------------------------------------------------------------------------
WIND_CAPEX_FACTOR = {
    "EU":                 1.40,   # High labour, strict permitting, long lead times
    "Non-EU Europe":      1.25,   # Norway/UK: offshore premium; others moderate
    "North Africa":       0.85,   # Low labour; Morocco has mature wind sector
    "Sub-Saharan Africa": 1.05,   # Low labour, logistics offset
    "Middle East":        0.90,   # Some experience; Saudi/UAE projects competitive
    "Central Asia":       1.05,   # Limited infrastructure
    "South Asia":         0.85,   # India competitive wind market
    "East Asia":          0.85,   # China manufacturing proximity
    "Southeast Asia":     0.95,   # Moderate
    "Oceania":            1.15,   # Australia: remote sites, high wages
    "North America":      1.00,   # US reference market
    "Latin America":      0.90,   # Chile and Argentina competitive
    "Russia":             1.10,   # Logistics constraints
    "Other":              1.00,   # Default fallback
}

# ---------------------------------------------------------------------------
# WACC by H2_Region (%)
# Applied to annualisation of solar, wind, and electrolyser capex.
# Reflects country risk premium on top of base cost of capital.
# ---------------------------------------------------------------------------
WACC = {
    "EU":                 0.041,   # 4.1% — low sovereign risk, green finance access
    "Non-EU Europe":      0.065,   # 6.5% — Norway/UK lower; Balkans/Ukraine higher
    "North Africa":       0.080,   # 8.0% — moderate risk; EU proximity helps
    "Sub-Saharan Africa": 0.120,   # 12.0% — high political/currency risk
    "Middle East":        0.075,   # 7.5% — Gulf states lower; Iraq/Yemen higher
    "Central Asia":       0.110,   # 11.0% — high risk, limited green finance
    "South Asia":         0.095,   # 9.5% — India ~8%; Pakistan/Bangladesh higher
    "East Asia":          0.038,   # 3.8% — China/Korea competitive financing
    "Southeast Asia":     0.085,   # 8.5% — varies widely across region
    "Oceania":            0.065,   # 6.5% — Australia low risk
    "North America":      0.054,   # 6.5% — US/Canada stable
    "Latin America":      0.085,   # 8.5% — Chile lower (~7%); others higher
    "Russia":             0.130,   # 13.0% — sanctions, very high risk premium
    "Other":              0.090,   # Default fallback
}


def _load_wacc_csv(filename):
    """Load a WACC CSV from /Data/, return {iso_code: wacc_decimal}.
    CSV values are in percentage terms (e.g. 11.29) and are divided by 100."""
    path = _os.path.join(_os.path.dirname(__file__), 'Data', filename)
    result = {}
    with open(path, newline='') as f:
        reader = _csv.DictReader(f)
        for row in reader:
            code = row['Country code'].strip()
            try:
                result[code] = float(row['WACC']) / 100.0
            except (ValueError, KeyError):
                pass
    return result


# Country-level WACC from CSV datasets (177 countries, 2024 data).
# Wind ≈ solar WACC throughout; wind CSV used for renewable WACC.
# ~17 countries in renewables.csv have no match → fall back to WACC regional dict.
WACC_COUNTRY_REN  = _load_wacc_csv('yearly-costsofcapital-global-onshore-wind.csv')
WACC_COUNTRY_ELEC = _load_wacc_csv('yearly-costsofcapital-global-electrolysis.csv')


def get_region_factors(h2_region):
    """
    Returns (solar_capex_factor, wind_capex_factor, wacc_ren, wacc_elec)
    for a given H2_Region string. Falls back to 'Other' defaults if not recognised.
    wacc_ren  = renewable WACC (solar/wind)
    wacc_elec = electrolyser WACC (~2% higher due to technology risk premium)
    """
    solar     = SOLAR_CAPEX_FACTOR.get(h2_region, SOLAR_CAPEX_FACTOR["Other"])
    wind      = WIND_CAPEX_FACTOR.get(h2_region, WIND_CAPEX_FACTOR["Other"])
    wacc_ren  = WACC.get(h2_region, WACC["Other"])
    wacc_elec = wacc_ren + 0.020   # +2% technology risk premium for electrolysis
    return solar, wind, wacc_ren, wacc_elec
