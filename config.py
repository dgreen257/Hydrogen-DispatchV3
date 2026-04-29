"""
config.py
---------
Central parameter store for all model runs.
"""

scenario          = 'Current trajectory'   # 'Ambitious', 'Intermediate', 'Current trajectory'
year              = 2040           # used by run_corridors.py; main.py uses run_years instead
elec_type         = 'alkaline'       # 'alkaline', 'PEM', 'SOEC'
heat_source       = 'renewable'      # 'renewable' or 'gas' (affects reconversion emissions)
centralised       = True             # centralised reconversion of NH3/LOHC/LH2
pipeline          = True             # allow pipelines (individual corridors may override)
max_pipeline_dist = 2000             # max pipeline distance [km]
end_tuple         = (37.26, -6.94)    # destination [lat, lon] — used by main.py only (Rotterdam)

# Years for multi-year main.py runs (transport costs computed once, gen costs loop over these)
run_years         = list(range(2026, 2041))


# Demand profiles [year → kt H₂/yr]

DEMAND_PROFILES: dict = {
    'Ambitious': {
        2025:   500, 2027:  4_700, 2030: 11_600,
        2035: 20_000, 2040: 23_800, 2045: 31_300, 2050: 46_000,
    },
    'Intermediate': {
        2025:   400, 2027:  2_700, 2030:  5_900,
        2035:  9_500, 2040: 12_100, 2045: 16_700, 2050: 25_000,
    },
    'Current trajectory': {
        2025:   300, 2027:  1_000, 2030:  1_700,
        2035:  2_500, 2040:  4_000, 2045:  6_000, 2050:  9_000,
    },
}


def demand_for_year(scenario: str, year: int) -> float:
    """
    Return demand [kt/yr] for a given scenario and year,
    linearly interpolating between anchor years.
    Clamps to the first/last value outside the defined range.
    """
    profile = DEMAND_PROFILES[scenario]
    years   = sorted(profile.keys())
    if year <= years[0]:
        return profile[years[0]]
    if year >= years[-1]:
        return profile[years[-1]]
    for i in range(len(years) - 1):
        ya, yb = years[i], years[i + 1]
        if ya <= year <= yb:
            frac = (year - ya) / (yb - ya)
            return profile[ya] + frac * (profile[yb] - profile[ya])
    return profile[years[-1]]


h2_demand = demand_for_year(scenario, year)

# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print(f'  scenario          = {scenario!r}')
    print(f'  year              = {year}')
    print(f'  h2_demand         = {h2_demand:,.0f} kt/yr  ({h2_demand/1000:.2f} Mt/yr)')
    print(f'  elec_type         = {elec_type!r}')
    print(f'  heat_source       = {heat_source!r}')
    print(f'  centralised       = {centralised}')
    print(f'  pipeline          = {pipeline}')
    print(f'  max_pipeline_dist = {max_pipeline_dist} km')
    print(f'  end_tuple         = {end_tuple}  (main.py only)')
