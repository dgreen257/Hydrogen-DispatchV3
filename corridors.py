"""
corridors.py
------------
Defines the 5 European Hydrogen Backbone (EHB) import corridor configurations
used in the multi-corridor model run (run_corridors.py).

Each corridor represents a distinct geographical pathway for delivering green H2
into the EU, as described in the EHB framework for 10 Mt/yr by 2030.

Corridor overview
-----------------
  A  North Africa Pipeline via Italy
     Piped green H2 from Algeria/Tunisia through the Mediterranean into Italy
     and Central Europe.

  B  Iberian Gateway (SW Europe & N. Africa)
     Pipeline from Morocco/Algeria + shipping from West Africa/Latin America,
     landing in Spain/Portugal and flowing north into France/Germany.

  C  North Sea Hub (Global Seaborne)
     Seaborne imports (NH3, LOHC, LH2) from anywhere in the world landing at
     major decarb. port clusters: Rotterdam, Antwerp, Zeebrugge, Wilhelmshaven.
     Pipeline disabled — this corridor is defined by its shipping nature.
     EU member states are excluded at runtime (non-EU imports only).

  D  Nordic & Baltic (Offshore Wind)
     Piped H2 generated from large offshore wind projects in Norway, UK and
     Iceland flowing south into Hamburg.  EU members (DNK, FIN, SWE) excluded.

  E  SE Europe & Ukraine Gateway
     High solar/wind potential in Ukraine, Turkey and the Caucasus piped
     westward into Austria/Central Europe.  EU members (ROU, GRC, BGR) excluded.

Usage
-----
    from corridors import CORRIDORS, EU_MEMBER_ISOS
    cfg = CORRIDORS['C']   # Rotterdam seaborne corridor
    print(cfg['destination'], cfg['source_isos'])
"""

# ---------------------------------------------------------------------------
# EU-27 member states (ISO_A3).  These are excluded from all import corridors
# because the model covers non-EU imports only; domestic EU supply is handled
# separately.
# ---------------------------------------------------------------------------
EU_MEMBER_ISOS: frozenset[str] = frozenset({
    'AUT', 'BEL', 'BGR', 'HRV', 'CYP', 'CZE', 'DNK', 'EST',
    'FIN', 'FRA', 'DEU', 'GRC', 'HUN', 'IRL', 'ITA', 'LVA',
    'LTU', 'LUX', 'MLT', 'NLD', 'POL', 'PRT', 'ROU', 'SVK',
    'SVN', 'ESP', 'SWE',
})

# Country name equivalents — used as a fallback when ISO_A3 is '-99' or '---'
# (Natural Earth 110m sometimes assigns -99 to coastal/border points).
EU_MEMBER_NAMES: frozenset[str] = frozenset({
    'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czechia',
    'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece',
    'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg',
    'Malta', 'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia',
    'Slovenia', 'Spain', 'Sweden',
})

CORRIDORS = {
    'A': {
        'name':              'Corridor A',
        'gateway':           'Italy',
        'subtitle':          'North Africa & Middle East via Italy',
        'description':       ('Piped H\u2082 from Algeria/Tunisia through the Mediterranean '
                              'into Italy and Central Europe; plus Middle East and West African '
                              'sources shipping NH\u2083/LOHC/LH\u2082 to Italian ports.'),
        'destination':       (41.90, 12.50),     # Central Italy (Rome area) — EU entry point
        'source_isos':       [
            'DZA', 'TUN', 'MAR', 'LBY', 'EGY', 'MRT',   # North Africa — pipeline viable
            'SAU', 'OMN', 'YEM', 'IRQ', 'IRN', 'JOR',   # Middle East — shipping to Italy
            'ARE', 'QAT', 'KWT',                          # Gulf states — shipping
            'SEN', 'NGA', 'GHA', 'CIV',                  # West Africa — shipping
        ],
        'pipeline':          True,
        'max_pipeline_dist': 4000,               # pipeline for close sources; farther ones route via ship
        'centralised':       True,
        'colour':            '#E63946',          # red
    },

    'B': {
        'name':              'Corridor B',
        'gateway':           'Spain',
        'subtitle':          'Iberian Gateway (SW Europe, Africa & Americas)',
        'description':       ('Pipeline from Morocco/Algeria + shipping from West/East Africa, '
                              'Latin America and the Middle East, landing in Spain/Portugal.'),
        'destination':       (37.26, -6.94),     # Huelva, Spain (major emerging H2 hub)
        'source_isos':       [
            'DZA', 'MAR', 'TUN', 'MRT',                         # N. Africa — pipeline viable
            'SEN', 'NGA', 'ZAF', 'NAM', 'AGO',                  # West/Southern Africa — shipping
            'GHA', 'CIV', 'CMR', 'MOZ', 'TZA', 'ETH',          # Expanded Africa — shipping
            'CHL', 'ARG', 'BRA', 'PER', 'MEX',                  # Latin America — shipping
            'COL', 'URY', 'ECU',                                 # Expanded Latin America
            'SAU', 'OMN', 'QAT', 'ARE',                         # Middle East — shipping to Spain
        ],
        'pipeline':          True,
        'max_pipeline_dist': 2500,
        'centralised':       True,
        'colour':            '#F4A261',          # orange
    },

    'C': {
        'name':              'Corridor C',
        'gateway':           'Netherlands',
        'subtitle':          'North Sea Hub (Global Seaborne)',
        'description':       ('Global seaborne imports (NH\u2083/LOHC/LH\u2082) arriving at '
                              'Rotterdam, Antwerp, Zeebrugge, Wilhelmshaven.'),
        'destination':       (51.92, 4.48),      # Rotterdam
        'source_isos':       None,               # Truly global — no geographic filter
        'pipeline':          False,              # Strictly shipping; no pipelines
        'max_pipeline_dist': 2000,               # Irrelevant when pipeline=False
        'centralised':       True,
        'colour':            '#2196F3',          # blue
    },

    'D': {
        'name':              'Corridor D',
        'gateway':           'Germany',
        'subtitle':          'Nordic, Baltic & Atlantic (Offshore Wind)',
        'description':       ('Offshore wind H\u2082 from Norway, UK and Iceland piped '
                              'south into Hamburg; plus Canada and Greenland shipping '
                              'NH\u2083/LH\u2082 across the North Atlantic (non-EU imports only).'),
        'destination':       (53.55, 9.99),      # Hamburg (Brunsbüttel pipeline gateway)
        'source_isos':       ['NOR', 'GBR', 'ISL', 'CAN', 'GRL'],   # DNK/FIN/SWE removed — EU members
        # Fallback country names for entries where ISO_A3 = '-99' in renewables.csv
        # (Natural Earth assigns -99 to Norway; Iceland has no grid points)
        'source_countries':  ['Norway', 'Iceland', 'Greenland'],     # Finland removed — EU member
        'pipeline':          True,
        'max_pipeline_dist': 3000,               # pipeline for NOR/GBR; CAN/GRL will route via ship
        'centralised':       True,
        'colour':            '#4CAF50',          # green
    },

    'E': {
        'name':              'Corridor E',
        'gateway':           'Austria',
        'subtitle':          'SE Europe, Ukraine & Middle East Gateway',
        'description':       ('High solar/wind potential in Ukraine, Turkey and the Caucasus '
                              'piped westward into Austria; plus Middle East sources via '
                              'Turkish pipeline corridor or Black Sea shipping.'),
        'destination':       (48.21, 16.37),     # Vienna — Central European hub
        'source_isos':       [
            'UKR', 'TUR', 'MDA',                                   # SE Europe & Black Sea (ROU/GRC/BGR removed — EU members)
            'AZE', 'GEO', 'KAZ', 'UZB', 'TKM',                    # Caucasus & Central Asia
            'IRQ', 'IRN', 'SAU', 'JOR', 'SYR', 'OMN', 'ARE',      # Middle East — pipeline via Turkey or shipping
            'MNG',                                                   # Mongolia — Central Asian wind
        ],
        'pipeline':          True,
        'max_pipeline_dist': 3000,               # pipeline for close sources; farther ones route via ship
        'centralised':       True,
        'colour':            '#9C27B0',          # purple
    },

    'EU': {
        'name':              'EU Domestic',
        'gateway':           'EU-27',
        'subtitle':          'EU Domestic Production',
        'description':       ('Green H\u2082 produced within EU-27 member states. '
                              'No transport cost — generation cost only.'),
        'destination':       None,               # No transport destination
        'source_isos':       list(EU_MEMBER_ISOS),
        'source_countries':  list(EU_MEMBER_NAMES),
        'pipeline':          False,
        'max_pipeline_dist': 0,
        'centralised':       False,
        'colour':            '#009688',          # teal
        'is_domestic':       True,               # Skip EU exclusion and transport costs
    },
}
