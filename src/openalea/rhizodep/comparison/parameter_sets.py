"""
Parameter sets of the root growth components (RootArchitecture, RootGrowth, RootGrowthCN), as scenarios' "parameters".

WHEAT_SOISSON   winter wheat (cv. Soissons), the growth component's parameters in Wheat-BRIDGES' scenario table,
                column GB_soil_1.0: Wheat-BRIDGES/simulations/hydro/inputs/Scenarios_26-10-04.xlsx, sheet
                scenarios_as_columns (read 2026-10-23; the table's 'True' / 'False' read as booleans, as
                MakeScenarios.from_table does)
PEA, MAIZE, PEACH
                ArchiSimple's calibrations (Pagès et al., 2014, Ecological Modelling, Table 3: Pisum sativum, Zea mays,
                Prunus persica; values in parentheses there are temporary ones), for RootArchitecture (C-free) or
                RootGrowth. Converted to the model's units: lengths in m; EL from mm mm-1 day-1 (÷ 86400); GDs from
                day mm-2 (× 86400 × 1e6); the tissue density from g cm-3 (× 1e6 g m-3); ER from day-1 (÷ 86400);
                LDs from day mm mg-1 (× 86400), the life duration ∝ D² as in the paper (life_duration_diameter_exponent
                2). MNP, the number of primary roots, as one primary (seminal) root and nodal (adventitious) roots
                emitted at ER from the start (MAIZE: 5 seminal roots, nodal ones from day 5, an assumption). ArchiSimple
                has no extender on the main roots' growth durations (main_roots_growth_extender 1) and its durations
                are GDs·d² (growth_duration_rule "simple"). Not in the model: the tropism G, the soil's effects. Durations
                are in thermal time at the temperature factor of the soil temperature.
                Oilseed rape and other legumes: no root architecture calibration in rhizodep, Wheat-BRIDGES or colzette
                (whose parameters are its shoots'); Pagès et al. (2014) has none for rapeseed.

LEGACY_DEFAULTS the legacy's own defaults (an empty scenario)

The variants (the comparisons with the legacy) switch the legacy's flags on the legacy's sets (simple_growth_duration;
the nodules stay off).
"""
DAY = 86400.

LEGACY_DEFAULTS = {}

WHEAT_SOISSON = {
    'root_tissue_density': 140000,
    'D_ini': 0.0007,
    'root_hair_radius': 6e-06,
    'root_hairs_lifespan': 165600,
    'root_hairs_density': 375000000,
    'root_hair_max_length': 0.001,
    'root_hairs_elongation_rate': 0.0037037037037037,
    'LDs': 50000,
    'ER': 1.38888888888889e-05,
    'n_seminal_roots': 5,
    'n_adventitious_roots': 50,
    'random_choice': 1,
    'D_sem_to_D_ini_ratio': 1,
    'CVDD': 0.16,
    'starting_time_for_adventitious_roots_emergence': 1453890.89418848,
    'D_adv_to_D_ini_ratio': 0.95,
    'growing_zone_factor': 21,
    'emergence_delay': 282528,
    'EL': 0.000650115740740741,
    'Km_elongation': 0.000104166666666667,
    'SGC': 0,
    'relative_root_thickening_rate_max': 5.79e-07,
    'C_hexose_min_for_thickening': 0,
    'Km_thickening': 0.000208333333333333,
    'struct_mass_C_content': 0.036636137,
    'yield_growth': 0.8,
    'segment_length': 0.0045,
    'Dmin': 0.000122,
    'RMD': 0.57,
    'IPD': 0.00474,
    'GDs': 745476480000,
    'main_roots_growth_extender': 100,
    'GD_highest': 5184000,
    'GD_high': 518400,
    'GD_medium': 60480,
    'GD_low': 21600,
    'GD_by_frequency': True,
    'GD_prob_low': 0.5,
    'GD_prob_medium': 0.85,
    'random': True,
    'initial_segment_length': 0.001,
    'initial_apex_length': 0,
    'initial_C_hexose_root': 0.0001,
    'forcing_seminal_roots_events': False,
    'forcing_adventitious_roots_events': False,
    'radial_growth': False,
    'root_order_limitation': True,
    'root_order_treshold': 3,
}



def archisimple(Di, EL, Dmin, GDs, IPD, RDM, CVDD, SGC, LDs, RTD, n_seminal_roots=1, n_adventitious_roots=0, ER=None,
                starting_time_for_adventitious_roots_emergence=0., emergence_delay=5.):
    """A scenario from ArchiSimple's parameters in Pagès et al.'s (2014) units: Di, Dmin, IPD in mm; EL in mm mm-1
    day-1; GDs in day mm-2; LDs in day mm mg-1; RTD in g cm-3; ER in day-1; the delays in days."""
    scenario = dict(D_ini=Di * 1e-3, EL=EL / DAY, Dmin=Dmin * 1e-3, GDs=GDs * DAY * 1e6, IPD=IPD * 1e-3, RMD=RDM,
                    CVDD=CVDD, SGC=SGC, LDs=LDs * DAY, root_tissue_density=RTD * 1e6, new_root_tissue_density=RTD * 1e6,
                    life_duration_diameter_exponent=2., main_roots_growth_extender=1, growth_duration_rule="simple",
                    n_seminal_roots=n_seminal_roots, n_adventitious_roots=n_adventitious_roots,
                    D_sem_to_D_ini_ratio=1., D_adv_to_D_ini_ratio=1., radial_growth=SGC > 0., random=True,
                    starting_time_for_adventitious_roots_emergence=starting_time_for_adventitious_roots_emergence * DAY,
                    emergence_delay=emergence_delay * DAY)
    if ER is not None:
        scenario["ER"] = ER / DAY
    return scenario


PEA = archisimple(Di=1.1, EL=18, Dmin=0.19, GDs=120, IPD=7.8, RDM=0.50, CVDD=0.18, SGC=0.9, LDs=3000, RTD=0.05)
MAIZE = archisimple(Di=1.2, EL=51, Dmin=0.14, GDs=50, IPD=2.0, RDM=0.12, CVDD=0.30, SGC=0.0, LDs=3000, RTD=0.08,
                    n_seminal_roots=5, n_adventitious_roots=35, ER=0.8,
                    starting_time_for_adventitious_roots_emergence=5.)
PEACH = archisimple(Di=1.2, EL=14, Dmin=0.15, GDs=150, IPD=3.1, RDM=0.27, CVDD=0.27, SGC=0.9, LDs=3000, RTD=0.15,
                    n_seminal_roots=1, n_adventitious_roots=9, ER=0.4)
SPECIES = {"wheat (Soissons)": WHEAT_SOISSON, "pea": PEA, "maize": MAIZE, "peach": PEACH}

PARAMETER_SETS = {
    "legacy_defaults": LEGACY_DEFAULTS,
    "wheat_soisson": WHEAT_SOISSON,
}


def variants():
    """{name: parameters}: each set, with simple_growth_duration off and on."""
    out = {}
    for name, parameters in PARAMETER_SETS.items():
        for flag in (False, True):
            out[f"{name}/simple_growth_duration={flag}"] = dict(parameters, simple_growth_duration=flag,
                                                                 nodules=False)
    return out
