"""
The parameter sets the port is compared with the legacy on (devplan_metafspm.md, Q-R2).

LEGACY_DEFAULTS   the legacy's own defaults (an empty scenario)
GB_SOIL_1_0       the growth component's parameters in Wheat-BRIDGES' scenario table, column GB_soil_1.0:
                  Wheat-BRIDGES/simulations/hydro/inputs/Scenarios_26-10-04.xlsx, sheet scenarios_as_columns
                  (read 2026-10-23; the table's 'True' / 'False' read as booleans, as MakeScenarios.from_table does)

The variants switch the legacy's flags on either set (simple_growth_duration; the nodules stay off, their draw being
unseeded, devplan §5).
"""
LEGACY_DEFAULTS = {}

GB_SOIL_1_0 = {
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

PARAMETER_SETS = {
    "legacy_defaults": LEGACY_DEFAULTS,
    "GB_soil_1.0": GB_SOIL_1_0,
}


def variants():
    """{name: parameters}: each set, with simple_growth_duration off and on."""
    out = {}
    for name, parameters in PARAMETER_SETS.items():
        for flag in (False, True):
            out[f"{name}/simple_growth_duration={flag}"] = dict(parameters, simple_growth_duration=flag,
                                                                 nodules=False)
    return out
