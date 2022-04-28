#  -*- coding: utf-8 -*-

"""
    rhizodep.parameters
    ~~~~~~~~~~~~~~~~~~~~~~

    The module :mod:`rhizodep.parameters` defines the constant parameters.


"""

random_choice = 8

# ArchiSimple parameters for root growth:
# ---------------------------------------

# Initial tip diameter of the primary root (in m):
D_ini = 0.80 / 1000.
# => Reference: Di=0.5 mm

# Proportionality coefficient between the tip diameter of an adventitious root and D_ini (dimensionless):
D_adv_to_D_ini_ratio = 0.8

# Threshold tip diameter, below which there is no elongation (i.e. the diameter of the finest observable roots)(in m):
Dmin = 0.35 / 1000.
# => Reference: Dmin=0.05 mm

# Slope of the potential elongation rate versus tip diameter (in m m-1 s-1):
EL = 1.39 * 20 / (60. * 60. * 24.)
# => Reference: EL = 5 mm mm-1 day-1

# Proportionality coefficient between the section area of the segment and the sum of distal section areas
# (dimensionless, 0 = no radial growth):
SGC = 0.2

# Average ratio of the diameter of the daughter root to that of the mother root (dimensionless):
RMD = 0.638

# Relative variation of the daughter root diameter (dimensionless):
CVDD = 0.2

# Delay of emergence of the primordium (in s):
emergence_delay = 158 / 20. * (60. * 60. * 24.)
# => Reference: emergence_delay = 3 days

# Inter-primordium distance (in m):
IPD = 5.11 / 1000.
# => Reference: IPD = 7.6 mm

# Maximal number of roots emerging from the base (including primary and seminal roots)(dimensionless):
n_adventitious_roots = 0

# Emission rate of adventitious roots (in s-1):
ER = 1.0 / (60. * 60. * 24.)
# => Reference: ER = 0.5 day-1

# Coefficient of growth duration (in s m-2):
GDs = 800 * (60. * 60. * 24.) * 1000. ** 2. * 100. # We introduce here *100 as it was before in the main code of model.py prior to March 2022!
# => Reference: GDs=400. day mm-2

# Coefficient of the life duration (in s m g-1 m3):
LDs = 4000. * (60. * 60. * 24.) * 1000 * 1e-6
# => Reference: LDs = 5000 day mm-1 g-1 cm3

# Root tissue density (in g m-3):
root_tissue_density = 0.10 * 1e6
# => Reference: RTD=0.1 g cm-3

# C content of struct_mass (mol of C per g of struct_mass):
struct_mass_C_content = 0.44 / 12.01
# => We assume that the structural mass contains 44% of C.

# Gravitropism (dimensionless):
gravitropism_coefficient = 0.06
# tropism_intensity = 2e6 # Value between 0 and 1.
# tropism_direction = (0,0,-1) # Force of the tropism

# Length of a segment (in m):
segment_length = 3. / 1000.

# Parameters for temperature adjustments:
# ---------------------------------------
# Reference temperature for growth, i.e. at which the relative effect of temperature is 1 (in degree Celsius):
T_ref_growth = 20.
# => We assume that the parameters for root growth were obtained for T = 20 degree Celsius

# Proportionality coefficient between growth and temperature (in unit of growth per degree Celsius)
# - note that, when this parameter is equal to 0, temperature will have no influence on growth:
growth_increase_with_temperature = 1 / T_ref_growth
# => We suggest to calculate this parameter so that relative growth is 0 at 0 degree Celsius and 1 at reference temperature.

# Parameters for growth respiration:
# -----------------------------------
# Growth yield (in mol of CO2 per mol of C used for struct_mass):
yield_growth = 0.8
# => We use the range value (0.75-0.85) proposed by Thornley and Cannell (2000)

# Parameters for maintenance respiration:
# ----------------------------------------
# Maximal maintenance respiration rate (in mol of CO2 per g of struct_mass per s):
resp_maintenance_max = 4.1e-6 / 6. * (0.44 / 60. / 14.01) * 5
# => According to Barillot et al. (2016): max. residual maintenance respiration rate is 4.1e-6 umol_C umol_N-1 s-1,
# i.e. 4.1e-6/60*0.44 mol_C g-1 s-1 assuming an average struct_C:N_tot molar ratio of root of 60 [cf simulations by CN-Wheat 47 in 2020]
# and a C content of struct_mass of 44%. And according to the same simulations, total maintenance respiration is about
# 5 times more than residual maintenance respiration.

# Affinity constant for maintenance respiration (in mol of hexose per g of struct_mass):
Km_maintenance = 1.67e3 * 1e-6 / 6.
# => According to Barillot et al. (2016): Km=1670 umol of C per gram for residual maintenance respiration (based on sucrose concentration!)

# Expected concentrations in a typical root segment (used for estimating a few parameters):
# ------------------------------------------------------------------------------------------
# Expected sucrose concentration in root (in mol of sucrose per g of root):
expected_C_sucrose_root = 0.01 / 12.01 / 12
# => 0.0025 is a plausible value according to the results of Gauthier (2019, pers. communication), but here, we use
# a plausible sucrose concentration (10 mgC g-1) in roots according to various experimental results.

# Expected hexose concentration in root (in mol of hexose per g of root):
expected_C_hexose_root = 0.01 / 12.01 / 6
# => Here, we use a plausible hexose concentration in roots (10 mgC g-1) according to various experimental results.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
expected_C_hexose_root = 1e-3

# Expected hexose concentration in soil (in mol of hexose per g of root):
expected_C_hexose_soil = expected_C_hexose_root / 100.
# => We expect the soil concentration to be 2 orders of magnitude lower than the root concentration.

# Expected hexose concentration in the reserve pool (in mol of hexose per g of reserve pool):
expected_C_hexose_reserve = expected_C_hexose_root * 2.
# => We expect the reserve pool to be two times higher than the mobile one.

# Parameters for estimation the surfaces of exchange between root symplasm, apoplasm and  phloem:
# ------------------------------------------------------------------------------------------------

# Phloem surface as a fraction of the root external surface (in m2 of phloem surface per m2 of root external surface):
phloem_surfacic_fraction = 1.

# Symplasm surface as a fraction of the root external surface (in m2 of symplasm surface per m2 of root external surface):
symplasm_surfacic_fraction = 10.

# Parameters for phloem unloading/reloading:
# ----------------------------------
# Maximum unloading rate of sucrose from the phloem through the surface of the phloem vessels (in mol of sucrose per m2 per second):
surfacic_unloading_rate_reference = 0.03 / 12 * 1e-6 / (0.5 * phloem_surfacic_fraction)
# => According to Barillot et al. (2016b), this value is 0.03 umol C g-1 s-1, and we assume that 1 gram of dry root mass
# is equivalent to 0.5 m2 of surface. However, the unloading of sucrose is not done exactly the same in CN-Wheat as in RhizoDep (
# for example, exudation is a fraction of the sucrose unloading, unlike in RhizoDep where it is a fraction of hexose).
# CHEATING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
surfacic_unloading_rate_reference = surfacic_unloading_rate_reference * 200.

# Maximum unloading rate of sucrose from the phloem through the section of a primordium (in mol of sucrose per m2 per second):
surfacic_unloading_rate_primordium = surfacic_unloading_rate_reference * 3.
# => We expect the maximum unloading rate through an emerging primordium to be xxx times higher than the usual unloading rate

# ALTERNATIVE: We use a permeability coefficient and unloading occurs through diffusion only !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Coefficient of permeability of unloading phloem (in gram per m2 per second):
phloem_permeability = 2e-4
# => Quite arbitrary value...

# gamma_unloading = 0.3
# # Arbitrary

# Maximum reloading rate of hexose inside the phloem vessels (in mol of hexose per m2 per second):
surfacic_loading_rate_reference = surfacic_unloading_rate_reference * 2.
# => We expect the maximum loading rate of hexose to be equivalent to the maximum unloading rate.

# Affinity constant for sucrose unloading (in mol of sucrose per g of struct_mass):
Km_unloading = 1000 * 1e-6 / 12.
# => According to Barillot et al. (2016b), this value is 1000 umol C g-1

# Affinity constant for sucrose loading (in mol of hexose per g of struct_mass):
Km_loading = Km_unloading * 2.
# => We expect the Km of loading to be equivalent as the Km of unloading, as it may correspond to the same transporter.

gamma_loading = 0.0

# Parameters for mobilization/immobilization in the reserve pool:
# ----------------------------------------------------------------

# Maximum concentration of hexose in the reserve pool (in mol of hexose per gram of structural mass):
C_hexose_reserve_max = 5e-3
# Minimum concentration of hexose in the reserve pool (in mol of hexose per gram of structural mass):
C_hexose_reserve_min = 0.
# Minimum concentration of hexose in the mobile pool for immobilization (in mol of hexose per gram of structural mass):
C_hexose_root_min_for_reserve = 5e-3

# Maximum immobilization rate of hexose in the reserve pool (in mol of hexose per gram of structural mass per second):
max_immobilization_rate = 8 * 1e-6 / 6.
# => According to Gauthier et al. (2020): the new maximum rate of starch synthesis for vegetative growth in the shoot is 8 umolC g-1 s-1

# Affinity constant for hexose immobilization in the reserve pool (in mol of hexose per gram of structural mass):
Km_immobilization = expected_C_hexose_root * 1.

# Maximum mobilization rate of hexose from the reserve pool (in mol of hexose per gram of structural mass per second):
max_mobilization_rate = max_immobilization_rate

# Affinity constant for hexose remobilization from the reserve (in mol of hexose per gram of structural mass):
Km_mobilization = expected_C_hexose_reserve * 5.

# Parameters for root hexose exudation:
# --------------------------------------
# Expected exudation rate (in mol of hexose per m2 per s):
expected_exudation_efflux = 608 * 0.000001 / 12.01 / 6 / 3600 * 1 / (0.5 * 10)

# => According to Jones and Darrah (1992): the net efflux of C for a root of maize is 608 ug C g-1 root DW h-1,
# and we assume that 1 gram of dry root mass is equivalent to 0.5 m2 of external surface.
# OR:
# expected_exudation_efflux = 5.2 / 12.01 / 6. * 1e-6 * 100. ** 2. / 3600.
# => Explanation: According to Personeni et al. (2007), we expect a flux of 5.2 ugC per cm2 per hour

# Permeability coefficient (in g of struct_mass per m2 per s):
Pmax_apex = expected_exudation_efflux / (expected_C_hexose_root - expected_C_hexose_soil)
# => We calculate the permeability according to the expected exudation flux and expected concentration gradient between cytosol and soil.

# Coefficient affecting the decrease of permeability with distance from the apex (dimensionless):
gamma_exudation = 0.4
# => According to Personeni et al (2007), this gamma coefficient showing the decrease in permeability along the root is 0.41-0.44

# Parameters for root hexose uptake from soil:
# ---------------------------------------------
# Maximum rate of influx of hexose from soil to roots (in mol of hexose per m2 per s):
# uptake_rate_max = 0.3/3600.*3*0.000001/12.01/6.*10000
# => According to Personeni et al (2007), the uptake coefficient beta used in the relationship Uptake = beta*S*Ce should be
# 0.20-0.38 cm h-1 with concentrations in the external solution of about 3 ugC cm-3.
# OR:
uptake_rate_max = 277 * 0.000000001 / (60 * 60 * 24) * 1000 * 1 / (0.5 * symplasm_surfacic_fraction)
# => According to Jones and Darrah (1996), the uptake rate measured for all sugars tested with an individual external concentration
# of 100 uM is equivalent to 277 nmol hexose mg-1 day-1, and we assume that 1 gram of dry root mass is equivalent to 0.5 m2 of external surface.

# Affinity constant for hexose uptake (in mol of hexose per g of struct_mass):
Km_uptake = expected_C_hexose_soil * 10
# => We assume that half of the max influx is reached when the soil concentration equals 10 times the expected root concentration
# CHEATING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Km_uptake = Km_loading

# Parameters for soil hexose degradation:
# ----------------------------------------
# Maximum degradation rate of hexose in soil (in mol of hexose per m2 per s):
degradation_rate_max = uptake_rate_max * 10.
# => We assume that the maximum degradation rate is 10 times higher than the maximum uptake rate

# Affinity constant for soil hexose degradation (in mol of hexose per g of struct_mass):
Km_degradation = Km_uptake / 2.
# => According to what Jones and Darrah (1996) suggested, we assume that this Km is 2 times lower than the Km corresponding
# to root uptake of hexose (350 uM against 800 uM in the original article).

# Parameters for C-controlled growth rates:
# -----------------------------------------

# Proportionality factor between the radius and the length of the root apical zone in which C can be used to sustain root elongation:
growing_zone_factor = 8  # m m-1
# => According to illustrations by by Kozlova et al. (2020), the length of the growing zone corresponding to the root cap,
# meristem and elongation zones is about 8 times the diameter of the tip.

# Affinity constant for root elongation (in mol of hexose per g of struct_mass):
Km_elongation = 1250 * 1e-6 / 6.
# => According to Barillot et al. (2016b): Km for root growth is 1250 umol C g-1 for sucrose
# => According to Gauthier et al (2020): Km for regulation of the RER by sucrose concentration in hz = 100-150 umol C g-1
# CHEATING:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Km_elongation = Km_elongation*10
# Km_elongation = Km_loading

# Affinity constant for root thickening (in mol of hexose per g of struct_mass):
Km_thickening = Km_elongation
# => We assume that the Michaelis-Menten constant for thickening is the same as for root elongation.
# Maximal rate of relative increase in root radius (in m per m per s):
relative_root_thickening_rate_max = 5. / 100. / (24. * 60. * 60.)
# => We consider that the radius can't increase by more than 5% every day

# We define a probability (between 0 and 1) of nodule formation for each apex that elongates:
nodule_formation_probability = 0.5
# Maximal radius of nodule (in m):
nodule_max_radius = D_ini * 20.
# Affinity constant for nodule thickening (in mol of hexose per g of struct_mass):
Km_nodule_thickening = Km_elongation * 100
# Maximal rate of relative increase in nodule radius (in m per m per s):
relative_nodule_thickening_rate_max = 20. / 100. / (24. * 60. * 60.)


# => We consider that the radius can't increase by more than 20% every day
