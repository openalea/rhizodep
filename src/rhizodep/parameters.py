#  -*- coding: utf-8 -*-

"""
    rhizodep.parameters
    ~~~~~~~~~~~~~~~~~~~~~~

    The module :mod:`rhizodep.parameters` defines the constant parameters of the model.

"""

from math import pi

# We set the random seed, so that the same simulation can be repeted with the same seed:
random_choice = 8

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# ArchiSimple parameters for root growth:
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Initial tip diameter of the primary root (in meters):
#------------------------------------------------------
D_ini = 0.80 / 1000.
# => Reference: Di=0.8 mm

# Initial external surface of the collar element (in square meters)
external_surface = 1e-6

# Initial volume of the collar element (in cube meters)
volume = 1e-7

# Proportionality coefficient between the tip diameter of a seminal root and D_ini (dimensionless):
#--------------------------------------------------------------------------------------------------
D_sem_to_D_ini_ratio = 0.95

# Proportionality coefficient between the tip diameter of an adventitious root and D_ini (dimensionless):
#--------------------------------------------------------------------------------------------------------
D_adv_to_D_ini_ratio = 0.8

# Minimal threshold tip diameter (i.e. the diameter of the finest observable roots)(in meters):
#----------------------------------------------------------------------------------------------
Dmin = 0.1 / 1000.
# => Reference: Dmin=0.05 mm

# Slope of the elongation rate = f(tip diameter) (in meters of root per meter of radius per second equivalent to T_ref_growth):
#-----------------------------------------------------------------------------------------------------------------------------
EL = 1.39 * 20 / (60. * 60. * 24.)
# => Reference: EL = 5 mm mm-1 day-1

# Proportionality coefficient between the section area of the segment and the sum of distal section areas
#--------------------------------------------------------------------------------------------------------
# (dimensionless, 0 = no radial growth):
SGC = 0.0

# Average ratio of the diameter of the daughter root to that of the mother root (dimensionless):
#-----------------------------------------------------------------------------------------------
RMD = 0.3

# Relative variation of the daughter root diameter (dimensionless):
#------------------------------------------------------------------
CVDD = 0.2

# Delay of emergence of the primordium (in second equivalent to temperature T_ref_growth):
#-----------------------------------------------------------------------------------------
emergence_delay = 3. * (60. * 60. * 24.)
# => Reference: emergence_delay = 3 days

# Inter-primordia distance (in meters of root):
#----------------------------------------------
IPD = 5. / 1000.
# => Reference: IPD = 7.6 mm

# Maximal number of roots emerging from the base (including primary and seminal roots)(dimensionless):
#-----------------------------------------------------------------------------------------------------
n_seminal_roots = 5

# Maximal number of roots emerging from the base (including primary and seminal roots)(dimensionless):
#-----------------------------------------------------------------------------------------------------
n_adventitious_roots = 10

# Time when adventitious roots start to successively emerge (in second equivalent to temperature T_ref_growth):
#--------------------------------------------------------------------------------------------------------------
starting_time_for_adventitious_roots_emergence = (60. * 60. * 24.) * 9.

# Emission rate of adventitious roots (per second equivalent to temperature T_ref_growth):
#-----------------------------------------------------------------------------------------
ER = 0.2 / (60. * 60. * 24.)
# WATCH OUT: If the file 'adventitious_roots_inputs.csv' exist in the proper directory, information regarding
# the emergence of adventitious roots will actually be read from this file.

# Coefficient of growth duration (in second equivalent to temperature T_ref_growth per square meter of root radius):
#-------------------------------------------------------------------------------------------------------------------
GDs = 800 * (60. * 60. * 24.) * 1000. ** 2.
# => Reference: GDs=400. day mm-2
# NEW: Coefficient of extension of growth duration for specific roots (in second per second):
main_roots_growth_extender = 100
# => We assume that for some roots (e.g. seminal and adventious roots) there is an extender of growth compared to
# what is predicted by their diameter.

# Selective growth durations (in second equivalent to temperature T_ref_growth):
#-------------------------------------------------------------------------------
# As an alternative to using a single value of growth duration depending on diameter, we offer the possibility to rather
# define the growth duration as a random choice between three values (low, medium and high), depending on their respective
# probability (between 0 and 1):
GD_by_frequency = False # If True, then four possible values of growth duration can be used!
# The growth duration has a probability of [GD_prob_low] to equal GD_low:
GD_low = 0.25 * (60. * 60. * 24.) # Estimated from the shortest lateral wheat roots observed in rhizoboxes (Rees et al., unpublished)
GD_prob_low = 0.50 # Estimated from the shortest lateral wheat roots observed in rhizoboxes (Rees et al., unpublished)
# The growth duration has a probability of [GD_prob_medium - GD_prob_low] to equal GD_medium:
GD_medium = 0.70 * (60. * 60. * 24.) # Estimated from the medium lateral wheat roots observed in rhizoboxes (Rees et al., unpublished)
GD_prob_medium = 0.85 # Estimated from the medium lateral wheat roots observed in rhizoboxes (Rees et al., unpublished)
# The growth duration has a probability of [1-GD_prob_medium] to equal GD_high:
GD_high = 6 * (60. * 60. * 24.) # Estimated the longest observed lateral wheat roots observed in rhizoboxes (Rees et al., unpublished)
# For seminal and adventitious roots, a longer growth duration is applied:
GD_highest = 60 * (60. * 60. * 24.) # Expected growth duration of a seminal root

# Coefficient of the life duration (in second equivalent to temperature T_ref_growth x meter-1 x cubic meter per gram):
#--------------------------------------------------------------------------------------------------------------------
LDs = 4000. * (60. * 60. * 24.) * 1000 * 1e-6
# => Reference: LDs = 5000 day mm-1 g-1 cm3

# Root tissue density (in gram of structural mass per cubic meter):
#------------------------------------------------------------------
root_tissue_density = 0.10 * 1e6
# => Reference: RTD=0.1 g cm-3

# C content of structural mass (mol of C per gram of structural mass):
#-----------------------------------------------------------------
struct_mass_C_content = 0.44 / 12.01
# => We assume that the structural mass contains 44% of C.

# Gravitropism (dimensionless):
#------------------------------
gravitropism_coefficient = 0.06

# Length of a segment (in meters of root):
#-----------------------------------------
segment_length = 3. / 1000.

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Parameters for growth temperature adjustments:
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# T_ref_growth has been defined as 20 degree Celsius ; the parameters below therefore set that any temperaure-dependent
# process will be multiplied by 1 at 20 degrees, by 0 at 0 degree, and by a value higher than 1 for temperatures higher
# than 20 degrees.
T_ref_growth = 20
relative_process_value_at_T_ref_growth = 1

# Temperature dependence for root growth:
#""""""""""""""""""""""""""""""""""""""""
root_growth_T_ref = 0
root_growth_A = relative_process_value_at_T_ref_growth / T_ref_growth
root_growth_B = 0
root_growth_C = 0
# => We assume that relative growth is 0 at T_ref=0 degree Celsius, and linearily increases to reach 1 at 20 degree.

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Parameters for root hairs dynamics:
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Average radius of root hair (in meters):
#-----------------------------------------
root_hair_radius = 12 * 1e-6 /2.
# => According to the work of Gahoonia et al. (1997), the root hair diameter is relatively constant for different
# genotypes of wheat and barley, i.e. 12 microns.

# Average maximal length of a root hair (in meters):
#---------------------------------------------------
root_hair_max_length = 1 * 1e-3
# => According to the work of Gahoonia et al. (1997), the root hair maximal length for wheat and barley evolves between
# 0.5 and 1.3 mm.

# Average density of root hairs (number of hairs par meter of root per meter of root radius):
#--------------------------------------------------------------------------------------------
root_hairs_density = 30 * 1e3 / (0.16 / 2. * 1e-3)
# => According to the work of Gahoonia et al. (1997), the root hair density is about 30 hairs per mm for winter wheat,
# for a root radius of about 0.16 mm.

# Average elongation rate of root hairs (in meter per second per meter of root radius):
#--------------------------------------------------------------------------------------
root_hairs_elongation_rate = 0.080 * 1e-3 / (60. * 60.) / root_hair_radius
# => According to the data from McElgunn and Harrison (1969), the elongation rate for wheat root hairs
# is about 0.080 mm h-1.

# Average lifespan of a root hair (in seconds equivalent to a temperature of T_ref_growth):
#------------------------------------------------------------------------------------------
root_hairs_lifespan = 46 * (60. * 60.)
# => According to the data from McElgunn and Harrison (1969), the lifespan of wheat root hairs is 40-55h,
# depending on the temperature. For a temperature of 20 degree Celsius, the linear regression from this data gives 46h. .

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Parameters for growth respiration:
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Growth yield (in mol of CO2 per mol of C used for structural mass):
#--------------------------------------------------------------------
yield_growth = 0.8
# => We use the range value (0.75-0.85) proposed by Thornley and Cannell (2000)

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Parameters for maintenance respiration:
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Maximal maintenance respiration rate (in mol of CO2 per gram of structural mass per second):
resp_maintenance_max = 4.1e-6 / 6. * (0.44 / 60. / 14.01) * 5
# => According to Barillot et al. (2016): max. residual maintenance respiration rate is 4.1e-6 umol_C umol_N-1 s-1,
# i.e. 4.1e-6/60*0.44 mol_C g-1 s-1 assuming an average struct_C:N_tot molar ratio of root of 60 [cf simulations by
# CN-Wheat 47 in 2020] and a C content of struct_mass of 44%. According to the same simulations, total maintenance
# respiration is about 5 times more than residual maintenance respiration.
resp_maintenance_max = 5e-8
# => According to Gifford (1995): the total maintenance respiration rate of the whole plant of wheat is about
# 0.024 gC gC-1 day-1, i.e. 5.28 e-8 assuming that the C to which this rate is related represents 44% of the dry
# structural biomass.
# Temperature dependence for this parameter:
#"""""""""""""""""""""""""""""""""""""""""""
resp_maintenance_max_T_ref = 20
resp_maintenance_max_A = -0.0442
resp_maintenance_max_B = 1.55
resp_maintenance_max_C = 1
# => We fitted the parameters on the mean curve of maintenance respiration of whole-plant wheat obtained from
# Gifford (1995).

# Affinity constant for maintenance respiration (in mol of hexose per gram of structural mass):
#----------------------------------------------------------------------------------------------
Km_maintenance = 1.67e3 * 1e-6 / 6.
# => According to Barillot et al. (2016): Km=1670 umol of C per gram for residual maintenance respiration
# (based on sucrose concentration!).

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Expected concentrations in a typical root segment (used for estimating a few parameters):
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Expected sucrose concentration in root (in mol of sucrose per gram of structural mass):
#----------------------------------------------------------------------------------------
expected_C_sucrose_root = 0.0100 / 12.01 / 12
# => 0.0025 is a plausible value according to the results of Gauthier (2019, pers. communication), but here, we use
# a plausible sucrose concentration (10 mgC g-1) in roots according to various experimental results.

# Expected hexose concentration in root (in mol of hexose per g of root):
#------------------------------------------------------------------------
expected_C_hexose_root = 1e-3

# Expected hexose concentration in soil (in mol of hexose per g of root):
#------------------------------------------------------------------------
expected_C_hexose_soil = expected_C_hexose_root / 100.
# => We expect the soil concentration to be 2 orders of magnitude lower than the root concentration.

# Expected hexose concentration in the reserve pool (in mol of hexose per g of reserve pool):
#--------------------------------------------------------------------------------------------
expected_C_hexose_reserve = expected_C_hexose_root * 2.
# => We expect the reserve pool to be two times higher than the mobile one.

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Parameters for phloem unloading/reloading:
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Maximum unloading rate of sucrose through the surface of the phloem vessels (in mol of sucrose per m2 per second):
# -------------------------------------------------------------------------------------------------------------------
surfacic_unloading_rate_reference = 0.03 / 12 * 1e-6 / (0.5 * 1)
# => According to Barillot et al. (2016b), this value is 0.03 umol C g-1 s-1, and we assume that 1 gram of dry root mass
# is equivalent to 0.5 m2 of surface. However, the unloading of sucrose is not done exactly the same in CN-Wheat as in
# RhizoDep (for example, exudation is a fraction of the sucrose unloading, unlike in RhizoDep where it is a fraction
# of hexose).
max_unloading_rate = 2e-7

# ALTERNATIVE: We use a permeability coefficient and unloading occurs through diffusion only !
unloading_by_diffusion = True
# Coefficient of permeability of unloading phloem (in gram per m2 per second):
phloem_permeability = 5.76
# => According to Ross-Eliott et al. (2017), an unloading flow of sucrose of 1.2e-13 mol of sucrose per second can be
# calculated for a sieve element of 3.6 µm and the length of the unloading zone of 350 µm, # assuming a phloem
# concentration of 0.5 mol/l. We calculated that this concentration corresponded to 5.3 e-6 mol/gDW, considering that
# the sieve element was filled with phloem sap, that the root diameter was 111 µm, and that root tissue
# density was 0.1 g cm-3. Calculating an exchange surface of the sieve tube of 4e-9 m2, we obtained a permeability
# coefficient of 5.76 gDW m-2 s-1 using the values of the flow, of the gradient of sugar concentration (assuming hexose
# concentration was 0) and of the exchange surface.
# CHEATING:
phloem_permeability = 2e-4

# Reference consumption rate of hexose for growth for a given root element (used to multiply the reference unloading rate
# when growth has consumed hexose) (mol of hexose per second):
reference_rate_of_hexose_consumption_by_growth = 3e-14

# Temperature dependence for this parameter:
#"""""""""""""""""""""""""""""""""""""""""""
phloem_unloading_T_ref = 10
phloem_unloading_A = -0.04
phloem_unloading_B = 2.9
phloem_unloading_C = 1
# => We reuse the observed evolution of Frankenberger and Johanson (1983) on invertase activity in different soils
# with temperature from 10 to 100 degree Celsius which show an increase of about 5 times between 20 degrees and
# 50 degrees (maximum), assuming that the activity of invertase outside the phloem tissues is correlated to
# the unloading rate of sucrose from phloem.

# Maximum reloading rate of hexose inside the phloem vessels (in mol of hexose per m2 per second):
#-------------------------------------------------------------------------------------------------
surfacic_loading_rate_reference = 1.2e-13/(3.6e-6*pi*350e-6) * 2.
# => We assume that the maximum loading rate should equal twice the unloading rate estimated from the calculations of
# Ross-Eliott et al. (2017), corresponding to an unloading flow of sucrose of 1.2e-13 mol of sucrose per second
# for a sieve element of 3.6 µm and the length of the unloading zone of 350 µm.

max_loading_rate = 2e-7

# Temperature dependence for this parameter:
#"""""""""""""""""""""""""""""""""""""""""""
max_loading_rate_T_ref = 10
max_loading_rate_A = -0.04
max_loading_rate_B = 2.9
max_loading_rate_C = 1
# => We reuse the temperature-evolution used for phloem unloading, based on the work of Frankenberger
# and Johanson (1983) (see above).

# Affinity constant for sucrose unloading (in mol of sucrose per g of struct_mass):
#----------------------------------------------------------------------------------
Km_unloading = 1000 * 1e-6 / 12.
# => According to Barillot et al. (2016b), this value is 1000 umol C g-1

# Affinity constant for sucrose loading (in mol of hexose per g of struct_mass):
#-------------------------------------------------------------------------------
Km_loading = Km_unloading * 2.
# => We expect the Km of loading to be equivalent as the Km of unloading, as it may correspond to the same transporter.

# Coefficient affecting the increase of loading with distance from the apex (dimensionless):
#-------------------------------------------------------------------------------------------
gamma_loading = 0.0

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Parameters for mobilization/immobilization in the reserve pool:
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Maximum concentration of hexose in the reserve pool (in mol of hexose per gram of structural mass):
#----------------------------------------------------------------------------------------------------
C_hexose_reserve_max = 5e-3

# Minimum concentration of hexose in the reserve pool (in mol of hexose per gram of structural mass):
#----------------------------------------------------------------------------------------------------
C_hexose_reserve_min = 0.

# Minimum concentration of hexose in the mobile pool for immobilization (in mol of hexose per gram of structural mass):
#----------------------------------------------------------------------------------------------------------------------
C_hexose_root_min_for_reserve = 5e-3

# Maximum immobilization rate of hexose in the reserve pool (in mol of hexose per gram of structural mass per second):
#---------------------------------------------------------------------------------------------------------------------
max_immobilization_rate = 8 * 1e-6 / 6.
# => According to Gauthier et al. (2020): the new maximum rate of starch synthesis for vegetative growth in the shoot
# is 8 umolC g-1 s-1
max_immobilization_rate = 1.8e-9
# => According to the work of Mohabir and John (1988) on starch synthesis in potatoe tubers based on labelled sucrose
# incorporation in disks of starch, the immobilization rate is about 1.8e-9 at the temperature of 20 degree Celsius,
# assuming that the potatoe starch content is 65.6% of dry matter and that the structural mass is 28.5% (data taken from
# the data of Jansen et al. (2001)).
# Temperature dependence for this parameter:
#"""""""""""""""""""""""""""""""""""""""""""
max_immobilization_rate_T_ref = 20
max_immobilization_rate_A = -0.0521
max_immobilization_rate_B = 0.861
max_immobilization_rate_C = 1
# => According to the work of Mohabir and John (1988) on starch synthesis in potatoe tubers based on labelled sucrose
# incorporation in disks of starch, the immobilization increased until 21.5 degree Celsius and then decreases again.
# We fitted the evolution of starch synthesis with temperature (8-30 degrees) to get the parameters estimation.

# Affinity constant for hexose immobilization in the reserve pool (in mol of hexose per gram of structural mass):
#----------------------------------------------------------------------------------------------------------------
Km_immobilization = expected_C_hexose_root * 1.

# Maximum mobilization rate of hexose from the reserve pool (in mol of hexose per gram of structural mass per second):
#---------------------------------------------------------------------------------------------------------------------
max_mobilization_rate = max_immobilization_rate
# Temperature dependence for this parameter:
#"""""""""""""""""""""""""""""""""""""""""""
max_mobilization_rate_T_ref = max_immobilization_rate_T_ref
max_mobilization_rate_A = max_immobilization_rate_A
max_mobilization_rate_B = max_immobilization_rate_B
max_mobilization_rate_C = max_immobilization_rate_C
# => We assume that the mobilization obeys to the same evolution with temperature as the immobilization process.

# Affinity constant for hexose remobilization from the reserve (in mol of hexose per gram of structural mass):
#-------------------------------------------------------------------------------------------------------------
Km_mobilization = expected_C_hexose_reserve * 5.

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Parameters for root hexose exudation:
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Expected exudation rate (in mol of hexose per m2 per s):
#---------------------------------------------------------
expected_exudation_efflux = 608 * 0.000001 / 12.01 / 6 / 3600 * 1 / (0.5 * 10)
# => According to Jones and Darrah (1992): the net efflux of C for a root of maize is 608 ug C g-1 root DW h-1,
# and we assume that 1 gram of dry root mass is equivalent to 0.5 m2 of external surface.
# OR:
# expected_exudation_efflux = 5.2 / 12.01 / 6. * 1e-6 * 100. ** 2. / 3600.
# => Explanation: According to Personeni et al. (2007), we expect a flux of 5.2 ugC per cm2 per hour

# Permeability coefficient (in g of struct_mass per m2 per s):
#-------------------------------------------------------------
Pmax_apex = expected_exudation_efflux / (expected_C_hexose_root - expected_C_hexose_soil)
# => We calculate the permeability according to the expected exudation flux and expected concentration gradient
# between cytosol and soil.
# Temperature dependence for this parameter:
#"""""""""""""""""""""""""""""""""""""""""""
permeability_coeff_T_ref = 20
permeability_coeff_A = 0
permeability_coeff_B = 1
permeability_coeff_C = 0
# => We assume that the permeability does not directly depend on temperature, according to the contrasted results
# obtained by Wan et al. (2001) on poplar, Shen and Yan (2002) on crotalaria, Hill et al. (2007) on wheat,
# or Kaldy (2012) on a sea grass.

# Coefficient affecting the decrease of permeability with distance from the apex (dimensionless):
#------------------------------------------------------------------------------------------------
gamma_exudation = 0.4
# => According to Personeni et al (2007), this gamma coefficient showing the decrease in permeability
# along the root is 0.41-0.44.

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Parameters for root hexose uptake from soil:
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Maximum rate of influx of hexose from soil to roots (in mol of hexose per m2 per s):
#-------------------------------------------------------------------------------------
uptake_rate_max = 277 * 0.000000001 / (60 * 60 * 24) * 1000 * 1 / (0.5 * 1)
# => According to Jones and Darrah (1996), the uptake rate measured for all sugars tested with an individual external
# concentration of 100 uM is equivalent to 277 nmol hexose mg-1 day-1, and we assume that 1 gram of dry root mass is
# equivalent to 0.5 m2 of external surface.
# Temperature dependence for this parameter:
#"""""""""""""""""""""""""""""""""""""""""""
uptake_rate_max_T_ref = 20
uptake_rate_max_A = 0
uptake_rate_max_B = 3.82
uptake_rate_max_C = 1
# => The value for B (Q10) is adapted from the work of Coody et al. (1986, SBB),
# who provided the evolution of the maximal uptake of glucose by soil microorganisms at 4, 12 and 25 degree C.

# Affinity constant for hexose uptake (in mol of hexose per g of struct_mass):
#-----------------------------------------------------------------------------
Km_uptake = Km_loading
# We assume that the transporters able to reload sugars in the phloem and to take up sugars from the soil behave
# in a similar manner.

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Parameters for root mucilage secretion:
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Maximum rate of mucilage secretion (in mol of hexose per m2 per s):
#--------------------------------------------------------------------
secretion_rate_max = 1e-5/(12.01*6)/(0.1*pi)*1e4/(60.*60.*24.)
# => According to the measurements of Paull et al. (1975) and Chaboud (1983) on maize and of Horst et al. (1982)
# on cowpea, the mucilage secretion rate at root tip evolves within 5 to 10 ugC per cm of root per day.
# We therefore chose 10 gC per cm per day as the maximal rate, and convert it in mol of hexose per m2 per second,
# assuming that the root tip is a cylinder of 1 mm diameter.
# Temperature dependence for this parameter:
#"""""""""""""""""""""""""""""""""""""""""""
secretion_rate_max_T_ref = 20
secretion_rate_max_A = 0
secretion_rate_max_B = 2
secretion_rate_max_C = 1
# => We arbitrarily assume that the secretion of mucilage exponentially increases with soil temperature with a Q10 of 2,
# although we could not find any experimental evidence for this.

# Affinity constant for hexose uptake (in mol of hexose per g of struct_mass):
#-----------------------------------------------------------------------------
Km_secretion = Km_loading/2.
# => We assume that the concentration of root hexose for which mucilage secretion is half of the maximal rate
# is two-times lower than the one for which phloem reloading is half of the maximal rate.

# Coefficient affecting the decrease of mucilage secretion with distance from the apex (dimensionless):
#------------------------------------------------------------------------------------------------------
gamma_secretion = 1
# => We assume that the mucilage secretion rapidly decreases when moving away from the apex.

# Maximal surfacic concentration of mucilage at the soil-root interface, above which no mucilage secretion is possible:
#----------------------------------------------------------------------------------------------------------------------
# (in mol of equivalent-hexose per m2 of external surface):
Cs_mucilage_soil_max = 10. # TODO: do a real estimation!

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Parameters for root cells release:
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Average release of root cells (in mol of equivalent-hexose per m2 of lateral external surface per second):
#-----------------------------------------------------------------------------------------------------------
surfacic_cells_release_rate = 1279/6476*(33.9*1e6*1e-18)*0.1*1000000*0.44/12.01/6/(24*60*60)/(0.8*0.001*pi*0.8*0.001)
#=> We used the measurements by Clowes and Wadekar 1988 on Zea mays root cap cells obtained at 20 degree Celsius,
# i.e. 1279 cells per day.
# We recalculated the amount of equivalent hexose by relating the number of cap cells produced per day to a volume
# knowing that the whole cap was made of 6476 cells and had a volume of 33.9 *10^6 micrometer^3.
# The volume was later converted into a mass assuming a density of 0.1 g cm-3. We then assumed that the
# surface of root cap was equivalent to the lateral surface of a cylinder of radius 0.8 mm and height 0.8 mm
# (meristem size = 0.79-0.81 mm).
# Temperature dependence for this parameter:
#"""""""""""""""""""""""""""""""""""""""""""
surfacic_cells_release_rate_T_ref = 20
surfacic_cells_release_rate_A = -0.187
surfacic_cells_release_rate_B = 2.48
surfacic_cells_release_rate_C = 1
# => This corresponds to a bell-shape where the maximum is obtained at 31 degree Celsius, obtained by fitting the data
# from Clowes and Wadekar (1988) on Zea mays roots between 15 and 35 degree.

# Maximal surfacic concentration of root cells in soil, above which no release of cells is possible
#--------------------------------------------------------------------------------------------------
# (in mol of equivalent-hexose per m2 of root external surface):
Cs_cells_soil_max = 10 # TODO: do a real estimation!

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Parameters for soil degradation of root-released materials:
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Maximum degradation rate of hexose in soil (in mol of hexose per m2 per s):
#----------------------------------------------------------------------------
hexose_degradation_rate_max = uptake_rate_max * 10.
# => We assume that the maximum degradation rate is 10 times higher than the maximum uptake rate
# Temperature dependence for this parameter:
#"""""""""""""""""""""""""""""""""""""""""""
hexose_degradation_rate_max_T_ref = 20
hexose_degradation_rate_max_A = 0
hexose_degradation_rate_max_B = 3.98
hexose_degradation_rate_max_C = 1
# => The value for B (Q10) has been fitted from the evolution of Vmax measured by Coody et al. (1986, SBB),
# who provided the evolution of the maximal uptake of glucose by soil microorganisms at 4, 12 and 25 degree C.

# Affinity constant for soil hexose degradation (in mol of hexose per g of struct_mass):
#---------------------------------------------------------------------------------------
Km_hexose_degradation = Km_uptake / 2.
# => According to what Jones and Darrah (1996) suggested, we assume that this Km is 2 times lower than the Km
# corresponding to root uptake of hexose (350 uM against 800 uM in the original article).

# Maximum degradation rate of mucilage in soil (in mol of equivalent-hexose per m2 per s):
#-----------------------------------------------------------------------------------------
mucilage_degradation_rate_max = hexose_degradation_rate_max
# => We assume that the maximum degradation rate for mucilage is equivalent to the one defined for hexose.
# Temperature dependence for this parameter:
#"""""""""""""""""""""""""""""""""""""""""""
mucilage_degradation_rate_max_T_ref = hexose_degradation_rate_max_T_ref
mucilage_degradation_rate_max_A = hexose_degradation_rate_max_A
mucilage_degradation_rate_max_B = hexose_degradation_rate_max_B
mucilage_degradation_rate_max_C = hexose_degradation_rate_max_C
# => We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.

# Affinity constant for soil mucilage degradation (in mol of hexose per g of struct_mass):
#---------------------------------------------------------------------------------------
Km_mucilage_degradation = Km_hexose_degradation
# => We assume that Km for mucilage degradation is identical to the one for hexose degradation.

# Maximum degradation rate of root cells at the soil/root interface (in mol of equivalent-hexose per m2 per s):
#---------------------------------------------------------------------------------------------------------
cells_degradation_rate_max = hexose_degradation_rate_max / 2.
# => We assume that the maximum degradation rate for cells is equivalent to the half of the one defined for hexose.
# Temperature dependence for this parameter:
#"""""""""""""""""""""""""""""""""""""""""""
cells_degradation_rate_max_T_ref = hexose_degradation_rate_max_T_ref
cells_degradation_rate_max_A = hexose_degradation_rate_max_A
cells_degradation_rate_max_B = hexose_degradation_rate_max_B
cells_degradation_rate_max_C = hexose_degradation_rate_max_C
# => We assume that all other parameters for cells degradation are identical to the ones for hexose degradation.

# Affinity constant for soil cells degradation (in mol of equivalent-hexose per g of struct_mass):
#-------------------------------------------------------------------------------------------------
Km_cells_degradation = Km_hexose_degradation
# => We assume that Km for cells degradation is identical to the one for hexose degradation.

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Parameters for C-controlled growth rates:
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Proportionality factor between the radius and the length of the root apical zone in which C can sustain root elongation:
# #-----------------------------------------------------------------------------------------------------------------------
growing_zone_factor = 8 *2.  # m m-1
# => According to illustrations by Kozlova et al. (2020), the length of the growing zone corresponding to the root cap,
# meristem and elongation zones is about 8 times the diameter of the tip.

# Affinity constant for root elongation (in mol of hexose per g of struct_mass):
#-------------------------------------------------------------------------------
Km_elongation = 1250 * 1e-6 / 6.
# => According to Barillot et al. (2016b): Km for root growth is 1250 umol C g-1 for sucrose
# => According to Gauthier et al (2020): Km for regulation of the RER by sucrose concentration in hz = 100-150 umol C g-1

# Affinity constant for root thickening (in mol of hexose per g of struct_mass):
#-------------------------------------------------------------------------------
Km_thickening = Km_elongation
# => We assume that the Michaelis-Menten constant for thickening is the same as for root elongation.

# Maximal rate of relative increase in root radius (in m per m per s):
#---------------------------------------------------------------------
relative_root_thickening_rate_max = 5. / 100. / (24. * 60. * 60.)
# => We consider that the radius can't increase by more than 5% every day

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Parameters from nodules dynamics:
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Probability (between 0 and 1) of nodule formation for each apex that elongates:
#--------------------------------------------------------------------------------
nodule_formation_probability = 0.5

# Maximal radius of nodule (in m):
#---------------------------------
nodule_max_radius = D_ini * 20.

# Affinity constant for nodule thickening (in mol of hexose per g of struct_mass):
#---------------------------------------------------------------------------------
Km_nodule_thickening = Km_elongation * 100

# Maximal rate of relative increase in nodule radius (in m per m per s):
#-----------------------------------------------------------------------
relative_nodule_thickening_rate_max = 20. / 100. / (24. * 60. * 60.)
# => We consider that the radius can't increase by more than 20% every day

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Parameters for calculating surfaces and barriers to transport
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Relative ratio between the exchange surface of a compartment in the root and its external surface (m2 per m2):
#---------------------------------------------------------------------------------------------------------------
phloem_surfacic_fraction = 0.64 # According to Frederic Rees' first estimations
stelar_parenchyma_surfacic_fraction  = 8 # According to Tristan Gerault's first estimation
cortical_parenchyma_surfacic_fraction  = 10 # According to Tristan Gerault's first estimation
epidermal_parenchyma_surfacic_fraction  = 8 # According to Tristan Gerault's first estimation

# Ratio between the length of the meristem zone and root radius (m per m):
#-------------------------------------------------------------------------
meristem_limite_zone_factor = 1. # We assume that the length of the meristem zone is equal to the radius of the root

# Relative conductance in the meristem zone (m per m2):
#-------------------------------------------------------
relative_conductance_at_meristem = 0.5 # We assume that the conductance of cells walls is reduced by 2 in the meristem zone.

# Ratio between the distance from tip where barriers formation starts/ends, and root radius (m per m):
#-----------------------------------------------------------------------------------------------------
start_distance_for_endodermis_factor = meristem_limite_zone_factor
end_distance_for_endodermis_factor = growing_zone_factor * 3.
start_distance_for_exodermis_factor = growing_zone_factor
end_distance_for_exodermis_factor = growing_zone_factor * 10.

# Thermal age of root cells for starting/ending the formation of endodermis and exodermis (second equivalent to T_ref_growth):
#-----------------------------------------------------------------------------------------------------------------------------
start_thermal_time_for_endodermis_formation = 60.*60.*24.*1. # We assume that the endodermis formation starts after 1 day at T_ref_growth.
end_thermal_time_for_endodermis_formation = 60.*60.*24.*3. # We assume that the endodermis formation is completed after 3 days at T_ref_growth.
start_thermal_time_for_exodermis_formation = 60.*60.*24.*10. # We assume that the exodermis formation starts after 10 days at T_ref_growth.
end_thermal_time_for_exodermis_formation = 60.*60.*24.*30. # We assume that the exodermis formation is completed after 30 days at T_ref_growth.

# Gompertz parameters for describing the evolution of apoplastic barriers with cell age:
#---------------------------------------------------------------------------------------
# The following parameters estimations are derived from the works of Enstone et al. (2005, PCE) and Dupuy et al. (2016,
# Chemosphere) on the formation of apoplastic barriers in maize, fitting their data with a Gompertz curve.
endodermis_a = 100. # This parameter corresponds to the asymptote of the process (adimensional, as cond_endodermis)
exodermis_a = 100. # This parameter corresponds to the asymptote of the process (adimensional, as cond_exodermis)
endodermis_b = 3.25 * (60.*60.*24.) # This parameter corresponds to the time lag before the large increase (second equivalent to T_ref_growth)
exodermis_b = 5.32 * (60.*60.*24.) # This parameter corresponds to the time lag before the large increase (second equivalent to T_ref_growth)
endodermis_c = 1.11 / (60.*60.*24.) # This parameter reflects the slope of the increase (per second equivalent to T_ref_growth)
exodermis_c = 1.11 / (60.*60.*24.) # This parameter reflects the slope of the increase (per second equivalent to T_ref_growth)
# Note: Interestingly, the mean value for parameter c from the two studies was identical for both endodermis and exodermis,
# while within each study, c was distinct between endodermis and exodermis.

# Maximal thermal time above which no barrier disruption is considered anymore after a lateral root has emerged
# (in second equivalent to temperature T_ref_growth):
#-------------------------------------------------------------------------------------------------------------
max_thermal_time_since_endodermis_disruption = 6 * 60. * 60. # We assume that after 6h, no disruption is observed anymore!
max_thermal_time_since_exodermis_disruption = 48 * 60. * 60. # We assume that after 48h, no disruption is observed anymore!

