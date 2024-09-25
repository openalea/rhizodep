import pandas as pd

########################################################################################################################

# We define a class containing all the properties of a soil voxel:
class soil_voxel(object):

    # We initiate the object with a list of properties:
    def __init__(self):
        self.volume = 1.
        self.bulk_density = 1.3 # g cm-3
        self.POC = 2. # gC per kg of soil
        self.MAOC = 8. # gC per kg of soil
        self.DOC = 0.5 # gC per kg of soil
        self.PON = 0.1 # gN per kg of soil
        self.MAON = 0.8 # gN per kg of soil
        self.DON = 0.05 # gN per kg of soil
        self.microbial_C = 0.2 # gC per kg of soil
        self.microbial_N = 0.03 # gN per kg of soil
        self.dissolved_mineral_N = 0.01 # gN per kg of soil
        self.CO2 = 0. # gC per kg of soil

########################################################################################################################

# Defining the parameters:
k_POC = 0.5 # year-1
k_MAOC = 0.01 # year-1
k_DOC = 20. # year-1
k_MBC = 10. # year-1

microbial_C_min = 0.1
microbial_C_max = 0.5

CN_ratio_POM = 20 # gC per gN
CN_ratio_MAOM = 10 # gC per gN
CN_ratio_microbial_biomass = 8 # gC per gN

CUE_POC = 0.2
CUE_MAOC = 0.4
CUE_DOC = 0.4
CUE_MBC = 0.3

max_N_uptake_per_microbial_C = 1e-7 # gN per second per gC of microbial C
Km_microbial_N_uptake = 0.01 # gN per kg of soil

def managing_external_exchanges(v,
                                net_root_necromass_C_inputs=0.01,
                                net_root_cells_C_inputs=0.01,
                                net_exudates_C_inputs=0.01,
                                net_mucilage_C_inputs=0.01,
                                net_root_necromass_N_inputs=0.01,
                                net_root_cells_N_inputs=0.01,
                                net_exudates_N_inputs=0.01,
                                net_mucilage_N_inputs=0.01,
                                ):

    # We reset all net inputs to 0:
    v.net_inputs_to_POC = 0.
    v.net_inputs_to_MAOC = 0.
    v.net_inputs_to_DOC = 0.
    v.net_inputs_to_microbial_C = 0.
    v.net_inputs_to_PON = 0.
    v.net_inputs_to_MAON = 0.
    v.net_inputs_to_DON = 0.
    v.net_inputs_to_microbial_N = 0.
    v.net_inputs_to_dissolved_mineral_N = 0.

    # We consider exchanges with all roots present within the voxel:
    v.net_inputs_to_POC += net_root_necromass_C_inputs + net_root_cells_C_inputs
    v.net_inputs_to_DOC += net_exudates_C_inputs + net_mucilage_C_inputs

    v.net_inputs_to_PON += net_root_necromass_N_inputs + net_root_cells_N_inputs
    v.net_inputs_to_DON += net_exudates_N_inputs + net_mucilage_N_inputs

    # We consider exchanges with the other voxels around:
    v.net_inputs_to_DOC += 0.01
    v.net_inputs_to_DON += 0.001
    v.net_inputs_to_dissolved_mineral_N += -0.001

    return


def organic_matter_turnover(v, time_step_in_seconds=60.*60.*24.*365.):

    # We adjust the time step to years:
    dt = time_step_in_seconds / (60.*60.*24.*365) # We convert the time step in years

    # We calculate a rate modifier that can increase by up to 2 the rate of degradation, based on the size of the
    # microbial biomass C pool:
    if v.microbial_C > microbial_C_min and v.microbial_C < microbial_C_max:
        rate_modifier = 1. + (v.microbial_C - microbial_C_min)  / (microbial_C_max - microbial_C_min)
        print("The rate modifier is", rate_modifier)
    else:
        rate_modifier = 1.

    # We calculate the net change in the SOC pools due to degradation:
    v.dPOC = -k_POC * rate_modifier * v.POC * dt
    v.dMAOC = -k_MAOC * rate_modifier * v.MAOC * dt
    v.dDOC = - k_DOC * rate_modifier * v.DOC * dt
    v.dMBC = -k_MBC * rate_modifier * v.microbial_C * dt

    # We calculate the corresponding net change of the SON pools based on stoechiometric ratios of degradation products:
    v.dPON =  v.dPOC / CN_ratio_POM
    v.dMAON = v.dMAOC / CN_ratio_MAOM
    v.dDON = v.dDOC * v.DON / v.DOC
    v.dMBN = v.dMBC / CN_ratio_microbial_biomass

    return

def N_organization(v, time_step_in_seconds=60.*60.*24.*365.):

    # We calculate the amount of mineral N taken by microorganisms over this time step:
    v.mineral_N_microbial_uptake = (max_N_uptake_per_microbial_C * v.microbial_C
                            * v.dissolved_mineral_N / (Km_microbial_N_uptake + v.dissolved_mineral_N) * time_step_in_seconds)

    return

def CN_balance(v):

    # For POC, MAOC and DOC, the new concentrations only results from the turnover and the new net inputs:
    v.POC += v.dPOC + v.net_inputs_to_POC
    v.MAOC += v.dMAOC + v.net_inputs_to_MAOC
    v.DOC += v.dDOC + v.net_inputs_to_DOC

    # For microbial biomass C, the new concentrations results i) from the turnover of this pool, ii) from a fraction of
    # the degradation products of each SOC pool, including microbial biomass itself, and iii) from new net inputs, if any:
    v.microbial_C += v.dMBC + (-(v.dPOC * CUE_POC + v.dMAOC * CUE_MAOC + v.dDOC * CUE_DOC + v.dMBC * CUE_MBC)
                      + v.net_inputs_to_microbial_C)
    v.CO2 += -(v.dPOC * (1 - CUE_POC) + v.dMAOC * (1 - CUE_MAOC) + v.dMBC * (1 - CUE_MBC) + v.dDOC * (1 - CUE_DOC))

    v.PON += v.dPON + v.net_inputs_to_PON
    v.MAON += v.dMAON + v.net_inputs_to_MAON
    v.DON += v.dDON + v.net_inputs_to_DON
    v.microbial_N += v.dMBN + (-(v.dPOC * CUE_POC / CN_ratio_POM + v.dMAOC * CUE_MAOC / CN_ratio_MAOM
                                 + v.dDOC * CUE_DOC * v.DON / v.DOC + v.dMBC * CUE_MBC * CN_ratio_microbial_biomass)
                               + v.mineral_N_microbial_uptake + v.net_inputs_to_microbial_N)

    v.dissolved_mineral_N += -v.dMBN - v.mineral_N_microbial_uptake + v.net_inputs_to_dissolved_mineral_N

    return

########################################################################################################################

# MAIN PROGRAM:

# We initialize a voxel:
v = soil_voxel()

print("The POC of the voxel is initially", v.POC)
print("The MAOC of the voxel is initially", v.MAOC)
print("The microbial C of the voxel is initially", v.microbial_C)
print("")

list_of_properties = [property for property in v.__dict__.keys()]
column_names = ["Time_step"] + list_of_properties
table = [column_names]
step_in_seconds = 60.*60.*24.*10

for i in range(1,301):

    # We attribute the net inputs and outputs in the soil voxel to the suitable C or N compartment:
    managing_external_exchanges(v,
                                net_root_necromass_C_inputs=0.01,
                                net_root_cells_C_inputs=0.01,
                                net_exudates_C_inputs=0.01,
                                net_mucilage_C_inputs=0.01,
                                net_root_necromass_N_inputs=0.01,
                                net_root_cells_N_inputs=0.001,
                                net_exudates_N_inputs=0.001,
                                net_mucilage_N_inputs=0.001
                                )
    # We compute soil organic matter dynamics:
    organic_matter_turnover(v, time_step_in_seconds = step_in_seconds)
    # We compute the uptake of N by microorganisms:
    N_organization(v, time_step_in_seconds = step_in_seconds)
    # We compute C and N balance in the voxel:
    CN_balance(v)

    print("The POC of the voxel is now", v.POC)
    print("The MAOC of the voxel is now", v.MAOC)
    print("The microbial C of the voxel is now", v.microbial_C)

    # We record the main properties of the voxel and add it to a general table:
    current_v_properties = [i]
    for property in list_of_properties:
        # We add the value of this property to the list:
        current_v_properties.append(getattr(v, property, "NA"))
    table.append(current_v_properties)
    print("")

# We create the final table of results and export it:
data_frame = pd.DataFrame(table)
# We record the dataframe as a csv file:
data_frame.to_csv('voxel_results.csv', na_rep='NA', index=False, header=False)