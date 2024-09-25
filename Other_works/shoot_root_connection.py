# We define a class that will be used to contain the properties of the interface between shoot and roots:
class shoot_root_interface(object):
    # We initiate the object "interface":
    def __init__(self):
        self.shoot_struct_mass = 0.
        self.root_struct_mass = 0.

        self.shoot_labile_C_pool = 0. # mol of C per shoot system
        self.shoot_labile_N_pool = 0. # mol of N per shoot system
        self.root_labile_C_pool = 0.  # mol of C per root system
        self.root_labile_N_pool = 0.  # mol of N per root system

        self.C_flux_to_shoot = 0.  # mol of C per second
        self.N_flux_to_shoot = 0.  # mol of N per second
        self.C_flux_to_root = 0.  # mol of C per second
        self.N_flux_to_root = 0.  # mol of N per second

        self.transpiration_flux = 0. # mol of water per second
#-----------------------------------------------------------------------------------------------------------------------

def shoot_root_exchange(i, computing_C_exchange=False, computing_N_exchange=False,
                        C_option="by_diffusion", N_option="by_diffusion",
                        resistance_to_C = 1., resistance_to_N = 1., water_concentration = 10.):

    # If C is to be exchanged by diffusion based on the labile pools in shoots and roots:
    if computing_C_exchange and C_option == "by_diffusion":
        # Then the net C flux is calculated using a gradient of concentration and a resistance:
        shoot_to_root_C_flux = (i.shoot_labile_C_pool / i.shoot_struct_mass - i.root_labile_C_pool / i.root_struct_mass) \
                               / resistance_to_C
        if shoot_to_root_C_flux > 0.:
            i.C_flux_to_shoot = 0.
            i.C_flux_to_root = shoot_to_root_C_flux
        else:
            i.C_flux_to_shoot = - shoot_to_root_C_flux
            i.C_flux_to_root = 0.
    # Otherwise, the net flux of C has been imposed to i by the shoots (or even by the roots)!

    if computing_N_exchange:
        # If N is to be exchanged by diffusion based on the labile pools in shoots and roots:
        if N_option == "by_diffusion":
            # Then the net N flux is calculated using a gradient of concentration and a resistance:
            root_to_shoot_N_flux = (i.root_labile_N_pool / i.root_struct_mass - i.shoot_labile_N_pool /
                                    i.shoot_struct_mass) \
                                   / resistance_to_N
            if root_to_shoot_N_flux > 0.:
                i.N_flux_to_shoot = root_to_shoot_N_flux
                i.N_flux_to_root = 0.
            else:
                i.N_flux_to_shoot = 0.
                i.N_flux_to_root = - root_to_shoot_N_flux
        # Otherwise, a transpiration flux has to be provided, from which the N flux is calculated based on the concentration
        # of N in the root labile pool:
        elif N_option == "by_water_flux":
            if i.transpiration_flux >= 0.:
                i.N_flux_to_shoot = i.transpiration_flux / water_concentration * i.root_labile_N_pool / i.root_struct_mass
                # NOTE: N_flux in mol of N per plant and per second
                #       transpiration_flux in mol of water per plant and per second
                #       water_concentration in mol of water per gDW of root structural mass
                #       root_labile_N_pool in mol of N per plant
                #       root_struct_mass in gDW of root structural mass
            else:
                print("ERROR - there is a problem with the water transpiration flux!")
        else:
            print("ERROR - the option", N_option, "is not recognized!")

    return
#-----------------------------------------------------------------------------------------------------------------------

# EXAMPLE:
##########

i = shoot_root_interface()
i.shoot_struct_mass = 2.
i.root_struct_mass = 0.5

i.shoot_labile_C_pool = 5.
i.shoot_labile_N_pool = 0.01
i.root_labile_C_pool = 0.5
i.root_labile_N_pool = 0.02

i.transpiration_flux = 0.1

shoot_root_exchange(i, C_option="by_diffusion", N_option="by_water_flux")
print("The new flux of C from shoot to root is", i.C_flux_to_root, "mol of C per second.")
print("The new flux of N from root to shoot is", i.N_flux_to_shoot, "mol of N per second.")