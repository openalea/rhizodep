import pickle

from rhizodep.rhizodep import Model


def test_mtg_initiation():
    """
    This function tests MTG initiation by model in order to check if the initiation is reproducible.
    """
    scenario = {}
    # We initialize the model
    rhizodep = Model(time_step=3600, **scenario)

    # We get the MTG attribute initiated during __init__
    g = rhizodep.g

    # We import a reference MTG supposed to be identical to produced MTG if no random effect have been applied
    with open('inputs/root_initiated.pckl', "rb") as f:
        reference_g = pickle.load(f)

    assert g.vertices(scale=g.max_scale()) == reference_g.vertices(scale=reference_g.max_scale())
