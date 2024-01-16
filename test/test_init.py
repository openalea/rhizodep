from rhizodep.rhizodep import Model


def test_model_initialisation():
    model = Model(time_step=3600)

    print(model.documentation)
