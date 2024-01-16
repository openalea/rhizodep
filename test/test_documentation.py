from rhizodep.rhizodep import Model


def test_get_documentation():
    rhizodep = Model(time_step=3600, random=False)

    print(rhizodep.documentation)


