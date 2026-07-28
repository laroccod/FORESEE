import numpy as np

from src.foresee import Model
from src.utils.utility import ensure_model_layout
from src.utils.detectors import default_detectors


def meson_br(coef):
    return (
        f"{coef} * coupling**2 * np.sqrt("
        "(1-(mass+self.masses('pid1'))**2/self.masses('pid0')**2)"
        "*(1-(mass-self.masses('pid1'))**2/self.masses('pid0')**2))"
    )


def build_model(
    path,
    energy="14",
    nsample_2body=100,
    generators_light=None,
):
    """
    Build and return the FORESEE model object for this model.

    Parameters
    ----------
    path: str
        Model directory (Models/UpPhilic/). Supplied by Foresee.load_model.
    energy: str
        Collider energy in TeV. Defaults to "14".
    nsample_2body: int
        Sampling points per production channel. Defaults to 100.
    generators_light: [str]
        Generators for the eta, eta', and kaon channels. Defaults to
        ["EPOSLHC", "SIBYLL", "QGSJET"].

    Returns
    -------
        Model
    """
    if generators_light is None:
        generators_light = ["EPOSLHC", "SIBYLL", "QGSJET"]

    ensure_model_layout(path, link_direct=False)

    model = Model("UpPhilic", path=path)

    model.add_production_2bodydecay(
        pid0="221",
        pid1="111",
        br=meson_br("1.26e5"),
        generator=generators_light,
        energy=energy,
        nsample=nsample_2body,
    )

    model.add_production_2bodydecay(
        pid0="331",
        pid1="111",
        br=meson_br("273."),
        generator=generators_light,
        energy=energy,
        nsample=nsample_2body,
    )

    model.add_production_2bodydecay(
        pid0="321",
        pid1="211",
        br=meson_br("7.42"),
        generator=generators_light,
        energy=energy,
        nsample=nsample_2body,
    )
    model.add_production_2bodydecay(
        pid0="-321",
        pid1="-211",
        br=meson_br("7.42"),
        generator=generators_light,
        energy=energy,
        nsample=nsample_2body,
    )

    model.set_ctau_1d(filename="model/ctau.txt")
    decay_modes = ["gamma", "pi0_pi0", "pi+_pi-"]
    model.set_br_1d(
        modes=decay_modes,
        finalstates=[[22, 22], [111, 111], [211, -211]],
        filenames=["model/br/" + mode + ".txt" for mode in decay_modes],
    )

    return model
