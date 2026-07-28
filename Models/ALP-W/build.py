import numpy as np

from src.foresee import Model
from src.utils.utility import ensure_model_layout
from src.utils.detectors import default_detectors


BR_KAON = (
    "{prefactor} * coupling**2 * np.sqrt("
    "(1-(mass+self.masses('pid1'))**2/self.masses('pid0')**2)*"
    "(1-(mass-self.masses('pid1'))**2/self.masses('pid0')**2)"
    ")"
)
BR_B = "2.3e4 * coupling**2 * (1-(mass/self.masses('pid0'))**2)**2"


def build_model(
    path,
    energy="14",
    nsample_2body=100,
    generators_light=None,
    generators_heavy=None,
):
    """
    Build and return the FORESEE model object for this model.

    Parameters
    ----------
    path: str
        Model directory (Models/ALP-W/). Supplied by Foresee.load_model.
    energy: str
        Collider energy in TeV.
    nsample_2body: int
        Sampling points per production channel. Applied uniformly across all 9
        channels (3 kaon + 4 B + 2 B_s).
    generators_light: [str]
        Generators for light-meson (K) production. Defaults to ['EPOSLHC', 'SIBYLL', 'QGSJET'].
    generators_heavy: [str]
        Generators for heavy-meson (B, B_s) production. Defaults to ['NLO-P8', 'NLO-P8-Max', 'NLO-P8-Min'].

    Returns
    -------
        Model
    """
    if generators_light is None:
        generators_light = ['EPOSLHC', 'SIBYLL', 'QGSJET']
    if generators_heavy is None:
        generators_heavy = ['NLO-P8', 'NLO-P8-Max', 'NLO-P8-Min']

    ensure_model_layout(path, link_direct=False)

    model = Model("ALP-W", path=path)

    for pid0, pid1, prefactor in [
        ("130",  "111", "4.5"),
        ("321",  "211", "10.5"),
        ("-321", "211", "10.5"),
    ]:
        model.add_production_2bodydecay(
            pid0=pid0,
            pid1=pid1,
            br=BR_KAON.format(prefactor=prefactor),
            generator=generators_light,
            energy=energy,
            nsample=nsample_2body,
        )

    for pid0, pid1 in [
        ("511",  "130"),
        ("-511", "130"),
        ("521",  "321"),
        ("-521", "-321"),
        ("531",  "333"),
        ("-531", "333"),
    ]:
        model.add_production_2bodydecay(
            pid0=pid0,
            pid1=pid1,
            br=BR_B,
            generator=generators_heavy,
            energy=energy,
            nsample=nsample_2body,
        )

    model.set_ctau_1d(
        filename="model/ctau.txt",
    )

    decay_modes = ["gamma_gamma", "e_e_gamma"]
    model.set_br_1d(
        modes=decay_modes,
        finalstates=[[22, 22], [11, -11, 22]],
        filenames=["model/br/" + mode + ".txt" for mode in decay_modes],
    )

    return model
