import numpy as np

from src.foresee import Model
from src.utils.utility import ensure_model_layout
from src.utils.detectors import default_detectors


BR_B = "76 * coupling**2 * (1-(mass/self.masses('pid0'))**2)**2"

MIXING_PI0    = "(3.671) * coupling * (-0.180 * mass**2)/(mass**2 - self.masses('pid')**2)"
MIXING_ETA    = "(3.671) * coupling * (-0.3596 * mass**2 + 0.0021)/(mass**2 - self.masses('pid')**2)"
MIXING_ETA_PR = "(3.671) * coupling * (-0.3362 * mass**2 + 0.0092)/(mass**2 - self.masses('pid')**2)"


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
        Model directory (Models/ALP-g/). Supplied by Foresee.load_model.
    energy: str
        Collider energy in TeV.
    nsample_2body: int
        Sampling per B-decay production channel. add_production_mixing does not
        consume an nsample.
    generators_light: [str]
        Generators for the pi^0, eta, eta' mixing channels. Defaults to ['EPOSLHC', 'SIBYLL', 'QGSJET'].
    generators_heavy: [str]
        Generators for the B-meson decay channels. Defaults to ['NLO-P8', 'NLO-P8-Max', 'NLO-P8-Min'].

    Returns
    -------
        Model
    """
    if generators_light is None:
        generators_light = ['EPOSLHC', 'SIBYLL', 'QGSJET']
    if generators_heavy is None:
        generators_heavy = ['NLO-P8', 'NLO-P8-Max', 'NLO-P8-Min']

    ensure_model_layout(path, link_direct=False)

    model = Model("ALP-g", path=path)

    for pid0, pid1 in [
        ("511",  "130"),
        ("-511", "130"),
        ("521",  "321"),
        ("-521", "-321"),
    ]:
        model.add_production_2bodydecay(
            pid0=pid0,
            pid1=pid1,
            br=BR_B,
            generator=generators_heavy,
            energy=energy,
            nsample=nsample_2body,
        )

    for pid, mixing in [
        ("111", MIXING_PI0),
        ("221", MIXING_ETA),
        ("331", MIXING_ETA_PR),
    ]:
        model.add_production_mixing(
            pid=pid,
            mixing=mixing,
            generator=generators_light,
            energy=energy,
        )

    model.set_ctau_1d(filename="model/ctau.txt")

    decay_modes = ["br_2Gamma", "br_3Pi0", "br_Pi+Pi-Gamma", "br_Pi+Pi-Pi0", "br_2PiEta"]
    model.set_br_1d(
        modes=decay_modes,
        finalstates=[[22, 22], [111, 111, 111], [211, -211, 22], [211, -211, 111], [211, -211, 221]],
        filenames=["model/br/" + mode + ".dat" for mode in decay_modes],
    )

    return model
