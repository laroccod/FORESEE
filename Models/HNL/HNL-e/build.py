import os
import sys

import numpy as np

from src.foresee import Model
from src.utils.utility import ensure_model_layout


HNL_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HNL_PKG not in sys.path:
    sys.path.insert(0, HNL_PKG)
from HNLCalc import HeavyNeutralLepton  


def build_model(
    path,
    energy="14",
    nsample_2body=10,    nsample_3body=10,
    generators_light=None,
    generators_heavy=None,
    ve=1,
    vmu=0,
    vtau=0,
):
    """
    Build and return the FORESEE model object for this model.

    Parameters
    ----------
    path: str
        Model directory (Models/HNL/HNL-e/). Supplied by Foresee.load_model.
    energy: str
        Collider energy in TeV.
    nsample_2body, nsample_3body: int
        Sampling points per production channel. HNL has many channels; the
        original notebook uses 10. Raise for higher-statistics studies.
    generators_light: [str]
        Defaults to ['EPOSLHC', 'SIBYLL', 'QGSJET'].
    generators_heavy: [str]
        Defaults to ['NLO-P8', 'NLO-P8-Max', 'NLO-P8-Min'].
    ve, vmu, vtau: float
        Active-neutrino mixing components. Defaults (1, 0, 0) realize a pure
        electron-flavor HNL.

    Returns
    -------
        Model
    """
    if generators_light is None:
        generators_light = ["EPOSLHC", "SIBYLL", "QGSJET"]
    if generators_heavy is None:
        generators_heavy = ["NLO-P8", "NLO-P8-Max", "NLO-P8-Min"]

    ensure_model_layout(path, link_direct=False)

    model = Model("HNL-e", path=path)
    hnl = HeavyNeutralLepton(ve=ve, vmu=vmu, vtau=vtau)
    hnl.set_generators(
        generators_light=generators_light,
        generators_heavy=generators_heavy,
    )

    for label, pid0, pid1, br, generator, _description in hnl.get_channels_2body():
        model.add_production_2bodydecay(
            label=label,
            pid0=pid0,
            pid1=pid1,
            br=br,
            generator=generator,
            energy=energy,
            nsample=nsample_2body,
        )

    for (
        label, pid0, pid1, pid2, br, generator, integration, _description,
    ) in hnl.get_channels_3body():
        model.add_production_3bodydecay(
            label=label,
            pid0=pid0,
            pid1=pid1,
            pid2=pid2,
            br=br,
            generator=generator,
            energy=energy,
            nsample=nsample_3body,
            integration=integration,
        )

    prev_cwd = os.getcwd()
    os.chdir(path)
    try:
        hnl.get_br_and_ctau()
    finally:
        os.chdir(prev_cwd)

    model.set_ctau_1d(filename="model/ctau/ctau.txt", coupling_ref=1)
    modes, finalstates, filenames = hnl.set_brs()
    model.set_br_1d(modes=modes, finalstates=finalstates, filenames=filenames)

    return model
