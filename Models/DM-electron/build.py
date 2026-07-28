import numpy as np

from src.foresee import Model
from src.utils.utility import ensure_model_layout
from src.utils.detectors import default_detectors
from src.utils.utility import BREM_MASSES


def build_model(
    path,
    energy="14",
    nsample_3body=100,
    generators_light=None,
    brem_configurations=None,
):
    """
    Build and return the FORESEE model object for this model.

    Parameters
    ----------
    path: str
        Model directory (Models/DM-electron/). Supplied by Foresee.load_model.
    energy: str
        Collider energy in TeV.
    nsample_3body: int
        Sampling points per production channel.
    generators_light: [str]
        Hadron generators for light-meson (pi0, eta) production. Defaults to ['EPOSLHC', 'SIBYLL', 'QGSJET'].
    brem_configurations: [str]
        Bremsstrahlung spectrum columns to consume as the channel's variants.
        Defaults to ["Brem_QRA_L1.5", "Brem_QRA_L1.0", "Brem_QRA_L2.0"].

    Returns
    -------
        Model
    """
    if generators_light is None:
        generators_light = ["EPOSLHC", "SIBYLL", "QGSJET"]
    if brem_configurations is None:
        brem_configurations = ["Brem_QRA_L1.5", "Brem_QRA_L1.0", "Brem_QRA_L2.0"]

    ensure_model_layout(path, link_direct=True)

    model = Model("DM-electron", path=path)

    model.add_production_3bodydecay(
        pid0="111",
        pid1="22",
        pid2="0",
        br=[
            "2*2.*0.99 * coupling**2 * pow(1.-pow(3*mass/self.masses('111'),2),3)",
            "3*mass",
        ],
        generator=generators_light,
        energy=energy,
        nsample=nsample_3body,
        integration="chain_decay",
        massrange=[0, 0.135 / 3.0],
    )
    model.add_production_3bodydecay(
        pid0="221",
        pid1="22",
        pid2="0",
        br=[
            "2*2.*0.39 * coupling**2 * pow(1.-pow(3*mass/self.masses('221'),2),3)",
            "3*mass",
        ],
        generator=generators_light,
        energy=energy,
        nsample=nsample_3body,
        integration="chain_decay",
        massrange=[0, 0.547862 / 3.0],
    )

    masses_brem = [round(m / 3, 6) for m in BREM_MASSES]
    model.add_production_direct(
        label="Brem",
        energy=energy,
        configuration=brem_configurations,
        coupling_ref=1,
        masses=masses_brem,
    )

    masses_dy = [
        0.5283, 0.592767, 0.6651, 0.746233, 0.8373, 0.939467, 1.0541,
        1.327033, 1.670633, 2.1032, 2.647767, 3.333333, 4.0, 5.0,
        5.666667, 6.666667, 8.333333, 10.0, 16.666667, 23.333333, 33.333333,
    ]
    model.add_production_direct(
        label="DY",
        energy=energy,
        coupling_ref=1,
        masses=masses_dy,
    )

    model.set_dsigma_drecoil_1d(
        dsigma_der=(
            "coupling**2 * 8 * 3.1415 * 1./137. * 0.5 * 0.000511 "
            "/ (mass**2 * 3**2 +2*0.000511*recoil)**2"
        ),
        recoil_max=(
            "2 * 0.000511 * (energy**2-mass**2) "
            "/ (0.000511*(2*energy+mass) + mass**2)"
        ),
        coupling_ref=1,
    )

    return model
