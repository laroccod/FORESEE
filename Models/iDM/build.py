import os
import sys

from src.foresee import Model
from src.utils.utility import ensure_model_layout
from src.utils.detectors import default_detectors

IDM_PKG = os.path.dirname(os.path.abspath(__file__))
if IDM_PKG not in sys.path:
    sys.path.insert(0, IDM_PKG)
from IDMCalc import InelasticDarkMatter  


def build_model(
    path,
    energy="14",
    alphaD=0.1,
    delta=0.1,
    r=3,
    nsample_3body=100,
    nsample_direct = 10,
    generators_light=None,
    brem_configurations=None,
):
    """
    Build and return the FORESEE model object for this model.

    Parameters
    ----------
    path: str
        Model directory (Models/iDM/). Supplied by Foresee.load_model.
    energy: str
        Collider energy in TeV.
    alphaD: float
        Dark fine-structure constant.
    delta: float
        Fractional mass splitting (m_2 - m_1)/m_1.
    r: int or float
        Mass ratio m_{A'}/m_1. (alphaD, delta, r) = (0.1, 0.1, 3) is benchmark
        BP1; (0.1, 1.9, 4) is BP2.
    nsample_3body: int
        Sampling points per light-meson production channel.
    nsample_direct: int
        Sampling points for direct production channels.
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

    model = Model("iDM", path=path)
    idm = InelasticDarkMatter(alphaD=alphaD, delta=delta, r=r)

    massap = "mass*" + str(r) + "/(1+" + str(delta) + ")"

    model.add_production_3bodydecay(
        pid0="111",
        pid1="22",
        pid2="0",
        integration="chain_decay",
        br=[
            "2.*0.99*coupling**2*pow(1.-pow(" + massap + "/self.masses(111),2),3)",
            massap,
        ],
        generator=generators_light,
        energy=energy,
        nsample=nsample_3body,
        massrange=[0, model.masses(111) * (1 + delta) / r],
        scaling=2,
    )
    model.add_production_3bodydecay(
        pid0="221",
        pid1="22",
        pid2="0",
        integration="chain_decay",
        br=[
            "2.*0.39*coupling**2*pow(1.-pow(" + massap + "/self.masses(221),2),3)",
            massap,
        ],
        generator=generators_light,
        energy=energy,
        nsample=nsample_3body,
        massrange=[0, model.masses(221) * (1 + delta) / r],
        scaling=2,
    )
    model.add_production_3bodydecay(
        pid0="331",
        pid1="22",
        pid2="0",
        integration="chain_decay",
        br=[
            "2.*0.023*coupling**2*pow(1.-pow(" + massap + "/self.masses(331),2),3)",
            massap,
        ],
        generator=generators_light,
        energy=energy,
        nsample=nsample_3body,
        massrange=[0, model.masses(331) * (1 + delta) / r],
        scaling=2,
    )

    idm.obtain_ctau_br()
    idm.obtain_direct_production(energy=energy, nsample=10)
    model.add_production_direct(
        label="Brem",
        configuration=brem_configurations,
        energy=energy,
        coupling_ref=1,
        masses=idm.get_brem_masses(),
    )
    model.add_production_direct(
        label="DY",
        energy=energy,
        coupling_ref=1,
        masses=idm.get_dy_masses(),
        condition="True",
    )

    model.set_ctau_1d(filename="model/ctau.txt")

    return model
