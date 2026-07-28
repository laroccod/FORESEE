import numpy as np

import src.foresee as foresee_module
from src.foresee import Model
from src.utils.utility import ensure_model_layout
from src.utils.detectors import default_detectors


def th_aP(self, mass, coupling, pid0):
    v = 246.22
    cu, cd, cs, cc, cb, ct = 0.873891, 0.986109, 0.986109, 0.873891, 1.04676, 0.906242
    a0, a1, a2, a3 = -25.490, 49.019, -29.482, 5.702
    mPi, mEta, mEtap, fpi, delta = (
        self.masses('111'), self.masses('221'), self.masses('331'), 0.093, 0.370,
    )
    F = lambda m: 1 if m <= 1.4 else (a0 + a1*m + a2*m**2 + a3*m**3) if 1.4 < m <=2 else (1.4/m)**4
    if abs(int(pid0)) == 111:
        pf = fpi*F(mass)*mass**2*(coupling/(2*v)) / (mass**2 - mPi**2)
        return pf*( cd - cu + (delta*mPi**2/3)*(mass**2*(cd+2*cs+cu)/(mass**2 - mEtap**2) + 2*mass**2*(cu+cd-cs)/(mass**2-mEta**2)))
    elif abs(int(pid0)) == 221:
        pf = np.sqrt(2/3)*fpi*F(mass)*mass**2*(coupling/(2*v)) / (mass**2 - mEta**2)
        return pf*( cu+cd-cs - delta*mPi**2*(cu-cd)/(mPi**2 - mass**2))
    elif abs(int(pid0)) == 331:
        pf = np.sqrt(1/3)*fpi*F(mass)*mass**2*(coupling/(2*v)) / (mass**2 - mEtap**2)
        return pf*( -1*(cd+2*cs+cd) - delta*mPi**2*(cu-cd)/(mPi**2 - mass**2))


foresee_module.th_aP = th_aP


BR_B = (
    "(1.8e-3*coupling/(2*246.22))**2*"
    "(self.masses('pid0')**2 - mass**2)/"
    "(32*np.pi*self.masses('pid0')**3*self.widths('pid0'))"
)

DECAY_MODES = [
    "2e", "2mu", "2tau", "2gamma", "2PichargedPi0", "gamma2Picharged",
    "2Picharged2Pi0", "2Piplus2Piminus", "Eta2Pi0", "Eta2Picharged",
    "2Kstarcharged", "2Kstar0", "Omega2Picharged", "3pi0", "Etapr2Pi0",
    "Etapr2Picharged", "2omega", "Jets-GG", "Jets-cc", "Jets-ss",
    "2KLPi0", "2KSPi0", "KLKSpi0", "2Kstar0", "KminusKLPiplus",
    "KplusKLPiminus", "KminusKSPiplus", "KplusKSPiminus",
    "2Kstarcharged", "2KchargedPi0", "2rho0", "2rhocharged",
]


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
        Model directory (Models/ALP-fermion/). Supplied by Foresee.load_model.
    energy: str
        Collider energy in TeV. Notebook default is "14" (HL-LHC).
    nsample_2body: int
        Sampling per B-decay channel. Mixing channels do not consume nsample.
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

    model = Model("ALP-fermion", path=path)

    for pid0 in ['511', '521', '531', '541']:
        model.add_production_2bodydecay(
            pid0=pid0,
            pid1='3',
            br=BR_B,
            generator=generators_heavy,
            energy=energy,
            nsample=nsample_2body,
        )
        model.add_production_2bodydecay(
            pid0=f"-{pid0}",
            pid1='-3',
            br=BR_B,
            generator=generators_heavy,
            energy=energy,
            nsample=nsample_2body,
        )

    for pid in ['111', '221', '331']:
        model.add_production_mixing(
            pid=pid,
            mixing="th_aP(self,mass,coupling,pid0)",
            generator=generators_light,
            energy=energy,
        )

    model.set_ctau_1d(
        filename="model/ctau.csv",
        coupling_ref=2 * 246.22 / 1e6,
    )

    model.set_br_1d(
        modes=DECAY_MODES,
        finalstates=None,
        filenames=["model/br/" + mode + ".csv" for mode in DECAY_MODES],
    )

    return model
