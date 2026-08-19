import numpy as np

from src.foresee import Model
from src.utils.utility import ensure_model_layout


def conversion_factor(mass, coupling, p0):

    alpha, me, ZFe, AFe = 1./137., 0.000511, 26, 56
    a, d  = 111.*ZFe**(-1./3.) / me,  0.164 * AFe**(-2./3.)
    SMXSinIGeV2 = 13311.9696379

    logthmin, logthmax, nlogth = -12, 0, 20
    dlogth = (logthmax-logthmin)/float(nlogth)

    xs_primakoff = 0
    k = p0.e
    for ltheta in np.linspace(logthmin+0.5*dlogth,logthmax-0.5*dlogth,nlogth):
        theta = 10**ltheta
        t = mass**4 / (2.*k**2) + k**2 * theta**2
        if t<7.39*me*me: ff = a*a*t/(1.+a*a*t)
        else: ff = 1./(1.+t/d)
        xs_primakoff += alpha * ZFe**2 / 4. * ff**2 * k**4 * np.sin(theta)**3 *theta / t**2 * dlogth * np.log(10)
    prob = xs_primakoff / SMXSinIGeV2
    if np.isnan(prob): prob=0

    return np.sqrt(prob)


def build_model(
    path,
    energy="14",
    generators_light=None
):
    """
    Build and return the FORESEE model object for this model.

    Parameters
    ----------
    path: str
        Model directory (Models/ALP-photon/). Supplied by Foresee.load_model.
    energy: str
        Collider energy in TeV. Selects which files/hadrons/<E>TeV.txt.gz
        photon spectrum table is read.
    generators_light: [str]
        Photon-flux generators driving the Primakoff conversion. Defaults to ['EPOSLHC', 'SIBYLL', 'QGSJET'].

    Returns
    -------
        Model
    """
    if generators_light is None:
        generators_light = ['EPOSLHC', 'SIBYLL', 'QGSJET']

    ensure_model_layout(path, link_direct=False)

    model = Model("ALP-photon", path=path)

    model.add_production_mixing(
        label='Primakoff',
        pid = "22",
        mixing = conversion_factor,
        generator=generators_light,
        energy = energy,
    )

    model.set_ctau_1d(
        filename="model/ctau.txt",
    )

    decay_modes = ["gamma", "eegamma"]
    model.set_br_1d(
        modes = decay_modes,
        finalstates=[[22,22], [11,-11,22]],
        filenames=["model/br/"+mode+".txt" for mode in decay_modes],
    )

    return model
