import numpy as np

from src.foresee import Model
from src.utils.utility import ensure_model_layout


def build_model(
    path,
    energy="14",
    nsample_2body=100,
    nsample_3body=100,
    generators_light=None,
):
    """
    Build and return the FORESEE model object for this model.

    Parameters
    ----------
    path: str
        Model directory (Models/EM-EDM/). Supplied by Foresee.load_model.
    energy: str
        Collider energy in TeV. Selects which files/hadrons/<E>TeV.txt.gz
        spectrum table is read.
    nsample_2body: int
        Sampling points per 2-body vector-meson decay channel.
    nsample_3body: int
        Sampling points per 3-body Dalitz decay channel.
    generators_light: [str]
        Hadron generators for the light mesons (rho/omega/phi, pi0/eta/eta').
        Defaults to ['EPOSLHC', 'SIBYLL', 'QGSJET'].

    Returns
    -------
        Model
    """
    if generators_light is None:
        generators_light = ["EPOSLHC", "SIBYLL", "QGSJET"]

    ensure_model_layout(path, link_direct=True)

    model = Model("EDM", path=path)

    vector_meson_br = (
        "2 * {ratio} * coupling**2 *(296.342)**2. *({m}**2.)/(8.*np.pi/137.) "
        "* ((1-4*(mass/{m})**2)*np.sqrt(1-4*(mass/{m})**2) )"
        "/( (1+2*(0.000511/{m})**2)*np.sqrt(1-4*(0.000511/{m})**2) )"
    )
    twobody = [
        ("113",    "4.72e-5", "0.77545",  generators_light),
        ("223",    "7.36e-5", "0.78266",  generators_light),
        ("333",    "2.97e-4", "1.019461", generators_light),
        ("443",    "0.0597",  "3.096",    "Pythia8-Monash"),
        ("100443", "0.00993", "3.686",    "Pythia8-Monash"),
        ("553",    "0.0238",  "9.460",    "Pythia8-Monash"),
        ("100553", "0.0191",  "10.023",   "Pythia8-Monash"),
        ("200553", "0.0218",  "10.355",   "Pythia8-Monash"),
    ]
    for pid0, ratio, mres, gens in twobody:
        model.add_production_2bodydecay(
            pid0=pid0,
            pid1="0",
            br=vector_meson_br.format(ratio=ratio, m=mres),
            generator=gens,
            energy=energy,
            nsample=nsample_2body,
        )

    dalitz_br = (
        "2.*{ratio} * coupling**2 *(296.342)**2. * 1./16./(np.pi**2.) "
        "* (1-q**2/{m}**2)**3 * (1-4*mass**2/q**2)**0.5 "
        "* (1-4*mass**2/q**2)*np.sin(th)**2"
    )
    threebody = [
        ("111", "0.99",  "0.135"),
        ("221", "0.39",  "0.547"),
        ("331", "0.023", "0.957"),
    ]
    for pid0, ratio, mres in threebody:
        model.add_production_3bodydecay(
            pid0=pid0,
            pid1="22",
            pid2="0",
            br=dalitz_br.format(ratio=ratio, m=mres),
            generator=generators_light,
            energy=energy,
            nsample=nsample_3body,
        )

    masses_dy = [
        1.001, 1.122, 1.258, 1.412, 1.501, 1.541, 1.549, 1.584, 1.778, 1.801,
        1.821, 1.841, 1.844, 1.995, 2.238, 2.511, 2.818, 3.162, 3.548, 3.981,
        4.466, 4.601, 4.666, 4.701, 4.731, 4.901, 5.011, 5.012, 5.101, 5.171,
        5.178, 5.623, 6.309, 7.079, 7.943, 8.912, 10.01, 11.22, 12.58, 14.12,
        15.84, 17.78, 19.95, 22.38, 25.11, 28.18, 31.62, 35.48, 39.81, 44.66,
        50.11, 56.23, 63.09, 70.79, 79.43, 89.12, 100.01,
    ]
    model.add_production_direct(
        label="DY",
        energy=energy,
        coupling_ref=1,
        masses=masses_dy,
    )

    model.set_dsigma_drecoil_1d(
        dsigma_der="296.342**2/137./energy**2 * (energy**2/recoil -mass**2/(2*0.000511))",
        recoil_max="2 * 0.000511 * (energy**2-mass**2) / (0.000511*(2*energy+mass) + mass**2)",
        coupling_ref=1,
    )

    return model
