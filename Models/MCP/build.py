import numpy as np

from src.foresee import Model
from src.utils.utility import ensure_model_layout
from src.utils.detectors import default_detectors


def br_2body(coef):
    return (
        f"2 * {coef} * coupling**2 * "
        "((1+2*(mass/self.masses('pid0'))**2)*np.sqrt(1-4*(mass/self.masses('pid0'))**2) )"
        "/( (1+2*(self.masses('11')/self.masses('pid0'))**2)*np.sqrt(1-4*(self.masses('11')/self.masses('pid0'))**2) )"
    )


def br_3body(coef):
    return (
        f"2.*{coef} * coupling**2 * 1./137/4./3.1415/q**2 * (1-q**2/self.masses('pid0')**2)**3"
        " * (1-4*mass**2/q**2)**0.5 * (2-(1-4*mass**2/q**2)*np.sin(th)**2)"
    )


def build_model(
    path,
    energy="14",
    nsample_2body=100,    nsample_3body=100,
    generators_light=None,
    brem_configurations=None,
):
    """
    Build and return the FORESEE model object for this model.

    Parameters
    ----------
    path: str
        Model directory (Models/MCP/). Supplied by Foresee.load_model.
    energy: str
        Collider energy in TeV.
    nsample_2body, nsample_3body: int
        Sampling points per meson-decay production channel.
    generators_light: [str]
        Generators for the light-meson channels (rho, omega, phi, pi0, eta,
        eta'). Defaults to ["EPOSLHC", "SIBYLL", "QGSJET"].
    brem_configurations: [str]
        Which precomputed bremsstrahlung spectra to consume as the channel's
        variants. Defaults to ["Brem_QRA_L1.5", "Brem_QRA_L1.0", "Brem_QRA_L2.0"].

    Returns
    -------
        Model
    """
    if generators_light is None:
        generators_light = ["EPOSLHC", "SIBYLL", "QGSJET"]

    if brem_configurations is None:
        brem_configurations = ["Brem_QRA_L1.5", "Brem_QRA_L1.0", "Brem_QRA_L2.0"]

    ensure_model_layout(path, link_direct=True)

    model = Model("MCP", path=path)

    for pid0, coef in [("113", "4.72e-5"), ("223", "7.36e-5"), ("333", "2.97e-4")]:
        model.add_production_2bodydecay(
            pid0=pid0, pid1="0", br=br_2body(coef),
            generator=generators_light, energy=energy, nsample=nsample_2body,
        )
    for pid0, coef in [
        ("443", "0.0597"), ("100443", "0.00993"),
        ("553", "0.0238"), ("100553", "0.0191"), ("200553", "0.0218"),
    ]:
        model.add_production_2bodydecay(
            pid0=pid0, pid1="0", br=br_2body(coef),
            generator=["Pythia8-Monash"], energy=energy, nsample=nsample_2body,
        )

    for pid0, coef in [("111", "0.99"), ("221", "0.39"), ("331", "0.023")]:
        model.add_production_3bodydecay(
            pid0=pid0, pid1="22", pid2="0", br=br_3body(coef),
            generator=generators_light, energy=energy, nsample=nsample_3body,
        )

    model.add_production_direct(
        label="DY",
        energy=energy,
        coupling_ref=1,
        masses=[
            1.001, 1.122, 1.258, 1.412, 1.501, 1.541, 1.549, 1.584, 1.778, 1.801,
            1.821, 1.841, 1.844, 1.995, 2.238, 2.511, 2.818, 3.162, 3.548, 3.981,
            4.466, 4.601, 4.666, 4.701, 4.731, 4.901, 5.011, 5.012, 5.101, 5.171,
            5.178, 5.623, 6.309, 7.079, 7.943, 8.912, 10.01, 11.22, 12.58, 14.12,
            15.84, 17.78, 19.95, 22.38, 25.11, 28.18, 31.62, 35.48, 39.81, 44.66,
            50.11, 56.23, 63.09, 70.79, 79.43, 89.12, 100.01,
        ],
    )

    model.add_production_direct(
        label="Brem",
        energy=energy,
        configuration=brem_configurations,
        coupling_ref=1,
        masses=[round(x, 4) for x in np.logspace(-2, 0.6, 27)],
    )

    model.set_dsigma_drecoil_1d(
        dsigma_der="2.0*3.1415/(137.**2)/self.masses('11') * (1/recoil**2 - mass**2 / (2*self.masses('11')*recoil*energy**2))",
        recoil_max="2 * self.masses('11') * (energy**2-mass**2) / (self.masses('11')*(2*energy+mass) + mass**2)",
        coupling_ref=1,
    )

    return model
