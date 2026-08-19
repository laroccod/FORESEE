import numpy as np

from src.foresee import Model
from src.utils.utility import ensure_model_layout


def build_model(
    path,
    energy="14",
    nsample_2body=100,    nsample_3body=100,
    generators_heavy=None,
):
    """
    Build and return the FORESEE model object for this model.

    Parameters
    ----------
    path: str
        Model directory (Models/DarkHiggs/). Supplied by Foresee.load_model.
    energy: str
        Collider energy in TeV. Selects which files/hadrons/<E>TeV.txt.gz
        spectrum table is read.
    nsample_2body, nsample_3body: int
        Sampling points per production channel.
    generators_heavy: [str]
        Hadron generators for B-meson production. Defaults to ['NLO-P8', 'NLO-P8-Max', 'NLO-P8-Min'].

    Returns
    -------
        Model
    """
    if generators_heavy is None:
        generators_heavy = ["NLO-P8", "NLO-P8-Max", "NLO-P8-Min"]

    ensure_model_layout(path, link_direct=False)

    model = Model("DarkHiggs", path=path)

    for pid0, pid1 in [("511", "130"), ("-511", "130"),
                       ("521", "321"), ("-521", "321")]:
        model.add_production_2bodydecay(
            pid0=pid0,
            pid1=pid1,
            br="5.6 * coupling**2 * pow(1.-pow(mass/self.masses('pid0'),2),2)",
            generator=generators_heavy,
            energy=energy,
            nsample=nsample_2body,
        )

    for label, pid0, pid1 in [("511_di", "511", "130"), ("-511_di", "-511", "130"),
                              ("521_di", "521", "321"), ("-521_di", "-521", "321")]:
        model.add_production_3bodydecay(
            label=label,
            pid0=pid0,
            pid1=pid1,
            pid2="0",
            br="7.37e-10*(self.masses('5')/4.5)**3*np.sqrt(1-4*mass**2/q**2)*(1-q**2/self.masses('5')**2)**2",
            generator=generators_heavy,
            energy=energy,
            nsample=nsample_3body,
            scaling=0,
        )

    model.set_ctau_1d(
        filename="model/ctau.txt",
        coupling_ref=1,
    )

    decay_modes = ["e_e", "mu_mu", "K_K", "pi_pi"]
    model.set_br_1d(
        modes=decay_modes,
        finalstates=[[11, -11], [13, -13], [321, -321], [211, -211]],
        filenames=["model/br/" + mode + ".txt" for mode in decay_modes],
    )

    return model
