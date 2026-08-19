import numpy as np

from src.foresee import Model
from src.utils.utility import ensure_model_layout


def br_single(mres, wpid):
    return (
        f"9.68544e-13 * self.masses('{mres}')/ self.widths('{wpid}') "
        f"* (1-(mass/self.masses('{mres}'))**2)**2  * (-3/4/coupling -0.1415/coupling**3 "
        f"- (1.0386/coupling-0.1415/coupling**3)*np.sqrt(1-1/coupling**2)-0.608446664/coupling**3)**2 "
    )


def br_di(wpid):
    return (
        f"(5.49809027901328e-20) * ( self.masses('5')**3 / self.widths('{wpid}') ) "
        f"* np.sqrt(1-4*mass**2/q**2) * (1-q**2/self.masses('5')**2)**2"
    )


def build_model(path, energy="14", nsample_2body=100, nsample_3body=100, generators_heavy=None):
    """
    Build and return the FORESEE model object for this model.

    Parameters
    ----------
    path: str
        Model directory (Models/2HDM_H/). Supplied by Foresee.load_model.
    energy: str
        Collider energy in TeV.
    nsample_2body: int
        Sampling points per 2-body production channel.
    nsample_3body: int
        Sampling points per 3-body production channel.
    generators_heavy: [str]
        Generators for the B-meson channels. Defaults to
        ["NLO-P8", "NLO-P8-Max", "NLO-P8-Min"].

    Returns
    -------
        Model
    """
    if generators_heavy is None:
        generators_heavy = ["NLO-P8", "NLO-P8-Max", "NLO-P8-Min"]

    ensure_model_layout(path, link_direct=False)

    model = Model("2HDM_H", path=path)

    b_channels = [("511", "511"), ("511", "-511"), ("521", "521"), ("521", "-521")]

    for mres, pid in b_channels:
        model.add_production_2bodydecay(
            pid0=pid, pid1="321", br=br_single(mres, pid),
            generator=generators_heavy, energy=energy, nsample=nsample_2body, scaling="manual",
        )

    for mres, pid in b_channels:
        model.add_production_3bodydecay(
            label=f"{pid}_di", pid0=pid, pid1="321", pid2="0", br=br_di(pid),
            generator=generators_heavy, energy=energy, nsample=nsample_3body, scaling=0,
        )

    model.set_ctau_2d(filename="model/ctau.txt")
    decay_modes = ["e_e", "mu_mu", "ga_ga"]
    model.set_br_2d(
        modes=decay_modes,
        finalstates=[[11, -11], [13, -13], [22, 22]],
        filenames=["model/br/" + mode + ".txt" for mode in decay_modes],
    )

    return model
