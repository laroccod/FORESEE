import numpy as np

from src.foresee import Model
from src.utils.utility import ensure_model_layout


def meson_br(mmeson, coef=None):
    lead = mmeson if coef is None else f"{coef} * {mmeson}"
    m = mmeson
    return (
        f"coupling**2 * {lead} / (8 * 3.1415**2 * 0.105**2 * ({m}**2-0.105**2)**2 *(q2-0.105**2)**2)"
        f" * (({m}**2 - 2*{m}*energy+q2)*q2*(q2-0.105**2) - (q2**2-0.105**2*{m}**2)*(q2+0.105**2-mass**2)"
        f" + 2*0.105**2*q2*({m}**2-q2))"
    )


def build_model(
    path,
    energy="14",
    nsample_3body=100,
    generators_light=None,
    generators_heavy=None,
):
    """
    Build and return the FORESEE model object for this model.

    Parameters
    ----------
    path: str
        Model directory (Models/Muonphilic/). Supplied by Foresee.load_model.
    energy: str
        Collider energy in TeV.
    nsample_3body: int
        Sampling points per 3-body production channel (pi/K and charm D/D_s,
        all M -> mu nu S).
    generators_light: [str]
        Generators for pi/K production. Defaults to ["EPOSLHC", "SIBYLL", "QGSJET"].
    generators_heavy: [str]
        Generators for D/D_s production. Defaults to ["NLO-P8", "NLO-P8-Max", "NLO-P8-Min"].

    Returns
    -------
        Model
    """
    if generators_light is None:
        generators_light = ["EPOSLHC", "SIBYLL", "QGSJET"]
    if generators_heavy is None:
        generators_heavy = ["NLO-P8", "NLO-P8-Max", "NLO-P8-Min"]

    ensure_model_layout(path, link_direct=False)

    model = Model("Muonphilic", path=path)

    br_K = meson_br("0.494", coef="0.6456")
    br_pi = meson_br("0.13957")
    br_D = meson_br("1.86961", coef="3.7e-4")
    br_Ds = meson_br("1.96830", coef="5.4e-3")

    channels = [
        ("321", "14", "-13", br_K, generators_light, nsample_3body),
        ("-321", "-14", "13", br_K, generators_light, nsample_3body),
        ("211", "14", "-13", br_pi, generators_light, nsample_3body),
        ("-211", "-14", "13", br_pi, generators_light, nsample_3body),
        ("411", "14", "-13", br_D, generators_heavy, nsample_3body),
        ("-411", "-14", "13", br_D, generators_heavy, nsample_3body),
        ("431", "14", "-13", br_Ds, generators_heavy, nsample_3body),
        ("-431", "-14", "13", br_Ds, generators_heavy, nsample_3body),
    ]
    for pid0, pid1, pid2, br, generators, ns in channels:
        model.add_production_3bodydecay(
            pid0=pid0,
            pid1=pid1,
            pid2=pid2,
            br=br,
            generator=generators,
            energy=energy,
            nsample=ns,
            integration="dq2dE",
        )

    model.set_ctau_1d(filename="model/ctau.txt", coupling_ref=1)
    model.set_br_1d(
        modes=["gamma_gamma", "mu_mu"],
        finalstates=[[22, 22], [13, -13]],
        filenames=["model/br/gamma_gamma.txt", "model/br/mu_mu.txt"],
    )

    return model
