"""Builder for the HNL-tau (tau-flavor Heavy Neutral Lepton) model.

Same builder pattern as HNL-e / HNL-mu (see ../HNL-e/build.py for the canonical
doc). The flavor mix defaults to (ve, vmu, vtau) = (0, 0, 1), which unlocks an
additional production category, tau decays tau -> h^0 N, that doesn't appear for
the lighter-flavor HNLs. The reach-plot ylims and branching-curve labels are
tau-specific.

Invoked by Foresee.load_model("HNL-tau", **params) or
Foresee.load_model("HNL/HNL-tau", **params).
"""

import os
import sys

import numpy as np

from src.foresee import Model
from src.utils.utility import ensure_model_layout
from src.utils.detectors import default_detectors

HNL_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HNL_PKG not in sys.path:
    sys.path.insert(0, HNL_PKG)
from HNLCalc import HeavyNeutralLepton  # noqa: E402


def build_model(
    path,
    energy="13.6",
    nsample_2body=100,    nsample_3body=100,
    generators_light=None,
    generators_heavy=None,
    ve=0,
    vmu=0,
    vtau=1,
):
    """
    Build the HNL-tau Model.

    Parameters
    ----------
    path: str
        Model directory (Models/HNL/HNL-tau/). Supplied by Foresee.load_model.
    energy: str
        Collider energy in TeV.
    nsample_2body, nsample_3body: int
        Sampling points per production channel.
    generators_light: [str]
        Defaults to ['EPOSLHC', 'SIBYLL', 'QGSJET'].
    generators_heavy: [str]
        Defaults to ['NLO-P8', 'NLO-P8-Max', 'NLO-P8-Min'].
    ve, vmu, vtau: float
        Active-neutrino mixing components. Defaults (0, 0, 1) realize a pure
        tau-flavor HNL.

    Returns
    -------
        Model
    """
    if generators_light is None:
        generators_light = ["EPOSLHC", "SIBYLL", "QGSJET"]
    if generators_heavy is None:
        generators_heavy = ["NLO-P8", "NLO-P8-Max", "NLO-P8-Min"]

    ensure_model_layout(path, link_direct=False)

    model = Model("HNL-tau", path=path)
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


def build_presets(
    model,
    *,
    energy="13.6",
    generators_light=None,
    generators_heavy=None,
    **_,
):
    """
    Plot/scan defaults for the HNL-tau signature notebooks.

    Lifted from HNL-tau.ipynb. Differs from HNL-e/HNL-mu in three places:
    production_channels includes a tau -> h^0 N category whose channel keys
    contain "2body_tau" (no PDG-digit prefix); reach_plot["ylims"] is loosened
    to [1e-4, 1e-1]; and the branching-curve list and literal bounds overlay
    reflect tau-final-state modes. Returns a dict consumed by the signature
    notebooks; see the Foresee.load_model docstring for the schema.

    Parameters
    ----------
    model: Model
        The configured Model returned by build_model.
    energy: str
        Collider energy in TeV.
    generators_light, generators_heavy: [str]
        Hadron generators, defaulting to the build_model defaults.
    """
    if generators_light is None:
        generators_light = ["EPOSLHC", "SIBYLL", "QGSJET"]
    if generators_heavy is None:
        generators_heavy = ["NLO-P8", "NLO-P8-Max", "NLO-P8-Min"]

    visible_modes = [m for m in model.br_finalstate.keys() if m != ("nu", "nu", "nu")]

    # Group production channels by parent hadron for the production-rate plot
    # (mirrors the research notebook). Each filter is anchored to the parent's
    # position in the channel label so a hadron appearing as a decay daughter
    # is not miscounted as the parent.
    keys = list(model.production.keys())
    production_channels = [
        {"channels": [k for k in keys if any(s in k for s in ["3body_pseudo_421","3body_pseudo_-421","3body_vector_421","3body_vector_-421"])], "color": "deepskyblue", "label": r"$D^0$", "generators": generators_heavy},
        {"channels": [k for k in keys if any(s in k for s in ["2body_411","2body_-411","3body_pseudo_411","3body_pseudo_-411","3body_vector_411","3body_vector_-411"])], "color": "blue", "label": r"$D^\pm$", "generators": generators_heavy},
        {"channels": [k for k in keys if any(s in k for s in ["2body_431","2body_-431","3body_pseudo_431","3body_pseudo_-431","3body_vector_431","3body_vector_-431"])], "color": "dodgerblue", "label": r"$D_s^\pm$", "generators": generators_heavy},
        {"channels": [k for k in keys if any(s in k for s in ["2body_521","2body_-521","3body_pseudo_521","3body_pseudo_-521","3body_vector_521","3body_vector_-521"])], "color": "purple", "label": r"$B^\pm$", "generators": generators_heavy},
        {"channels": [k for k in keys if any(s in k for s in ["3body_pseudo_511","3body_pseudo_-511","3body_vector_511","3body_vector_-511"])], "color": "magenta", "label": r"$B^0$", "generators": generators_heavy},
        {"channels": [k for k in keys if any(s in k for s in ["3body_pseudo_531","3body_pseudo_-531","3body_vector_531","3body_vector_-531"])], "color": "violet", "label": r"$B_s^0$", "generators": generators_heavy},
        {"channels": [k for k in keys if any(s in k for s in ["2body_541","2body_-541","3body_pseudo_541","3body_pseudo_-541","3body_vector_541","3body_vector_-541"])], "color": "mediumpurple", "label": r"$B_c^\pm$", "generators": generators_heavy},
        {"channels": [k for k in keys if any(s in k for s in ["2body_tau_15","2body_tau_-15","3body_tau_15","3body_tau_-15"])], "color": "green", "label": r"$\tau^\pm$", "generators": generators_heavy},
    ]
    production_channels = [p for p in production_channels if p["channels"]]

    detectors = default_detectors("decay", channels=visible_modes)

    return {
        "benchmark": {"mass": 1.0, "coupling": 1e-3},
        "grid": {
            "masses": np.logspace(-1, np.log10(6.0), 50 ),
            "couplings": np.logspace(-5, 0, 50 ),
        },
        "detectors": detectors,
        "production_channels": production_channels,
        "production_plot": {
            "condition": "logth<-3.7 and logp>2",
            "xlims": [0.1,10],
            "ylims": [4e-4,4e11],
            "xlabel": r"Mass [GeV]",
            "ylabel": r"Production Rate $\sigma/\varepsilon^2$ [pb]",
            "title": r"$\theta < 0.2$ mrad and $E > 100$ GeV",
            "legendloc": (1.01,1),
            "fs_label": 12,
            "fs_label_br": 9,
            "ncol": 3,
            "figsize": (7,6),
        },
        "reach_plot": {
            "title": "HNL's",
            "xlims": [0.1, 10],
            "ylims": [1e-4, 1e-1],
            "xlabel": r"$m_{N}$ (GeV)",
            "ylabel": r"$\epsilon$",
            "legendloc": (1,0.2),
            "linewidths": 2,
        },
        "bounds": [
            ["bounds_001/bounds_cosmo.txt",        "BBN",               1.1e-1,  0.00264,    -20],
            ["bounds_001/bounds_bebc.txt",         "BEBC",              1,       0.000775,     0],
            ["bounds_001/bounds_charm.txt",        "CHARM",             1.27e-1, 0.00632*1.8, -25],
            ["bounds_001/bounds_delphi_long.txt",  "Delphi \n (long)",  2,       0.007,        0],
            ["bounds_001/bounds_delphi_short.txt", "Delphi \n (short)", 4,       0.01,         0],
            ["bounds_001/bounds_babar.txt",        "BaBar",             1.34*.7, 0.00224,     -85],
            ["bounds_001/bounds_argoneut.txt",     "Argoneut",          3.5e-1,  0.0224*1.3,  -49],
        ],
        "projections": [],
        "branchings": [
            [("nu", "nu", "nu"),          "black",  "solid", r"$\nu \nu \nu$",          0.17, 0.06  ],
            [("nu", "e", "anti_e"),       "red",    "solid", r"$\nu e^+ e^-$",          0.17, 0.02  ],
            [("nu", "mu", "anti_mu"),     "orange", "solid", r"$\nu \mu^+ \mu^-$",      1.1,  2e-2  ],
            [("nu", "pi0"),               "blue",   "solid", r"$\nu \pi^0$",            0.18, 0.45  ],
            [("mu", "anti_tau", "nu"),    "purple", "solid", r"$\nu \mu^\pm \tau^\mp$", 6.1,  0.02  ],
            [("tau", "anti_mu", "nu"),    "purple", "None",  None,                      5.0,  0.025 ],
            [("tau", "rho+"),             "brown",  "solid", r"$\tau^\mp \rho^\pm $",   2,    2.5e-2],
            [("anti_tau", "anti_rho+"),   "brown",  "None",  None,                      0.2,  0.13  ],
            [("nu", "rho0"),              "teal",   "solid", r"$\nu \rho^0 $",          1.1,  5.4e-2],
        ],
    }
