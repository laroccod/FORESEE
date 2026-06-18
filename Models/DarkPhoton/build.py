"""Builder for the Dark Photon model.

Returns a fully-configured Model with kinetic-mixing dark photon production
(light-meson decays, bremsstrahlung, and Drell-Yan), decay-mode branching
fractions, and lifetime. Equivalent to running the setup cells of
DarkPhoton.ipynb up to the foresee.set_model(...) call. A few sub-leading
channels (charged-rho decays, resonant vector-meson mixing) are mirrored from
the notebook as commented-out blocks and can be enabled if needed.

Invoked by Foresee.load_model("DarkPhoton", **params). Default kwargs reproduce
the existing notebook bit-for-bit.
"""

import numpy as np

from src.foresee import Model
from src.utils.utility import ensure_model_layout
from src.utils.detectors import default_detectors
from src.utils.utility import BREM_MASSES


def build_model(
    path,
    energy="14",
    nsample_2body=100,
    generators_light=None,
    generators_heavy=None,
    brem_configurations=None,
):
    """
    Build the DarkPhoton Model.

    Parameters
    ----------
    path: str
        Model directory (Models/DarkPhoton/). Supplied by Foresee.load_model.
    energy: str
        Collider energy in TeV. Selects which files/hadrons/<E>TeV.txt.gz
        spectrum table is read. Common values: "14", "13.6", "27", "100".
    nsample_2body: int
        Sampling points per production channel.
    generators_light: [str]
        Hadron generators for light-meson production. Defaults to
        ['EPOSLHC', 'SIBYLL', 'QGSJET'], the canonical triple used for
        shower-model uncertainty envelopes.
    generators_heavy: [str]
        Accepted for API uniformity across builders; DarkPhoton has no
        heavy-meson production channels, so this argument is unused.
    brem_configurations: [str]
        Which precomputed bremsstrahlung spectra (columns under
        files/direct/DarkPhoton/<E>TeV.txt.gz) to consume as the channel's
        variants. Defaults to ["Brem_QRA_L1.5", "Brem_QRA_L1.0", "Brem_QRA_L2.0"],
        the QRA pt-cut envelope (L1.5 is the central choice).

    Returns
    -------
        Model
    """
    if generators_light is None:
        generators_light = ["EPOSLHC", "SIBYLL", "QGSJET"]
    if brem_configurations is None:
        brem_configurations = ["Brem_QRA_L1.5", "Brem_QRA_L1.0", "Brem_QRA_L2.0"]

    ensure_model_layout(path, link_direct=True)

    model = Model("DarkPhoton", path=path)

    # Pseudoscalar-meson radiative decays: P -> A' gamma  for  P = pi0, eta, eta'.
    pseudoscalar_br = "2.*{coeff} * coupling**2 * (1-mass**2/self.masses(pid0)**2)**3 "
    for pid0, coeff in [("111", "0.99"), ("221", "0.39"), ("331", "0.023")]:
        model.add_production_2bodydecay(
            pid0=pid0,
            pid1="22",
            br=pseudoscalar_br.format(coeff=coeff),
            generator=generators_light,
            energy=energy,
            nsample=nsample_2body,
        )

    # Vector-meson decays  V -> A' P.
    vector_meson_br = (
        " coupling**2 * {prefactor} * ("
        "(self.masses('pid0')**2 - (self.masses('pid1') + mass)**2)*"
        "(self.masses('pid0')**2 - (self.masses('pid1') - mass)**2)"
        "/(self.masses('pid0')**2 - self.masses('pid1')**2)**2"
        ")**(3/2)"
    )
    for pid0, pid1, label, prefactor in [
        ("113", "111", "113_111", "4.7e-4"),
        ("113", "221", "113_221", "3.0e-4"),
        ("223", "111", "223_111", "8.33e-2"),
        ("223", "221", "223_221", "4.5e-4"),
        ("333", "221", None,      "1.306e-2"),
    ]:
        kwargs = dict(
            pid0=pid0, pid1=pid1,
            br=vector_meson_br.format(prefactor=prefactor),
            generator=generators_light,
            energy=energy,
            nsample=nsample_2body,
        )
        if label is not None:
            kwargs["label"] = label
        model.add_production_2bodydecay(**kwargs)

    # Optional production channels the notebook keeps commented out, mirrored
    # here (commented) so build.py matches DarkPhoton.ipynb and they can be
    # enabled easily:
    #
    #   Charged-rho decays  rho^+- -> A' pi^+- :
    # model.add_production_2bodydecay(
    #     pid0="213", pid1="211",    # rho+, pi+
    #     br=vector_meson_br.format(prefactor="4.5e-4"),
    #     generator=generators_light, energy=energy, nsample=nsample_2body)
    # model.add_production_2bodydecay(
    #     pid0="-213", pid1="-211",  # rho-, pi-
    #     br=vector_meson_br.format(prefactor="4.5e-4"),
    #     generator=generators_light, energy=energy, nsample=nsample_2body)
    #
    #   Resonant mixing with the QCD vector mesons rho/omega/phi (arXiv:1810.01879):
    # for pid, g_v in [("113", "5."), ("223", "17."), ("333", "(-12.88)")]:
    #     model.add_production_mixing(
    #         pid=pid,
    #         mixing=f"coupling * 0.3/{g_v} * self.masses('pid')**2"
    #                "/abs(mass**2 - self.masses('pid')**2 + self.masses('pid')*self.widths('pid')*1j)",
    #         generator=generators_light, energy=energy)

    # Bremsstrahlung (precomputed spectra under model/direct/). The mass grid is
    # shared with the U(1) vector models -- see BREM_MASSES in src/utils/utility.py.
    model.add_production_direct(
        label="Brem",
        energy=energy,
        configuration=brem_configurations,
        coupling_ref=1,
        masses=BREM_MASSES,
    )

    # Drell-Yan (qq -> A' X), dominant at high mass (arXiv:1810.01879). The
    # spectra ship only at 27/100 TeV, so this channel reads empty at 13.6/14 TeV
    # and the fixed-target beams and contributes only when the model
    # is built at 27 or 100 TeV.
    masses_dy = [1.5849, 1.7783, 1.9953, 2.2387, 2.5119, 2.8184,
                 3.1623, 3.9811, 5.0119, 6.3096, 7.9433, 10.]
    model.add_production_direct(
        label="DY",
        energy=energy,
        coupling_ref=1,
        masses=masses_dy,
        condition="True",
    )

    # Lifetime and branching fractions (DeLiVeR, arXiv:2201.01788).
    model.set_ctau_1d(filename="model/ctau.txt")

    # Decay final states as (mode, PDG IDs) pairs (DeLiVeR, arXiv:2201.01788).
    # Pairing avoids the parallel-list shift that misaligned the hadronic final
    # states. Each mode loads model/br/bfrac_<mode>.txt, including the
    # charge-specific hadronic sub-channels (KK_c, KK_n, 4pi_c, ...). 
    decay_channels = [
        ("elec", [11, -11]), ("muon", [13, -13]),
        ("2pi",          [211, -211]),
        ("3pi",          [211, -211, 111]),
        ("PiGamma",      [111, 22]),
        ("EtaGamma",     [221, 22]),
        ("EtaOmega",     [221, 223]),
        ("EtaPhi",       [221, 333]),
        ("PhiPi",        [333, 111]),
        ("OmegaPion",    [223, 111]),
        ("EtaPiPi",      [221, 211, -211]),
        ("EtaPrimePiPi", [331, 211, -211]),
        ("4pi_c",        None),              # 2pi+ 2pi-
        ("4pi_n",        None),              # pi+ pi- 2pi0
        ("6pi_c",        None),              # 3pi+ 3pi-
        ("6pi_n",        None),              # 2pi+ 2pi- 2pi0
        ("KKpipi_0",     None),              # K+ pi- K- pi+
        ("KKpipi_1",     None),              # KS pi0 K+- pi-+
        ("KKpipi_2",     None),              # K+- pi0 KS pi-+
        ("KKpipi_3",     None),              # KS pi-+ K-+ pi0
        ("KK_c",         [321, -321]),       # K+ K-
        ("KK_n",         [310, 130]),        # K0bar K0 -> K_S K_L
        ("KKpi_0",       [130, 310, 111]),   # K_L K_S pi0
        ("KKpi_1",       [321, -321, 111]),  # K+ K- pi0
        ("KKpi_2",       [321, -211, 311]),  # K+- pi-+ K0 (representative)
        ("OmPiPi_c",     [223, 211, -211]),  # omega pi+ pi-
        ("OmPiPi_n",     [223, 111, 111]),   # omega pi0 pi0
        ("PhiPiPi_c",    [333, 211, -211]),  # phi pi+ pi-
        ("PhiPiPi_n",    [333, 111, 111]),   # phi pi0 pi0
        ("ppbar",        [2212, -2212]),
        ("nnbar",        [2112, -2112]),
    ]
    decay_modes = [mode for mode, _ in decay_channels]
    finalstates = [fs for _, fs in decay_channels]
    filenames = ["model/br/bfrac_" + mode + ".txt" for mode in decay_modes]
    model.set_br_1d(modes=decay_modes, finalstates=finalstates, filenames=filenames)

    return model


def build_presets(model, *, energy="14", generators_light=None, **_):
    """
    Plot/scan defaults for the DarkPhoton signature notebooks.

    Lifted from DarkPhoton.ipynb. Returns a dict consumed by the signature
    notebooks (Examples/decay.ipynb); see the Foresee.load_model docstring for
    the schema.

    Parameters
    ----------
    model: Model
        The configured Model returned by build_model. Unused for DarkPhoton but
        part of the standard signature so model-derived defaults (e.g.
        visible_modes for HNL) work in other builders.
    energy: str
        Collider energy as a string. Controls which detector geometries are
        offered (Run-3 FASER at 13.6 TeV vs. HL-LHC FASER and FASER2 at 14 TeV)
        and is recorded for the notebook's reach-scan loop.
    """
    
    masses = (
        [round(x, 5) for x in np.logspace(-2, -1, 10)]
        + [round(x, 5) for x in np.logspace(-1, np.log10(2.05), 60)]
    )
    masses.sort()

    detectors = default_detectors("decay")

    # One light-meson generator per channel for the production plot legend.
    light_gen = (generators_light or ["EPOSLHC"])[:1]

    return {
        "benchmark": {"mass": 0.05, "coupling": 3e-5},
        "grid": {
            "masses": masses,
            "couplings": np.logspace(-9, -3, 121),
        },
        "detectors": detectors,
        "production_channels": [
            {"channels": ["111"],           "color": "tab:blue",   "label": r"$\pi^0 \to \gamma A'$",  "generators": light_gen},
            {"channels": ["221"],           "color": "tab:orange", "label": r"$\eta \to \gamma A'$",   "generators": light_gen},
            {"channels": ["331"],           "color": "tab:green",  "label": r"$\eta' \to \gamma A'$",  "generators": light_gen},
            {"channels": ["113_111"],       "color": "tab:red",    "label": r"$\rho^0 \to \pi^0 A'$",  "generators": light_gen},
            {"channels": ["113_221"],       "color": "tab:purple", "label": r"$\rho^0 \to \eta A'$",   "generators": light_gen},
            {"channels": ["223_111"],       "color": "tab:brown",  "label": r"$\omega \to \pi^0 A'$",  "generators": light_gen},
            {"channels": ["223_221"],       "color": "tab:pink",   "label": r"$\omega \to \eta A'$",   "generators": light_gen},
            {"channels": ["333"],           "color": "tab:gray",   "label": r"$\phi \to \eta A'$",     "generators": light_gen},
            {"channels": ["Brem"], "color": "tab:olive",  "label": "Bremsstrahlung",          "generators": ["Brem_QRA_L1.5"]},
        ],
        "production_plot": {
            "xlims": [0.01, 2.0],
            "ylims": [2e4, 4e13],
            "xlabel": "Mass [GeV]",
            "ylabel": r"Production Rate $\sigma/\varepsilon^2$ [pb]",
            "title": r"$\theta < 0.2$ mrad and $E > 100$ GeV",
            "condition": "logth<-3.7 and logp>2",
            "legendloc": (1.01, 1),
            "ncol": 3,
            "fs_label": 12,
            "fs_label_br": 9,
            "figsize": (7, 6),
        },
        "reach_plot": {
            "xlims": [0.01, 3],
            "ylims": [2e-8, 0.002],
            "xlabel": r"Dark Photon Mass $m_{A'}$ [GeV]",
            "ylabel": r"Kinetic Mixing $\epsilon$",
            "title": "Dark Photons (BC1)",
            "legendloc": (1.0, 0.7),
            "linewidths": 2,
        },
        "bounds": [
            ["bounds_LSND.txt",          "LSND",                  1e-1,    3e-8,    0],
            ["bounds_E137.txt",          "E137",                  0.015,   9.00e-8, 0],
            ["bounds_NA62ee.txt",        r"NA62 - $ee$",          0.10,    4.00e-6, -25],
            ["bounds_NuCal.txt",         "NuCal",                 0.101,   3.00e-6, -20],
            ["bounds_CHARM.txt",         "CHARM",                 0.101,   1.0e-6,  -20],
            ["bounds_Orsay.txt",         "Orsay",                 3.8e-2,  2.0e-6,  0],
            ["bounds_E141.txt",          "E141",                  0.021,   1.0e-4,  0],
            ["bounds_NA62mumu.txt",      r"NA62 - $\mu\mu$",      0.220,   7.0e-7,  0],
            ["bounds_FASER-27invfb.txt", "FASER '23",             4e-2,    1e-5,    -30],
            ["bounds_FASER-57ifb.txt",   r"FASER '24",            1.6e-2,  0.5e-5,  -10],
            ["bounds_NA48.txt",          "NA48",                  0.015,   1.20e-3, 0],
            ["bounds_LHCb1.txt",         "LHCb",                  0.220,   7.00e-5, 0],
            ["bounds_LHCb2.txt",         None,                    0,       0,       0],
            ["bounds_LHCb3.txt",         None,                    0,       0,       0],
            ["bounds_NA64.txt",          "NA64",                  0.015,   2.40e-4, -30],
            ["bounds_BaBar.txt",         "BaBar",                 0.320,   1.20e-3, 0],
            ["bounds_A1.txt",            "A1",                    0.100,   1.20e-3, 0],
            ["bounds_KLOE.txt",          "KLOE",                  0.620,   1.40e-3, 0],
        ],
        "bounds2": [
            ["bounds_DD.txt", "DM Direct\nDetection", 1.0, 1.0e-4, 0],
        ],
        "projections": [],
        # Scalar-DM relic-density target (arXiv:2105.07077), overlaid on the
        # reach via plot_reach's `lines` hook: [file, color, lw, [[text,x,y,rot],...]].
        "lines": [
            ["scalar_DM_Oh2_intermediate_eps_vs_mAprime.txt", "k", 2, [
                ["relic target",            0.010, 3.00e-5, 20],
                [r"$m_\chi\!=\!0.6 m_{A'}$", 0.011, 2.00e-5, 20],
                [r"$\alpha_D\!=\!0.1$",      0.013, 1.12e-5, 20],
            ]],
        ],
        
        "branchings": [
            ["elec",    "red",        "solid", r"$e^+e^-$",           0.02,  0.6  ],
            ["muon",    "orange",     "solid", r"$\mu^+\mu^-$",       0.14,  0.18 ],
            ["2pi",     "blue",       "solid", r"$\pi^+\pi^-$",       0.32,  0.12 ],
            ["3pi",     "brown",      "solid", r"$\pi^+\pi^-\pi^0$",  0.430, 0.045],
            ["KK_c",    "green",      "solid", r"$K^+K^-$",           1.3,   0.045],
            ["KK_n",    "darkgreen",  "solid", r"$K_S K_L$",          0.86,  0.58 ],
            ["4pi_n",   "purple",     "solid", r"$\pi^+\pi^-2\pi^0$", 1.1,   0.7  ],
            ["4pi_c",   "magenta",    "solid", r"$2\pi^+2\pi^-$",     1.18,  0.4  ],
            ["PiGamma", "dodgerblue", "solid", r"$\pi^0\gamma$",      0.580, 0.02 ],
        ],
    }
