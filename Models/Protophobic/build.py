"""Builder for the Protophobic gauge boson model.

Returns a fully-configured Model for a protophobic vector X (couples to
neutrons, not protons): production via the pseudoscalar radiative decays
eta/eta' -> gamma X with vector-meson-dominance mixing, the vector-meson
transitions rho/omega/phi -> P X, and bremsstrahlung, plus the standard
hadronic/leptonic visible decay modes. A custom BW propagator is patched
onto src.foresee for the eta/eta' branching strings. Equivalent to running the
setup cells of Protophobic.ipynb up to the foresee.set_model(...) call.

Invoked by Foresee.load_model("Protophobic", **params).
"""

import numpy as np

from src.foresee import Model
from src.utils.utility import ensure_model_layout, BREM_MASSES
from src.utils.detectors import default_detectors
import src.foresee as foresee_module


# Shared kinematic suffix for the vector-meson V -> P X branchings.
VP_KIN = (
    "((self.masses('pid0')**2 - (self.masses('pid1') + mass)**2)"
    "*(self.masses('pid0')**2 - (self.masses('pid1') - mass)**2)"
    "/(self.masses('pid0')**2 - self.masses('pid1')**2)**2)**(3/2)"
)


def BW(self, mass, pid):
    """
    Breit-Wigner propagator for resonance pid evaluated at mass.

    Used inside the eta/eta' production branching strings (vector-meson
    dominance mixing of the X with the rho/omega/phi). Bound onto the
    src.foresee module so it is in scope when those strings are eval'd.
    """
    return 1 / (1 - (mass / self.masses(pid))**2 - 1j * self.widths(pid) / self.masses(pid))


def build_model(
    path,
    energy="14",
    nsample_2body=100,
    generators_light=None,
    generators_heavy=None,
    brem_configurations=None,
):
    """
    Build the Protophobic Model.

    Parameters
    ----------
    path: str
        Model directory (Models/Protophobic/). Supplied by Foresee.load_model.
    energy: str
        Collider energy in TeV. Defaults to "14".
    nsample_2body: int
        Sampling points per production channel. Defaults to 100.
    generators_light: [str]
        Hadron generators for meson production. Defaults to
        ["EPOSLHC", "SIBYLL", "QGSJET"], the canonical triple for the
        shower-model uncertainty envelope.
    generators_heavy: [str]
        Accepted for API uniformity; Protophobic has no heavy-meson channels.
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

    # The eta/eta' BR strings call BW(self, mass, pid); make it resolvable in
    # the foresee module namespace where those strings are evaluated.
    foresee_module.BW = BW

    model = Model("Protophobic", path=path)

    # ---- Pseudoscalar radiative decays  P -> X gamma (with VMD mixing) ----
    model.add_production_2bodydecay(
        pid0="221",
        pid1="22",
        br="2.*0.39 * (coupling/0.303)**2 * (1-mass**2/self.masses(pid0)**2)**3 * np.abs( (BW(self,mass,'223')  - 9*BW(self,mass,'113')  + 4*BW(self,mass,'333')) / (BW(self,mass,'223')  + 9*BW(self,mass,'113')  - 2*BW(self,mass,'333')) )**2",
        generator=generators_light,
        energy=energy,
        nsample=nsample_2body,
    )
    model.add_production_2bodydecay(
        pid0="331",
        pid1="22",
        br="2.*0.023 * (coupling/0.303)**2 * (1-mass**2/self.masses(pid0)**2)**3 * np.abs( (BW(self,mass,'223')  - 9*BW(self,mass,'113')  - 8*BW(self,mass,'333')) / (BW(self,mass,'223')  + 9*BW(self,mass,'113')  + 4*BW(self,mass,'333')) )**2",
        generator=generators_light,
        energy=energy,
        nsample=nsample_2body,
    )

    # ---- Vector-meson decays  V -> P X ----
    # Each br has a leading space and a per-channel prefactor, then the shared
    # kinematic suffix; preserved verbatim (whitespace is normalized by the
    # verifier, but the leading space is kept on both sides).
    for pid0, pid1, label, lead in [
        ("113", "111", "113_111", " (coupling/0.303)**2 * 4.7e-4 * "),
        ("113", "221", "113_221", " (coupling/0.303)**2 * 3.0e-4 * "),
        ("223", "111", "223_111", " (coupling/0.303)**2 * 8.33e-2 * "),
        ("223", "221", "223_221", " (coupling/0.303)**2 * 4.5e-4 * "),
    ]:
        model.add_production_2bodydecay(
            pid0=pid0,
            pid1=pid1,
            label=label,
            br=lead + VP_KIN,
            generator=generators_light,
            energy=energy,
            nsample=nsample_2body,
        )
    # phi -> eta X carries an extra factor 4. and no explicit label (key "333").
    model.add_production_2bodydecay(
        pid0="333",
        pid1="221",
        br=" 4. * (coupling/0.303)**2 * 1.306e-2 * " + VP_KIN,
        generator=generators_light,
        energy=energy,
        nsample=nsample_2body,
    )

    # ---- Bremsstrahlung (precomputed QRA spectra, coupling g = 0.303 ref) ----
    model.add_production_direct(
        label="Brem",
        energy=energy,
        configuration=brem_configurations,
        coupling_ref=0.303,
        masses=BREM_MASSES,
    )

    # ---- Lifetime and visible branching fractions ----
    model.set_ctau_1d(filename="model/ctau.txt")

    # Decay final states as (mode, PDG IDs) pairs (DeLiVeR, arXiv:2201.01788).
    # Pairing avoids the parallel-list shift that misaligned the hadronic final
    # states. The protophobic current is not purely isoscalar, so (unlike the
    # U(1) models) the isovector channels survive; tau and the neutrino modes are
    # dropped as identically zero across the grid. Each mode loads
    # model/br/bfrac_<mode>.txt, including the charge-specific hadronic
    # sub-channels (KK_c, KK_n, 4pi_c, ...).
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


def build_presets(model, *, energy="14", **_):
    """
    Plot/scan defaults for the Protophobic signature notebook.

    Lifted from Protophobic.ipynb. The reach overlays Run-3 / HL-LHC FASER from
    saved results. See the Foresee.load_model docstring for the schema.

    Parameters
    ----------
    model: Model
        The configured Model returned by build_model.
    energy: str
        Collider energy as a string.
    """
    return {
        "benchmark": {"mass": 0.017, "coupling": 1e-5},
        "grid": {
            "masses": [round(x, 5) for x in np.logspace(-2, np.log10(2.0), 60)],
            "couplings": np.logspace(-8, -3, 121),
        },
        "detectors": default_detectors("decay"),
        "production_channels": [
            {"channels": ["221"],     "color": "tab:orange", "label": r"$\eta \to \gamma A'$",  "generators": ["EPOSLHC"]},
            {"channels": ["331"],     "color": "tab:green",  "label": r"$\eta' \to \gamma A'$", "generators": ["EPOSLHC"]},
            {"channels": ["113_111"], "color": "tab:red",    "label": r"$\rho^0 \to \pi^0 A'$", "generators": ["EPOSLHC"]},
            {"channels": ["113_221"], "color": "tab:purple", "label": r"$\rho^0 \to \eta A'$",  "generators": ["EPOSLHC"]},
            {"channels": ["223_111"], "color": "tab:brown",  "label": r"$\omega \to \pi^0 A'$", "generators": ["EPOSLHC"]},
            {"channels": ["223_221"], "color": "tab:pink",   "label": r"$\omega \to \eta A'$",  "generators": ["EPOSLHC"]},
            {"channels": ["333"],     "color": "tab:gray",   "label": r"$\phi \to \eta A'$",    "generators": ["EPOSLHC"]},
            {"channels": ["Brem"],    "color": "tab:olive",  "label": "Bremsstrahlung",         "generators": ["Brem_QRA_L1.5"]},
        ],
        "production_plot": {
            "condition": "logth<-3.7 and logp>2",
            "xlims": [0.01, 2.0],
            "ylims": [2e4, 4e13],
            "xlabel": r"Mass [GeV]",
            "ylabel": r"Production Rate $\sigma/g^2$ [pb]",
            "title": r"$\theta < 0.2$ mrad and $E > 100$ GeV",
            "legendloc": (1.01,1),
            "fs_label": 12,
            "fs_label_br": 9,
            "ncol": 3,
            "figsize": (7,6),
        },
        # Reach overlays (cell 21): saved result file, label, color, linestyle,
        # label-rotation, n_signal.
        "reach_setups": [
            ["13.6TeV_R3_FASER_EPOSLHC.npy", r"FASER (Run 3)",   "firebrick", "solid",  0.0, 3],
            ["14TeV_HL_FASER_EPOSLHC.npy",   r"FASER (HL-LHC)",  "red",       "dashed", 0.0, 3],
            ["14TeV_HL_FASER2_EPOSLHC.npy",  r"FASER2 (HL-LHC)", "salmon",    "dashed", 0.0, 3],
        ],
        "reach_plot": {
            "title": "Protophobic Gauge Boson",
            "xlims": [0.01,2.5],
            "ylims": [2e-8,1e-3],
            "xlabel": r"Gauge boson mass $m_{X}$ [GeV]",
            "ylabel": r"$g$",
            "legendloc": (1.025,0.6),
            "linewidths": 2,
        },
        "bounds": [
            ["bounds_FASER.txt",                        "FASER", 2.2e-2, 2.2e-5,     -25],
            ["bounds_E137_Bjorken1988as.txt",           "E137",  0.0101, 1.01*10**-7, 0],
            ["bounds_CHARM_Bergsma1985qz.txt",          "CHARM", 0.120,  1.2*10**-7,  0],
            ["bounds_Orsay_Davier1989wz.txt",           "Orsay", 0.042,  2.5*10**-6,  -28],
            ["bounds_E141_Riordan1987aw.txt",           "E141",  0.011,  1.8*10**-5,  0],
            ["bounds_NA64_Banerjee2019hmi.txt",         "NA64",  0.014,  1.0*10**-4,  -35],
            ["bounds_KLOE_Anastasi2015qla.txt",         "KLOE",  0.011,  7.0*10**-4,  0],
            ["bounds_BaBar_Lees2014xha.txt",            "BaBar", 0.060,  7.0*10**-4,  0],
            ["bounds_LHCb_Aaij2019bvg_prompt.txt",      "LHCb",  0.220,  2.0*10**-5,  0],
            ["bounds_LHCb_Aaij2019bvg_displaced_1.txt", None,    0,      0,           0],
            ["bounds_LHCb_Aaij2019bvg_displaced_2.txt", None,    0,      0,           0],
            ["bounds_LHCb_Aaij2019bvg_displaced_3.txt", None,    0,      0,           0],
            ["bounds_PADME.txt",                        "PADME", 1.6e-2, 2.2e-4,       0],
        ],
        "bounds2": [
            ["bounds_Anomaly.txt", "Anomaly", 0.07, 7.6*10**-5, 0],
        ],
        "projections": [],
        "branchings": [
            ["elec"    , "red"        , "solid" , r"$e^+e^-$"            , 0.050, 0.6],
            ["muon"    , "orange"     , "solid" , r"$\mu^+\mu^-$"        , 0.140, 0.10],
            ["2pi"     , "blue"       , "solid" , r"$\pi^+\pi^-$"        , 0.29,  0.1],
            ["PiGamma" , "dodgerblue" , "solid" , r"$\pi^0\gamma$" , 0.55,  0.02],
            ["3pi"     , "brown"       , "solid" , r"$\pi^0\pi^+\pi^-$"   , 0.45,   0.05],
            ["KK_c"    , "green"      , "solid" , r"$K^+K^-$"      , 1.05,  0.5],
        ],
    }
