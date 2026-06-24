"""Builder for the U(1)_B (gauged baryon number) gauge boson model.

Returns a fully-configured Model for a leptophobic vector X coupled to baryon
number: production via pseudoscalar radiative decays (pi0, eta, eta' -> gamma X)
with vector-meson-dominance mixing, vector -> pseudoscalar transitions
(rho/omega/phi), and bremsstrahlung, plus the visible hadronic/leptonic decay
modes. A custom BW propagator is patched onto src.foresee for the eta/eta'
branching strings; the bremsstrahlung spectra are reused from U(1)_{B-L} (no
files/direct/U(1)B/ exists). Equivalent to running the setup cells of
U(1)B.ipynb up to the foresee.set_model(...) call.

Invoked by Foresee.load_model("U(1)B", **params).
"""

import numpy as np

from src.foresee import Model
from src.utils.utility import ensure_model_layout, BREM_MASSES
from src.utils.detectors import default_detectors
import src.foresee as foresee_module


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
    energy="13.6",
    nsample_2body=100,
    generators_light=None,
    generators_heavy=None,
    brem_configurations=None,
):
    """
    Build the U(1)_B Model.

    Parameters
    ----------
    path: str
        Model directory (Models/U(1)B/). Supplied by Foresee.load_model.
    energy: str
        Collider energy in TeV. Defaults to "13.6".
    nsample_2body: int
        Sampling points per production channel. Defaults to 100.
    generators_light: [str]
        Hadron generators for meson production. Defaults to
        ["EPOSLHC", "SIBYLL", "QGSJET"], the canonical triple for the
        shower-model uncertainty envelope.
    generators_heavy: [str]
        Accepted for API uniformity; U(1)_B has no heavy-meson channels.
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

    # Reuse the U(1)_{B-L} bremsstrahlung spectra (no U(1)B direct dir exists);
    # the notebook does the same via an explicit symlink.
    ensure_model_layout(path, link_direct=True, direct_name="U(1)B-L")

    # The eta/eta' BR strings call BW(self, mass, pid); make it resolvable in
    # the foresee module namespace where those strings are evaluated.
    foresee_module.BW = BW

    model = Model("U(1)B", path=path)

    # ---- Pseudoscalar radiative decays  P -> X gamma ----
    model.add_production_2bodydecay(
        pid0="111",
        pid1="22",
        br="2.*0.99  * (coupling/0.303)**2 * (1-mass**2/self.masses(pid0)**2)**3 ",
        generator=generators_light,
        energy=energy,
        nsample=nsample_2body,
    )
    model.add_production_2bodydecay(
        pid0="221",
        pid1="22",
        br="4.*2.*0.39 * (coupling/0.303)**2 * (1-mass**2/self.masses(pid0)**2)**3 * np.abs( (BW(self,mass,'223') + BW(self,mass,'333')) / (BW(self,mass,'223')  + 9*BW(self,mass,'113')  - 2*BW(self,mass,'333')) )**2",
        generator=generators_light,
        energy=energy,
        nsample=nsample_2body,
    )
    model.add_production_2bodydecay(
        pid0="331",
        pid1="22",
        br="4.*2.*0.023 * (coupling/0.303)**2 * (1-mass**2/self.masses(pid0)**2)**3 * np.abs( (BW(self,mass,'223') - 2*BW(self,mass,'333')) / (BW(self,mass,'223')  + 9*BW(self,mass,'113')  + 4*BW(self,mass,'333')) )**2",
        generator=generators_light,
        energy=energy,
        nsample=nsample_2body,
    )

    # ---- Vector-meson decays  V -> X P ----
    # rho/omega carry a leading factor 4; phi does not. The Kallen kinematic
    # factor is identical across all three.
    vector_meson_br = (
        "{lead}(coupling/0.303)**2 * {prefactor} * ("
        "(self.masses('pid0')**2 - (self.masses('pid1') + mass)**2)*"
        "(self.masses('pid0')**2 - (self.masses('pid1') - mass)**2)"
        "/(self.masses('pid0')**2 - self.masses('pid1')**2)**2"
        ")**(3/2)"
    )
    for pid0, pid1, lead, prefactor in [
        ("113", "111", "4. * ", "4.7e-4"),    # rho0  -> pi0 X
        ("223", "221", "4. * ", "4.5e-4"),    # omega -> eta X
        ("333", "221", "",      "1.306e-2"),  # phi   -> eta X
    ]:
        model.add_production_2bodydecay(
            pid0=pid0,
            pid1=pid1,
            br=vector_meson_br.format(lead=lead, prefactor=prefactor),
            generator=generators_light,
            energy=energy,
            nsample=nsample_2body,
        )

    # ---- Bremsstrahlung (precomputed, shared with U(1)_{B-L}) ----
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
    # states. U(1)_B's quark current is isoscalar, so the isovector channels
    # (2pi, 4pi, 6pi, EtaPiPi, EtaPrimePiPi, OmegaPion, PhiPi) vanish across the
    # grid and are omitted; being leptophobic, the neutrino modes (and tau) are
    # zero too and dropped. Each mode loads model/br/bfrac_<mode>.txt, including
    # the charge-specific hadronic sub-channels (KK_c, KK_n, KKpi_0, ...).
    decay_channels = [
        ("elec", [11, -11]), ("muon", [13, -13]),
        ("3pi",          [211, -211, 111]),
        ("PiGamma",      [111, 22]),
        ("EtaGamma",     [221, 22]),
        ("EtaOmega",     [221, 223]),
        ("EtaPhi",       [221, 333]),
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


def build_presets(model, *, energy="13.6", generators_light=None, **_):
    """
    Plot/scan defaults for the U(1)_B signature notebook.

    Lifted from U(1)B.ipynb. The reach overlays Run-3 / HL-LHC FASER from saved
    results. See the Foresee.load_model docstring for the schema.

    Parameters
    ----------
    model: Model
        The configured Model returned by build_model.
    energy: str
        Collider energy as a string.
    """
    masses = (
        [round(x, 5) for x in np.logspace(-2, np.log10(2.0), 50)]
        + [round(x, 5) for x in np.logspace(np.log10(0.5), np.log10(1.5), 10)]
    )
    masses.sort()

    return {
        "benchmark": {"mass": 0.3, "coupling": 2e-4},
        "grid": {
            "masses": masses,
            "couplings": np.logspace(-8, -1, 121),
        },
        "detectors": default_detectors("decay"),
        "production_channels": [
            {"channels": ["111"],  "color": "tab:blue",   "label": r"$\pi^0 \to \gamma A'$",  "generators": ["EPOSLHC"]},
            {"channels": ["221"],  "color": "tab:orange", "label": r"$\eta \to \gamma A'$",   "generators": ["EPOSLHC"]},
            {"channels": ["331"],  "color": "tab:green",  "label": r"$\eta' \to \gamma A'$",  "generators": ["EPOSLHC"]},
            {"channels": ["113"],  "color": "tab:red",    "label": r"$\rho^0 \to \pi^0 A'$",  "generators": ["EPOSLHC"]},
            {"channels": ["223"],  "color": "tab:brown",  "label": r"$\omega \to \eta A'$",   "generators": ["EPOSLHC"]},
            {"channels": ["333"],  "color": "tab:gray",   "label": r"$\phi \to \eta A'$",     "generators": ["EPOSLHC"]},
            {"channels": ["Brem"], "color": "tab:olive", "label": "Bremsstrahlung",   "generators": ["Brem_QRA_L1.5"]},
        ],
        "production_plot": {
            "condition": "logth<-3.7 and logp>2",
            "xlims": [0.01, 2.0],
            "ylims": [2e4, 4e13],
            "xlabel": r"Mass [GeV]",
            "ylabel": r"Production Rate $\sigma/g_{B}^2$ [pb]",
            "title": r"$\theta < 0.2$ mrad and $E > 100$ GeV",
            "legendloc": (1.01,1),
            "fs_label": 12,
            "fs_label_br": 9,
            "ncol": 3,
            "figsize": (7,6),
        },
        # Reach overlay curves (cell 18): pre-saved result file, label, color,
        # linestyle, label-rotation, n_signal. Run-3 result is produced live;
        # the HL-LHC curves come from saved files at 14 TeV.
        "reach_setups": [
            ["13.6TeV_R3_FASER_EPOSLHC_pT=1.npy", r"FASER (Run 3)",  "firebrick", "solid",  0.0, 3],
            ["14TeV_HL_FASER_EPOSLHC_pT=1.npy",   r"FASER (HL-LHC)", "red",       "dashed", 0.0, 3],
            ["14TeV_HL_FASER2_EPOSLHC_pT=1.npy",  r"FASER2 (HL-LHC)", "salmon",   "dashed", 0.0, 3],
        ],
        "reach_plot": {
            "title": "U(1)B Gauge Boson",
            "xlims": [0.1,2.0],
            "ylims": [8e-9,1e-1],
            "xlabel": r"Gauge boson mass $m_{X}$ [GeV]",
            "ylabel": r"$g_{B}$",
            "legendloc": (1.00,0.650),
            "linewidths": 2,
        },
        "bounds": [
            ["bounds_LSND.txt",                        "LSND",            0.200, 2.5*10**-7, 0],
            ["bounds_LHCb_Aaij2019bvg_displaced.txt",  r"$LHCb-\mu\mu$",  0.350, 1*10**-3,   0],
            ["bounds_LHCb_Aaij2019bvg_prompt.txt",     r"$LHCb-\mu\mu$",  0.350, 1*10**-3,   0],
            ["bounds_KLOEII.txt",                      "KLOE II",         0.240, 3*10**-2,   0],
            ["bounds_FASER.txt",                       "FASER",           0.35,  .42*10**-4, -30],
            ["bounds_FASERB.txt",                      "",                0.105, 1*10**-2,   90],
            ["bounds_NuCal.txt",                       "NuCal",           0.35,  .135*10**-4, -30],
            ["bounds_CHARM.txt",                       "CHARM",           0.300, 0.8*10**-6, 0],
            ["bounds_E137.txt",                        "E137",            0.14,  0.6*10**-4, 0],
            ["bounds_PS191.txt",                       "PS191",           0.102, 1*10**-4,   0],
            ["bounds_NA48.txt",                        "NA48",            0.105, 1*10**-2,   90],
        ],
        "bounds2": [
            ["anomaly_B_KX.txt",   r"Anomaly", 0.700, 0.9*10**-2, 0],
            ["anomaly_K_piX.txt",  r"",        0.700, 0.9*10**-2, 0],
            ["anomaly_Z_gX.txt",   r"",        0.700, 0.9*10**-2, 0],
        ],
        "projections": [],
        "branchings": [
            ["elec"     , "blue"        , "solid" , r"$e^+e^-$"             , 0.03,  0.6 ],
            ["PiGamma"  , "black"       , "solid" , r"$\pi^0\gamma$"     , 0.12,  0.30 ],
            ["3pi"      , "red"         , "solid" , r"$\pi^0\pi^+\pi^-$"       , 0.35,  0.30 ],
            ["KK_c"     , "magenta"     , "solid" , r"$K^+K^-$"         , .67,  0.45 ],
            ["KK_n"     , "darkgreen"   , "solid" , r"$K_S K_L$"        , .75,  0.20 ],
            ["OmPiPi_c" , "purple"      , "solid" , r"$\omega\pi^+\pi^-$" , 1.05,  0.2 ],
            ["EtaOmega" , "green"       , "solid" , r"$\eta\omega$"       , 1.50,  0.015 ],
            ["OmPiPi_n" , "orange"      , "solid" , r"$\omega2\pi^0$" , 1.09,  0.12 ],
        ],
    }
