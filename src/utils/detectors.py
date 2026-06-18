"""
Named detector geometries for FORESEE forward experiments

A registry so models reference a forward detector by label instead of repeating
the same geometry dict in every build_presets. Each entry holds only the
geometry consumed by Foresee.set_detector (distance, selection, length,
luminosity, and for scattering numberdensity/ermin/ermax); model-specific fields
are passed as overrides to detector().

Geometries (LHC entries: arXiv:2203.05090 and the FASER/FLArE TDRs):

============  =========  ======  ============  ============  ========================  ===========  ====================================
preset        distance   length  lumi          energy        geometry                  ref          notes
============  =========  ======  ============  ============  ========================  ===========  ====================================
FASER_R3(HL)  480 m      1.5 m   250(3000)/fb  13.6(14) TeV  circular r<0.1                         Run-3 (HL-LHC) FASER
FASER_2022    474 m      4.0 m   60/fb         13.6 TeV      circular r<0.1, y-offset               FASER 2022-2023 layout
FASER2_HL     650 m      10 m    3000/fb       14 TeV        1.5x1.0 m box                          HL-LHC FASER2
FLArE         620 m      7 m     3000/fb       14 TeV        1x1 m box                              HL-LHC FLArE LAr (scattering)
SHiP          60 m       50 m    6.67e6/fb     400 GeV       5x10 m elliptical         2112.01487   SPS beam dump
NuCal         64 m       23 m    5.70e4/fb     70 GeV        circular r<1.3            1311.3870    U-70 beam dump
DUNE_ND       574 m      5 m     3.67e7/fb     120 GeV       7x3 m box                 1912.08468   LBNF beam dump (ArgonCube)
CHARM         480 m      35 m    8.0e4/fb      400 GeV       3x3 m box, 5 m off-axis   1112.5438    SPS beam dump
============  =========  ======  ============  ============  ========================  ===========  ====================================

Beam dumps (SHiP, NuCal, DUNE_ND, CHARM) are fixed-target experiments and differ
from the LHC entries in two ways. Their luminosity is a POT-equivalent
L_equiv[fb^-1] = N_POT / (sigma_inel[pb] * 1000), with sigma_inel ~ 30 mb and
N_POT = 2.0e20 (SHiP), 1.71e18 (NuCal), 1.1e21/yr (DUNE_ND), 2.4e18 (CHARM, same
beam as SHiP). They must be paired with the matching fixed-target production
spectra; DETECTOR_ENERGY records the FT energy
each one runs at. DUNE_ND reuses FLArE's liquid-argon scattering params
(numberdensity/ermin/ermax) pending a DUNE-specific recoil window.
"""

# Geometry only. label is added by detector(); channels/efficiency/etc. are
# overrides supplied per model.
DETECTORS = {
    "FASER_R3": dict(
        distance=480, length=1.5, luminosity=250,
        selection="np.sqrt(x.x**2 + x.y**2) < .1",
    ),
    "FASER_2022": dict(
        distance=474, length=4.0, luminosity=60,
        selection="np.sqrt(x.x**2 + (x.y+0.065)**2) < .1",
    ),
    "FASER_HL": dict(
        distance=480, length=1.5, luminosity=3000,
        selection="np.sqrt(x.x**2 + x.y**2) < .1",
    ),
    "FASER2_HL": dict(
        distance=650, length=10, luminosity=3000,
        selection="-1.5 < x.x < 1.5 and -.5 < x.y < .5",
    ),
    "FLArE": dict(
        distance=620, length=7, luminosity=3000,
        selection="abs(x.x) < 0.5 and abs(x.y) < 0.5",
        numberdensity=3.754e29, ermin=0.03, ermax=1,
    ),
    "SHiP": dict(
        distance=60, length=50, luminosity=6.67e6,
        selection="(x.x/2.5)**2 + (x.y/5.0)**2 < 1",
    ),
    "NuCal": dict(
        distance=64, length=23, luminosity=5.70e4,
        selection="np.sqrt(x.x**2 + x.y**2) < 1.3",
    ),
    "DUNE_ND": dict(
        distance=574, length=5, luminosity=3.67e7,
        selection="abs(x.x) < 3.5 and abs(x.y) < 1.5",
        numberdensity=3.754e29, ermin=0.03, ermax=1,
    ),
    "CHARM": dict(
        distance=480, length=35, luminosity=8.0e4,
        selection="3.5 < x.x < 6.5 and -1.5 < x.y < 1.5",
    ),
}

# Beam energy each detector runs at. LHC entries use bare-number TeV labels; beam
# dumps use fixed-target labels resolving to files/hadrons/FT-*GeV.txt.gz.
DETECTOR_ENERGY = {
    "FASER_R3": "13.6",
    "FASER_2022": "13.6",
    "FASER_HL": "14",
    "FASER2_HL": "14",
    "FLArE": "14",
    "SHiP": "FT-400GeV",
    "NuCal": "FT-70GeV",
    "DUNE_ND": "FT-120GeV",
    "CHARM": "FT-400GeV",
}

DETECTOR_LABELS = {
    "FASER_R3":   "FASER (Run 3)",
    "FASER_2022": "FASER (2022)",
    "FASER_HL":   "FASER (HL-LHC)",
    "FASER2_HL":  "FASER2 (HL-LHC)",
    "FLArE":      "FLArE",
    "SHiP":       "SHiP",
    "NuCal":      "NuCal",
    "DUNE_ND":    "DUNE ND",
    "CHARM":      "CHARM",
}

# Standard sets iterated over in the routine notebooks. 
# Build a model's detector list from these via default_detectors().
DEFAULT_DECAY = ["FASER_R3", "FASER_HL", "FASER2_HL"]
DEFAULT_SCATTER = ["FLArE"]
SIGNATURE_SETS = {"decay": DEFAULT_DECAY, "scatter": DEFAULT_SCATTER}


def detector(label, **overrides):
    """
    Return a fresh set_detector-ready dict for a named preset

    Parameters
    ----------
    label: str
        A key in DETECTORS (e.g. "FASER_R3", "FASER2_HL")
    **overrides
        Fields merged on top of the preset geometry, for model-specific bits
        such as channels=[...] or a non-standard luminosity. Pass label="..."
        to give the detector a different display name than the preset key

    Returns
    -------
        {"label": <label>, **geometry, **overrides}
    """
    if label not in DETECTORS:
        raise KeyError(
            f"unknown detector preset {label!r}; known presets: {sorted(DETECTORS)}"
        )
    return {"label": label, **DETECTORS[label], **overrides}


def default_detectors(signature, **overrides):
    """
    Return the standard detector set for a model signature

    Parameters
    ----------
    signature: str
        "decay" (the three FASER detectors) or "scatter" (FLArE)
    **overrides
        Fields merged onto every detector in the set (see detector())

    Returns
    -------
        set_detector-ready dicts, one per detector in the set
    """
    if signature not in SIGNATURE_SETS:
        raise ValueError(f"signature must be 'decay' or 'scatter', got {signature!r}")
    return [detector(label, **overrides) for label in SIGNATURE_SETS[signature]]
