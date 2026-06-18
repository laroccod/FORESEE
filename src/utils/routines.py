"""
Shared plotting routines for the FORESEE signature notebooks

decay.ipynb, scattering.ipynb, and the per-model model.ipynb notebooks are
model-agnostic: they pick a model name and params, then call the routines here
to produce the standard figures (spectrum, production rate, reach). Everything
model-specific comes from model.presets, built by each model's build_presets.

Typical use:

    from src.utils import routines
    foresee, presets = routines.load(MODEL_NAME, MODEL_PARAMS)                # build model, attach presets
    masses, couplings = np.logspace(-1,1,10), np.logspace(-4,-2,10)
    routines.plot_spectrum(foresee, mass=0.05, coupling=1)                    # LLP spectrum at one point
    routines.cache_spectra(foresee, masses)                                  # precompute LLP spectra
    routines.plot_production(foresee, presets, masses)                        # production rate vs mass
    routines.get_sens(foresee, presets, MODEL_PARAMS,                         # sensitivity reach across detectors
                      detectors=presets["detectors"],
                      masses=masses,
                      couplings=couplings)
    routines.gen_hepmc(foresee, presets, MODEL_PARAMS,                        # HepMC event files per scan point
                       masses=masses,
                       couplings=couplings)
"""
import os
import sys

import numpy as np

# resolve src.* against the library root (three directories up) so the imports
# below work regardless of the calling notebook's working directory
LIBRARY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if LIBRARY_ROOT not in sys.path:
    sys.path.insert(0, LIBRARY_ROOT)
from src.foresee import Foresee, energy_stem  # noqa: E402
from src.utils.detectors import DETECTOR_ENERGY, DETECTOR_LABELS  # noqa: E402

# reach-curve styling: one color per detector, one linestyle per generator
COLORS = ["firebrick", "red", "salmon", "darkorange", "gold",
          "tab:purple", "tab:blue", "green"]
STYLES = ["solid", "dashed", "dotted"]


def progress(iterable):
    """
    Wrap an iterable in a notebook progress bar, falling back to a no-op

    Parameters
    ----------
    iterable
        The iterable to wrap

    Returns
    -------
        A tqdm-wrapped iterable, or the original if tqdm is unavailable
    """
    try:
        from tqdm.notebook import tqdm
        return tqdm(iterable)
    except ImportError:
        return iterable


def is_scattering(model):
    """
    True for scattering-signature models (those that set a differential dsigma)

    Parameters
    ----------
    model: Model
        The model to check

    Returns
    -------
        True if the model defines dsigma_der, else False
    """
    return getattr(model, "dsigma_der", None) is not None


def save_figure(fig, name, kind):
    """
    Save a figure as both vector PDF and 300-dpi PNG into figures/<name>

    Parameters
    ----------
    fig
        Object exposing savefig (the pyplot module or a Figure)
    name: str
        Model name, selecting the figures/<name> directory
    kind: str
        Figure type ("Spectrum"/"Production"/"Reach"), used in the filename

    Returns
    -------
        Path to the saved PDF
    """
    directory = os.path.join("figures", name)
    os.makedirs(directory, exist_ok=True)
    pdf = os.path.join(directory, f"{kind}_{name}.pdf")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(os.path.join(directory, f"{kind}_{name}.png"), dpi=300, bbox_inches="tight")
    return pdf


def load(name, params, src_path=None, thin=1):
    """
    Load a model, attach its presets, thin the mass grid, return foresee + presets

    Parameters
    ----------
    name: str
        Model name passed to Foresee.load_model
    params: dict
        Build parameters forwarded to the model builder
    src_path: str
        Library root passed to Foresee (must end in a separator). Defaults to the
        root resolved from this module's location
    thin: int
        Stride applied to the mass grid; default 1 keeps the full grid, pass a
        larger stride to thin it for speed

    Returns
    -------
        (foresee, presets): the configured Foresee (its model already set via
        set_model, reachable as foresee.model) and its presets dict
    """
    if src_path is None:
        src_path = LIBRARY_ROOT + os.sep
    foresee = Foresee(path=src_path)
    model = foresee.load_model(name, **params)
    foresee.set_model(model=model)

    presets = getattr(model, "presets", None)
    if presets is None:
        raise RuntimeError(
            f"{name} has no build_presets(); the preset-driven cells "
            "cannot run. Add build_presets to its build.py first.")

    # full mass grid by default (thin=1); pass thin>1 to subsample for speed
    presets["grid"]["masses"] = presets["grid"]["masses"][::thin]

    grid = presets["grid"]
    print(f"Loaded model: {model.model_name}")
    print(f"Production channels: {len(model.production)}")
    print(f"Decay modes: {len(model.br_finalstate)}")
    print(f"Benchmark: m={presets['benchmark']['mass']} GeV, "
          f"g={presets['benchmark']['coupling']}")
    print(f"Mass grid: {len(grid['masses'])} points "
          f"[{min(grid['masses']):.3g} .. {max(grid['masses']):.3g}] GeV")
    print(f"Couplings: {len(grid['couplings'])} points "
          f"[{min(grid['couplings']):.3g} .. {max(grid['couplings']):.3g}]")
    print("Detectors:", [d["label"] for d in presets["detectors"]])

    return foresee, presets


def model_energy(model):
    """
    Beam energy the model's production channels were built at

    Read back off the model rather than passed in: every production channel is
    built at one energy (add_production_* stores it), and get_llp_spectrum names
    its cache file from that same energy. Deriving it here keeps the cache
    lookups in lockstep with what get_llp_spectrum writes.

    Parameters
    ----------
    model: Model
        The set model

    Returns
    -------
        str: the production-channel beam energy
    """
    energies = {str(p["energy"]) for p in model.production.values()}
    if not energies:
        raise RuntimeError(f"{model.model_name} has no production channels.")
    if len(energies) > 1:
        raise RuntimeError(
            f"{model.model_name} mixes production energies {sorted(energies)}; "
            "rebuild it at a single beam energy.")
    return energies.pop()


def cache_spectra(foresee, masses):
    """
    Generate any LLP spectrum files still missing for the model's beam energy

    get_llp_spectrum always recomputes, so masses already cached on disk are
    skipped to keep repeated runs fast. The cache lives under the set model's
    modelpath, the same directory get_llp_spectrum writes to. The beam energy is
    read off the set model (model_energy), matching the cache filenames
    get_llp_spectrum writes.

    Parameters
    ----------
    foresee: Foresee
        The configured Foresee instance (its set model holds the spectrum cache)
    masses: [float]
        Masses to ensure are cached
    """
    energy = model_energy(foresee.model)
    spectra_dir = os.path.join(foresee.model.modelpath, "model", "LLP_spectra")
    for mass in progress(masses):
        cached = os.path.join(spectra_dir, f"{energy_stem(energy)}_m_{mass}.txt.gz")
        if not os.path.exists(cached):
            foresee.get_llp_spectrum(mass=mass, coupling=1)


def plot_spectrum(foresee, mass, coupling=1):
    """
    Plot the LLP spectrum (or DM flux for scattering models) at a given point

    Parameters
    ----------
    foresee: Foresee
        The configured Foresee instance (its set model names the figure)
    mass: float
        Mass to plot the spectrum at
    coupling: float
        Coupling to plot at. Decay spectra are coupling-independent (use 1);
        scattering fluxes carry the production coupling

    Returns
    -------
        The saved figure
    """
    fig = foresee.get_llp_spectrum(mass=mass, coupling=coupling, do_plot=True)
    save_figure(fig, foresee.model.model_name, "Spectrum")
    fig.show()
    return fig


def plot_production(foresee, presets, masses=None):
    """
    Plot production rate vs mass, one curve per channel

    A single-energy illustration at the model's beam energy; the reach scan
    evaluates each detector at its own beam.

    Parameters
    ----------
    foresee: Foresee
        The configured Foresee instance (its set model is plotted)
    presets: dict
        Supplies production_channels and production_plot
    masses: [float]
        Masses to plot over; defaults to the presets grid

    Returns
    -------
        The saved figure, or None if the presets omit
        production_channels/production_plot
    """
    model = foresee.model
    if not (presets.get("production_channels") and presets.get("production_plot")):
        print(f"{model.model_name}: no production_channels/production_plot preset "
              "- skipping production-rate plot.")
        return None
    energy = model_energy(model)
    masses = presets["grid"]["masses"] if masses is None else masses
    cache_spectra(foresee, masses)
    prod = foresee.plot_production(
        masses=masses,
        productions=presets["production_channels"],
        energy=energy,
        branchings=presets.get("branchings") or None,
        **presets["production_plot"],
    )
    # plot_production returns (plt, ax, ax2) when it draws a BR sub-panel
    fig = prod[0] if isinstance(prod, tuple) else prod
    save_figure(fig, model.model_name, "Production")
    fig.show()
    return fig


def detector_energy(det):
    """
    Beam energy a detector runs at

    An inline "energy" key wins, so a one-off detector can be scanned without a
    registry entry; otherwise the label is looked up in DETECTOR_ENERGY.

    Parameters
    ----------
    det: dict
        Detector preset dict (carries a label and optionally an energy)

    Returns
    -------
        str: the detector's beam energy
    """
    if det.get("energy") is not None:
        return str(det["energy"])
    try:
        return DETECTOR_ENERGY[det["label"]]
    except KeyError:
        raise KeyError(
            f"detector {det['label']!r} is not in DETECTOR_ENERGY; register it "
            "there or pass an inline 'energy' on the detector dict.")


def group_detectors_by_energy(detectors):
    """
    Group detectors by the beam energy they run at (detector_energy)

    Parameters
    ----------
    detectors: [dict]
        Detector preset dicts; each must resolve via detector_energy (a registry
        entry or an inline "energy" key)

    Returns
    -------
        Dict mapping energy -> list of detectors
    """
    groups = {}
    for det in detectors:
        groups.setdefault(detector_energy(det), []).append(det)
    return groups


def result_filename(energy, detector, generator):
    """
    Filename of one detector's reach curve at a given beam energy/generator

    Parameters
    ----------
    energy: str
        Beam energy
    detector: dict
        Detector preset (supplies the label)
    generator: str
        Generator name

    Returns
    -------
        The .npy filename
    """
    return f"{energy_stem(energy)}_{detector['label']}_{generator}.npy"


def default_labels(model, modes):
    """
    Numeric column labels, one per production configuration

    Each production channel emits a column of event weights; the column count is
    the widest production list across the selected modes (or across the model's
    full production set when modes is None).

    Parameters
    ----------
    model: Model
        The set model, supplying the full production set when modes is None
    modes: dict
        Production modes (channel key -> production labels), or None

    Returns
    -------
        [int]: column indices 0..n-1
    """
    if modes is not None:
        nprods = max((len(v) for v in modes.values()), default=1)
    else:
        nprods = max((len(p["production"]) for p in model.production.values()), default=1)
    return list(range(nprods))


def get_sens(foresee, presets, params, detectors=None, masses=None,
             couplings=None, modes=None, labels=None, plot=True):
    """
    Scan the sensitivity reach across all detectors, each at its own beam energy

    Detectors are grouped by DETECTOR_ENERGY. For each group the model is rebuilt
    at that energy and its spectra cached; any detector missing a result .npy is
    scanned live (small nsample) over the given mass/coupling grid, writing one
    curve per (detector, configuration), while existing curves are read as-is.
    When plot is True the reach figure is drawn for the first (nominal)
    configuration of each detector, with bounds/projections overlaid.

    Parameters
    ----------
    foresee: Foresee
        The configured Foresee instance (its set model is scanned)
    presets: dict
        Supplies bounds, projections, reach_plot, and the defaults below
    params: dict
        Build parameters, forwarded to load_model when the model is rebuilt at
        each detector's beam energy
    detectors: [dict]
        Detectors to scan; defaults to the presets detectors
    masses: [float]
        Masses to scan; defaults to the presets grid
    couplings: [float]
        Couplings to scan; defaults to the presets grid
    modes: dict
        Production modes passed to get_events/get_events_interaction (channel key
        -> list of production labels). None uses the model's full production set
    labels: [str]
        Label for each configuration (column) defined by modes; the i-th label
        names column i of the event weights. Defaults to numeric column indices
        built from the set model's production
    plot: bool
        Draw and save the reach figure (default True); pass False to only
        scan and cache the result files

    Returns
    -------
        The saved figure, or None if plot is False or no reach results exist
    """
    # resolve the scan grid, falling back to the presets defaults
    model = foresee.model   # captured before the per-energy rebuilds below reassign foresee.model
    detectors = presets["detectors"] if detectors is None else detectors
    masses = presets["grid"]["masses"] if masses is None else masses
    couplings = presets["grid"]["couplings"] if couplings is None else couplings

    results_dir = os.path.join(model.modelpath, "model", "results")
    os.makedirs(results_dir, exist_ok=True)

    # one reach curve per production configuration (a column of the event
    # weights). modes selects which channels feed get_events; labels names each
    # column, and the plot shows only the first (nominal) configuration.
    labels = default_labels(model, modes) if labels is None else labels
    plot_labels = labels[:1]
    scattering = is_scattering(model)
    groups = group_detectors_by_energy(detectors)

    # scan each beam-energy group for any missing curves, then build the plot
    # setups from whatever result files exist on disk
    setups = []
    detector_i = 0
    for scan_energy, dets in groups.items():
        # scan the group only if some detector is missing a result file
        missing = [
            d for d in dets
            if not all(os.path.exists(os.path.join(results_dir, result_filename(scan_energy, d, label)))
                       for label in labels)
        ]
        if missing:
            try:
                # rebuild at this beam energy and cache its spectra before scanning
                foresee.set_model(model=foresee.load_model(
                    model.model_name, **{**params, "energy": scan_energy}))
                cache_spectra(foresee, masses)
                for det in missing:
                    foresee.set_detector(**{k: v for k, v in det.items() if k not in ("label", "energy")})
                    curves = {label: [] for label in labels}
                    for mass in progress(masses):
                        if scattering:
                            _, nevents, _, _ = foresee.get_events_interaction(
                                mass=mass, energy=scan_energy, couplings=couplings, nsample=5, modes=modes)
                        else:
                            _, _, nevents, _, _ = foresee.get_events(
                                mass=mass, energy=scan_energy, couplings=couplings, nsample=10, modes=modes)
                        for i, label in enumerate(labels):
                            # column i of the event weights -> this label's curve.
                            # nan/inf (e.g. exp overflow at extreme beam-dump lengths) -> no sensitivity
                            curves[label].append(np.nan_to_num(nevents.T[i]))
                    for label in labels:
                        result = np.array([masses, couplings, curves[label]], dtype="object")
                        np.save(os.path.join(results_dir, result_filename(scan_energy, det, label)), result)
            except Exception as exc:
                print(f"  scan failed at energy {scan_energy}: {exc}")
                print("  (this model likely needs direct/heavy-meson spectra not "
                      "shipped, or has no production at this beam)")
        # one plot setup per detector (nominal configuration only), skipping
        # any whose curve never made it to disk
        for det in dets:
            for label_i, label in enumerate(plot_labels):
                fn = result_filename(scan_energy, det, label)
                if not os.path.exists(os.path.join(results_dir, fn)):
                    continue
                display = DETECTOR_LABELS.get(det["label"], det["label"])
                legend = f"{display} ({label})" if len(plot_labels) > 1 else display
                setups.append([fn, legend, COLORS[detector_i % len(COLORS)],
                               STYLES[label_i % len(STYLES)], 0.0, 3])
            detector_i += 1

    if not plot:
        return None

    if not setups:
        print("No reach results available - nothing to plot.")
        return None

    # the BR sub-panel is drawn under the production-rate plot, not here
    reach = foresee.plot_reach(
        setups=setups,
        bounds=presets.get("bounds", []),
        bounds2=presets.get("bounds2", []),
        projections=presets.get("projections", []),
        lines=presets.get("lines", []),
        **presets["reach_plot"],
    )
    fig = reach[0] if isinstance(reach, tuple) else reach
    save_figure(fig, model.model_name, "Reach")
    fig.show()
    return fig


def gen_hepmc(foresee, presets, params, numberevent=100, masses=None,
              couplings=None, detectors=None, modes=None, labels=None,
              nsample=1, filetype="hepmc"):
    """
    Write one HepMC event file per (detector, mass, coupling) scan point

    For each point this calls foresee.write_events and writes
    <model>/model/events/<stem>_<detector>_m<mass>_g<coupling>.<filetype>.
    Detectors are grouped by beam energy; the model is rebuilt at each energy and
    its spectra cached first. Decay-signature models only. Points with no
    production at a detector's beam are skipped, not fatal.

    Parameters
    ----------
    foresee: Foresee
        The configured Foresee instance (its set model is used)
    presets: dict
        Supplies the default mass/coupling grid and detectors
    params: dict
        Build parameters forwarded to the builder
    numberevent: int
        Unweighted events sampled per file
    masses: [float]
        Masses to scan; defaults to the presets grid
    couplings: [float]
        Couplings to scan; defaults to the presets grid
    detectors: [dict]
        Detectors to scan; defaults to the presets detectors
    modes: dict
        Production modes passed to write_events (channel key -> list of production
        labels). None uses the model's full production set
    labels: [str]
        Label for each configuration (column), written as the per-event weight
        names. Defaults to numeric column indices built from modes (or the set
        model's production)
    nsample: int
        Sampling passed to write_events
    filetype: str
        Output file type (e.g. "hepmc")

    Returns
    -------
        List of written file paths
    """
    model = foresee.model
    if is_scattering(model):
        raise RuntimeError(
            "gen_hepmc writes LLP-decay events and applies only to decay-signature "
            "models; this is a scattering model.")

    # resolve the scan grid, falling back to the presets defaults
    name = model.model_name   # captured before the loop reassigns foresee.model
    masses = list(presets["grid"]["masses"]) if masses is None else list(masses)
    couplings = list(presets["grid"]["couplings"]) if couplings is None else list(couplings)
    detectors = presets["detectors"] if detectors is None else detectors
    labels = default_labels(model, modes) if labels is None else labels
    weightnames = [str(label) for label in labels]   # hepmc weight names must be strings

    events_dir = os.path.join(model.modelpath, "model", "events")
    os.makedirs(events_dir, exist_ok=True)
    print(f"Writing {filetype} files for {len(detectors)} detectors x {len(masses)} "
          f"masses x {len(couplings)} couplings into {events_dir}")

    # rebuild the model once per beam energy, then write one file per
    # (detector, mass, coupling) point in that group
    written, skipped = [], 0
    for scan_energy, dets in group_detectors_by_energy(detectors).items():
        model = foresee.load_model(name, **{**params, "energy": scan_energy})
        foresee.set_model(model=model)
        cache_spectra(foresee, masses)
        stem = energy_stem(scan_energy)
        for det in dets:
            foresee.set_detector(**{k: v for k, v in det.items() if k not in ("label", "energy")})
            for mass in progress(masses):
                for coupling in couplings:
                    fname = f"{stem}_{det['label']}_m{mass}_g{coupling}.{filetype}"
                    try:
                        foresee.write_events(
                            mass=mass, coupling=coupling, energy=scan_energy,
                            filename=f"model/events/{fname}", numberevent=numberevent,
                            nsample=nsample, filetype=filetype,
                            modes=modes, weightnames=weightnames)
                        written.append(os.path.join(events_dir, fname))
                    except Exception:
                        # no LLP production for this detector's beam -> nothing to write
                        skipped += 1
    print(f"Wrote {len(written)} files"
          + (f", skipped {skipped} empty points." if skipped else "."))
    return written
