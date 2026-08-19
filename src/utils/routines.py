import ast
import glob
import json
import os
import sys

import numpy as np

# resolve src.* against the library root (three directories up) so the imports
# below work regardless of the calling notebook's working directory
LIBRARY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if LIBRARY_ROOT not in sys.path:
    sys.path.insert(0, LIBRARY_ROOT)
from src.foresee import energy_stem  


def progress(iterable):
    """
    Wrap an iterable in a progress bar (notebook widget or console bar via
    tqdm.auto)

    Parameters
    ----------
    iterable
        The iterable to wrap

    Returns
    -------
        A tqdm-wrapped iterable, or the original if tqdm is unavailable
    """
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable)
    except ImportError:
        return iterable


def cache_spectra(foresee, masses, coupling=1, overwrite=False):
    """
    Generate any LLP spectrum files still missing for the model's beam energy

    Parameters
    ----------
    foresee: Foresee
        The configured Foresee instance (its set model holds the spectrum cache)
    masses: [float]
        Masses to ensure are cached
    coupling: float
        Coupling the spectra are generated at. Decay spectra are
        coupling-independent (use 1); scattering fluxes carry the production
        coupling
    overwrite: bool
        Regenerate spectra that are already cached
    """
    energy = str(next(iter(foresee.model.production.values()))["energy"])
    spectra_dir = os.path.join(foresee.model.modelpath, "model", "LLP_spectra")
    for mass in progress(masses):
        cached = os.path.join(spectra_dir, f"{energy_stem(energy)}_m_{mass}.txt.gz")
        if overwrite or not os.path.exists(cached):
            foresee.get_llp_spectrum(mass=mass, coupling=coupling)


def gen_events(foresee, masses, couplings, labels, modes, detectors,
               preselectioncuts="np.sqrt(p**2 + mass**2) > 100", outdir=None):
    """
    Scan the mass x coupling grid for each detector and save the event counts

    For each detector the counts come from get_events (decay signature) or
    get_events_interaction (scattering), one column per label, and are saved
    to <outdir>/<energy>_<detector>_<label>.npy as [masses, couplings,
    counts].

    Parameters
    ----------
    foresee: Foresee
        The configured Foresee instance with its LLP spectra already cached
    masses: [float]
        Masses to scan
    couplings: [float]
        Couplings to scan
    labels: [str]
        One label per production configuration (column of the event weights)
    modes: dict
        Production modes passed to get_events/get_events_interaction (channel
        key -> list of production labels), or None for the full production set
    detectors: [dict]
        Detector dicts, each a label plus set_detector kwargs
    preselectioncuts: str
        Cut applied when reading the spectrum files, defaulting to the research
        notebooks' energy cut; pass None to apply no cut
    outdir: str
        Directory the result files are written to, defaulting to
        results/<model name> under the current working directory
    """
    model = foresee.model
    energy = str(next(iter(model.production.values()))["energy"])
    scattering = getattr(model, "dsigma_der", None) is not None
    if outdir is None:
        outdir = os.path.join(os.getcwd(), "results", model.model_name)
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    for det in detectors:
        files = {label: os.path.join(
            outdir, f"{energy_stem(energy)}_{det['label']}_{label}.npy")
            for label in labels}
        foresee.set_detector(**{k: v for k, v in det.items() if k not in ("label", "energy")})
        counts = {label: [] for label in labels}
        for mass in progress(masses):
            if scattering:
                _, nevents, _, _ = foresee.get_events_interaction(
                    mass=mass, energy=energy, couplings=couplings, nsample=5,
                    modes=modes, preselectioncuts=preselectioncuts)
            else:
                _, _, nevents, _, _ = foresee.get_events(
                    mass=mass, energy=energy, couplings=couplings, nsample=10,
                    modes=modes, preselectioncuts=preselectioncuts)
            # column i of the event weights -> label i; nan/inf -> no sensitivity
            for i, label in enumerate(labels):
                counts[label].append(np.nan_to_num(nevents.T[i]))
        for label in labels:
            np.save(files[label], np.array([masses, couplings, counts[label]], dtype="object"))


def gen_hepmc(foresee, masses, couplings, labels, modes, detectors,
              nevent=100, nsample=1, filetype="hepmc", outdir=None):
    """
    Write one event file per (detector, mass, coupling) scan point

    For each point write_events writes
    <outdir>/<energy>_<detector>_m<mass>_g<coupling>.<filetype>. Decay
    signature only; points with no LLP production are skipped.

    Parameters
    ----------
    foresee: Foresee
        The configured Foresee instance with its LLP spectra already cached
    masses: [float]
        Masses to scan
    couplings: [float]
        Couplings to scan
    labels: [str]
        One label per production configuration, written as the per-event
        weight names
    modes: dict
        Production modes passed to write_events (channel key -> list of
        production labels), or None for the full production set
    detectors: [dict]
        Detector dicts, each a label plus set_detector kwargs
    nevent: int
        Unweighted events sampled per file
    nsample: int
        Sampling passed to write_events
    filetype: str
        Output file type ("hepmc" or "csv")
    outdir: str
        Directory the event files are written to, defaulting to
        results/<model name> under the current working directory
    """
    energy = str(next(iter(foresee.model.production.values()))["energy"])
    stem = energy_stem(energy)
    if outdir is None:
        outdir = os.path.join(os.getcwd(), "results", foresee.model.model_name)
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    for det in detectors:
        foresee.set_detector(**{k: v for k, v in det.items() if k not in ("label", "energy")})
        for mass in progress(masses):
            for coupling in couplings:
                filename = os.path.join(
                    outdir, f"{stem}_{det['label']}_m{mass}_g{coupling}.{filetype}")
                try:
                    foresee.write_events(
                        mass=mass, coupling=coupling, energy=energy,
                        filename=filename, numberevent=nevent, nsample=nsample,
                        filetype=filetype, modes=modes, weightnames=labels)
                except ValueError:
                    # no LLP production at this point -> nothing to write
                    continue


def get_presets(modelname):
    """
    Parse a model's scan and plotting presets from its research notebook.

    Parameters
    ----------
    modelname: str
        Model name resolved like Foresee.load_model: the directory under
        Models/ or a bare name matching a unique nested directory

    Returns
    -------
        Dict with the notebook's benchmark point (mass, coupling), scan grid
        (masses, couplings), scan setup (energy, modelname, setupnames,
        modes), plot data (productions, branchings, bounds, bounds2,
        projections, lines, setups), and the remaining cosmetic plot kwargs
        (production_plot, reach_plot)
    """
    import builtins
    import matplotlib.colors as mcolors
    from src.utils.utility import BREM_MASSES

    # locate Models/<modelname>/ and its single research notebook
    directory = os.path.join(LIBRARY_ROOT, "Models", modelname)
    if not os.path.isdir(directory):
        candidates = [
            os.path.dirname(p) for p in glob.glob(
                os.path.join(LIBRARY_ROOT, "Models", "**", "build.py"), recursive=True)
            if os.path.basename(os.path.dirname(p)) == modelname
        ]
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"no unique Models directory found for model '{modelname}'")
        directory = candidates[0]
    notebooks = [p for p in glob.glob(os.path.join(directory, "*.ipynb"))
                 if os.path.basename(p) != "model.ipynb"]
    if len(notebooks) != 1:
        raise RuntimeError(
            f"expected one research notebook in {directory}, found {len(notebooks)}")

    # code cells with IPython magics stripped, parsed once
    with open(notebooks[0], encoding="utf-8") as f:
        cells = json.load(f)["cells"]
    trees = []
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        src = "\n".join(line for line in "".join(cell["source"]).splitlines()
                        if not line.lstrip().startswith(("%", "!")))
        try:
            trees.append(ast.parse(src))
        except SyntaxError:
            continue

    # the namespace assignments and plot kwargs evaluate against
    safe = {name: getattr(builtins, name) for name in (
        "abs", "all", "any", "dict", "enumerate", "float", "int", "len",
        "list", "map", "max", "min", "pow", "range", "round", "sorted",
        "str", "sum", "tuple", "zip")}
    env = {"np": np, "numpy": np, "mcolors": mcolors, "BREM_MASSES": BREM_MASSES}

    def value_of(node):
        # env is merged into globals so names resolve inside comprehensions
        return eval(compile(ast.Expression(node), "<preset>", "eval"),
                    {"__builtins__": safe, **env})

    # fold every simple assignment, in order; the last one wins
    for tree in trees:
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            try:
                value = value_of(node.value)
            except Exception:
                continue
            if isinstance(target, ast.Name):
                env[target.id] = value
            elif (isinstance(target, ast.Tuple)
                    and all(isinstance(t, ast.Name) for t in target.elts)
                    and isinstance(value, (list, tuple))
                    and len(value) == len(target.elts)):
                env.update(zip([t.id for t in target.elts], value))

    # split each plot call's kwargs into the data arguments the notebook
    # passes around and the cosmetic kwargs forwarded via **production_plot /
    # **reach_plot; masses/energy are the notebook's own scan variables
    presets = {"modes": None, "productions": None, "branchings": None,
               "bounds": [], "bounds2": [], "projections": [], "lines": [],
               "setups": [], "production_plot": {}, "reach_plot": {}}
    plot_calls = {
        "plot_production": ("production_plot", ("productions", "branchings")),
        "plot_reach": ("reach_plot", ("setups", "bounds", "bounds2", "projections", "lines")),
    }
    seen = set()
    for tree in trees:
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "foresee"
                    and node.func.attr in plot_calls and node.func.attr not in seen):
                continue
            seen.add(node.func.attr)
            key, data_kwargs = plot_calls[node.func.attr]
            for kw in node.keywords:
                if kw.arg is None or kw.arg in ("masses", "energy"):
                    continue
                try:
                    value = value_of(kw.value)
                except Exception:
                    continue
                if kw.arg in data_kwargs:
                    presets[kw.arg] = value
                else:
                    presets[key][kw.arg] = value

    # drop production groups whose channel list folded empty (groupings built
    # from live model channels, e.g. the HNLs, are not recoverable by parsing)
    if presets["productions"]:
        presets["productions"] = [p for p in presets["productions"] if p.get("channels")]

    for name in ("energy", "modelname", "mass", "coupling", "masses",
                 "couplings", "setupnames", "modes"):
        if name in env:
            presets[name] = env[name]
    return presets
