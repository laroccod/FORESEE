"""Tests that each model's build.py reproduces its research notebook.

For every model with a Models/<Name>/build.py, two Models are constructed: a
reference one by executing the notebook's setup cells, and a ported one via
Foresee.load_model. The test then compares their production channels,
branching ratios, lifetime, and scattering config. Interpolation functions
are compared by sampling them on a small grid; every value is reduced to a
plain dict so pytest's assertion diff reports any mismatch.

To run (from the library root, with pytest + numpy installed):
    python -m pytest tests/test_library.py -v
A single model:
    python -m pytest tests/test_library.py -k DarkPhoton
"""
import glob
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# tests/ sits directly under the library root; that root carries src/.
LIBRARY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LIBRARY))

from src.foresee import Foresee

# model name -> its directory, discovered from the builders. Models without a
# build.py (ALP-LSW, 2HDM_A) have nothing to compare and are not tested.
MODEL_DIRS = {
    Path(p).parent.name: Path(p).parent
    for p in glob.glob(str(LIBRARY / "Models" / "**" / "build.py"), recursive=True)
}


# ---------- building the two Models ----------

# A cell mentioning a setup token always runs during the reference build; a
# cell that only mentions plot/inspection tokens is skipped -- running it would
# be slow or would error on files the tests never generate.
SETUP_TOKENS = (
    "add_production_", "set_ctau_", "set_br_", "set_dsigma_",
    "set_model", "Model(", "Foresee(",
)
PLOT_TOKENS = (
    "plt.", "plot.", ".show(", ".savefig(",
    "get_llp_spectrum", "get_events", "write_events",
    "set_detector", "plot_production", "plot_reach",
)


def strip_magics(src):
    """
    Drop IPython %magic lines so a cell runs under exec().

    Parameters
    src: str
        Cell source.

    Returns
    The source without magic lines.
    """
    return "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("%"))


def plot_only(src):
    """
    True when a cell only plots or inspects, so the reference build skips it.

    Parameters
    src: str
        Cell source.

    Returns
    True for an empty cell, or one referencing a plot token and no setup token.
    """
    code = [ln for ln in src.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not code:
        return True
    if any(token in src for token in SETUP_TOKENS):
        return False
    return any(token in src for token in PLOT_TOKENS)


def setup_code(notebook_path):
    """
    The notebook's Model-building code, cell by cell.

    Parameters
    notebook_path: Path
        Research notebook to read.

    Returns
    Sources from the first cell through the set_model cell, with magics
    stripped and plot-only cells dropped.
    """
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in notebook["cells"] if c.get("cell_type") == "code"]
    for last, src in enumerate(cells):
        if any("set_model" in ln for ln in src.splitlines() if not ln.strip().startswith("#")):
            break
    else:
        raise RuntimeError(f"no set_model cell in {notebook_path}")
    sources = [strip_magics(src) for src in cells[: last + 1]]
    return [src for src in sources if not plot_only(src)]


def reference_model(name):
    """
    Build the reference Model by executing the notebook's setup cells.

    Runs with the working directory set to the model's folder, so the
    model/... paths in set_ctau_1d etc. resolve, and with that folder on
    sys.path for cell-local helper modules.

    Parameters
    name: str
        Model name.

    Returns
    The Model bound to the notebook's model variable.
    """
    model_dir = MODEL_DIRS[name]
    notebooks = [p for p in model_dir.glob("*.ipynb")
                 if p.name not in ("model.ipynb", "routines.ipynb")]
    assert len(notebooks) == 1, f"expected one research notebook in {model_dir}"
    namespace = {"__name__": "__notebook__"}
    saved_cwd, saved_path = os.getcwd(), list(sys.path)
    sys.path.insert(1, str(model_dir))
    os.chdir(model_dir)
    try:
        for src in setup_code(notebooks[0]):
            exec(compile(src, str(notebooks[0]), "exec"), namespace)
    finally:
        os.chdir(saved_cwd)
        sys.path[:] = saved_path
    return namespace["model"]


def ported_model(name):
    """
    Build the Model via Foresee.load_model, i.e. the model's build.py.

    Parameters
    name: str
        Model name.

    Returns
    The configured Model.
    """
    return Foresee(path=str(LIBRARY) + os.sep).load_model(name)


# ---------- reducing a Model to comparable dicts ----------

BR_MASSES = np.array([0.05, 0.2, 0.5, 1.0])
CTAU_MASSES = np.array([0.01, 0.05, 0.2, 0.5, 1.0])


def rounded(values):
    """
    Format floats to 10 significant digits so sampled values compare stably.

    Parameters
    values: float or [float]
        Value(s) to format.

    Returns
    List of formatted strings.
    """
    return [f"{float(v):.10g}" for v in np.atleast_1d(values)]


def clean(value):
    """
    Normalize a config value for comparison.

    Whitespace runs in strings collapse to one space (a single space stays
    significant), lists normalize elementwise, and numpy arrays become rounded
    lists; everything else passes through.

    Parameters
    value: any
        Config-dict value.

    Returns
    The normalized value.
    """
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return rounded(value)
    return value


def sample(fn, masses, coupling=None):
    """
    Sample a branching/ctau interpolation at the given points.

    Parameters
    fn: callable
        f(mass) when coupling is None, else a 2D spline evaluated with .ev.
    masses: np.array
        Mass sample points.
    coupling: float
        Coupling for a 2D interpolation, or None for a 1D one.

    Returns
    Rounded sample values.
    """
    if coupling is None:
        return rounded(fn(masses))
    return rounded(fn.ev(masses, np.full_like(masses, float(coupling))))


def sample_mixing(fn):
    """
    Sample a callable mixing(mass, coupling, p0) on a small (mass, energy) grid.

    Parameters
    fn: callable
        Mixing function, with p0.e the incident-particle energy.

    Returns
    Rounded sample values.
    """
    return rounded([fn(mass, 1.0, SimpleNamespace(e=energy))
                    for mass in (0.01, 0.05, 0.2, 1.0)
                    for energy in (10.0, 100.0, 1000.0)])


def production_config(model):
    """
    Per-channel production config, ready for comparison.

    The nsample field is dropped (a sampling knob a builder may legitimately
    raise above the notebook's exploratory value), the generator lists in
    production/configuration reduce to their first entry (the notebooks scan a
    single generator against the builders' full default lists; the leading one
    must agree), and a callable mixing is sampled rather than compared as an
    object.

    Parameters
    model: Model
        Model to reduce.

    Returns
    Dict mapping channel label to its cleaned config fields.
    """
    config = {}
    for label, channel in model.production.items():
        fields = {}
        for field, value in channel.items():
            if field == "nsample":
                continue
            if field in ("production", "configuration") and isinstance(value, (list, tuple)):
                value = value[:1]
            fields[field] = sample_mixing(value) if callable(value) else clean(value)
        config[label] = fields
    return config


def br_config(model):
    """
    Branching ratios: mode, final states, and sampled interpolation values.

    Parameters
    model: Model
        Model to reduce.

    Returns
    Dict with br_mode, the final-state map, and per-mode samples (over a
    coupling grid for a 2D branching).
    """
    couplings = (1.0, 10.0, 100.0) if model.br_mode == "2D" else (None,)
    return {
        "br_mode": model.br_mode,
        "finalstates": {str(mode): fs for mode, fs in model.br_finalstate.items()},
        "values": {str(mode): None if fn is None else [sample(fn, BR_MASSES, c) for c in couplings]
                   for mode, fn in model.br_functions.items()},
    }


def ctau_config(model):
    """
    Lifetime: coupling reference and sampled interpolation values.

    Parameters
    model: Model
        Model to reduce.

    Returns
    Dict with the ctau coupling reference and samples (over a coupling grid
    for a 2D ctau, whose coupling reference is None).
    """
    coupling_ref = getattr(model, "ctau_coupling_ref", None)
    fn = getattr(model, "ctau_function", None)
    couplings = (1e-5, 1e-4, 1e-3) if coupling_ref is None else (None,)
    return {
        "coupling_ref": coupling_ref,
        "values": None if fn is None else [sample(fn, CTAU_MASSES, c) for c in couplings],
    }


def scattering_config(model):
    """
    Scattering cross-section expression, coupling reference, and recoil window.

    Parameters
    model: Model
        Model to reduce.

    Returns
    Dict of the cleaned scattering attributes.
    """
    return {attr: clean(getattr(model, attr, None))
            for attr in ("dsigma_der", "dsigma_der_coupling_ref", "recoil_max")}


# ---------- the test ----------

@pytest.mark.parametrize("name", sorted(MODEL_DIRS))
def test_build_matches_notebook(name):
    """build.py reproduces the notebook's Model config exactly."""
    ref = reference_model(name)
    port = ported_model(name)
    assert production_config(port) == production_config(ref)
    assert br_config(port) == br_config(ref)
    assert ctau_config(port) == ctau_config(ref)
    assert scattering_config(port) == scattering_config(ref)
