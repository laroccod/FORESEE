import numpy as np

from src.foresee import Model
from src.utils.utility import ensure_model_layout
from src.utils.detectors import default_detectors
from src.utils.utility import BREM_MASSES
import matplotlib.colors as mcolors


def build_model(
    path,
    energy="14",
    nsample_2body=100,
    generators_light=None,
    brem_configurations=None,
):
    """
    Build and return the FORESEE model object for this model.

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
        Hadron generators for light-meson production. Defaults to ['EPOSLHC', 'SIBYLL', 'QGSJET'], the
        canonical triple used for shower-model uncertainty envelopes.
    brem_configurations: [str]
        Which precomputed bremsstrahlung spectra (columns under
        files/direct/DarkPhoton/<E>TeV.txt.gz) to consume as the channel's
        variants. Defaults to ["Brem_QRA_L1.5", "Brem_QRA_L1.0", "Brem_QRA_L2.0"].

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

    model.add_production_direct(
        label="Brem",
        energy=energy,
        configuration=brem_configurations,
        coupling_ref=1,
        masses=BREM_MASSES,
    )

    masses_dy = [1.5849, 1.7783, 1.9953, 2.2387, 2.5119, 2.8184, 3.1623, 3.9811,
                 5.0119, 6.3096, 7.9433, 10.0, 12.0, 15.0, 17.0, 20.0, 25.0, 30.0,
                 50.0, 70.0, 100.0]
    model.add_production_direct(
        label="DY",
        energy=energy,
        coupling_ref=1,
        masses=masses_dy,
        condition="True",
    )

    model.set_ctau_1d(filename="model/ctau.txt")

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
        ("4pi_c",        None),
        ("4pi_n",        None),
        ("6pi_c",        None),
        ("6pi_n",        None),
        ("KKpipi_0",     None),
        ("KKpipi_1",     None),
        ("KKpipi_2",     None),
        ("KKpipi_3",     None),
        ("KK_c",         [321, -321]),
        ("KK_n",         [310, 130]),
        ("KKpi_0",       [130, 310, 111]),
        ("KKpi_1",       [321, -321, 111]),
        ("KKpi_2",       [321, -211, 311]),
        ("OmPiPi_c",     [223, 211, -211]),
        ("OmPiPi_n",     [223, 111, 111]),
        ("PhiPiPi_c",    [333, 211, -211]),
        ("PhiPiPi_n",    [333, 111, 111]),
        ("ppbar",        [2212, -2212]),
        ("nnbar",        [2112, -2112]),
    ]
    decay_modes = [mode for mode, _ in decay_channels]
    finalstates = [fs for _, fs in decay_channels]
    filenames = ["model/br/bfrac_" + mode + ".txt" for mode in decay_modes]
    model.set_br_1d(modes=decay_modes, finalstates=finalstates, filenames=filenames)

    return model
