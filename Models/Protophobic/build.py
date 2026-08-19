import numpy as np

from src.foresee import Model
from src.utils.utility import ensure_model_layout, BREM_MASSES
import src.foresee as foresee_module


VP_KIN = (
    "((self.masses('pid0')**2 - (self.masses('pid1') + mass)**2)"
    "*(self.masses('pid0')**2 - (self.masses('pid1') - mass)**2)"
    "/(self.masses('pid0')**2 - self.masses('pid1')**2)**2)**(3/2)"
)


def BW(self, mass, pid):
    return 1 / (1 - (mass / self.masses(pid))**2 - 1j * self.widths(pid) / self.masses(pid))


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
        Model directory (Models/Protophobic/). Supplied by Foresee.load_model.
    energy: str
        Collider energy in TeV. Defaults to "14".
    nsample_2body: int
        Sampling points per production channel. Defaults to 100.
    generators_light: [str]
        Hadron generators for meson production. Defaults to
        ['EPOSLHC', 'SIBYLL', 'QGSJET'], the canonical triple for the shower-model uncertainty envelope.
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

    foresee_module.BW = BW

    model = Model("Protophobic", path=path)

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
    model.add_production_2bodydecay(
        pid0="333",
        pid1="221",
        br=" 4. * (coupling/0.303)**2 * 1.306e-2 * " + VP_KIN,
        generator=generators_light,
        energy=energy,
        nsample=nsample_2body,
    )

    model.add_production_direct(
        label="Brem",
        energy=energy,
        configuration=brem_configurations,
        coupling_ref=0.303,
        masses=BREM_MASSES,
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
