import numpy as np

from src.foresee import Model
from src.utils.utility import ensure_model_layout
from src.utils.utility import BREM_MASSES


def build_model(
    path,
    energy="14",
    nsample_2body=100,
    nsample_3body=100,
    generators_light=None,
    generators_heavy=None,
    mass_DarkHiggs=2.0,
    theta_DarkHiggs=1e-4,
    brem_configurations=None,
):
    """
    Build and return the FORESEE model object for this model.

    Parameters
    ----------
    path: str
        Model directory (Models/DarkPhoton+DarkHiggs/). Supplied by Foresee.load_model.
    energy: str
        Collider energy in TeV. The notebook is set up for the HL-LHC.
    nsample_2body: int
        Sampling points for the A' meson-decay and bremsstrahlung channels.
    nsample_3body: int
        Sampling points for the dark-Higgs B-meson 3-body channels.
    generators_light: [str]
        Hadron generators for light-meson production. Defaults to ['EPOSLHC', 'SIBYLL', 'QGSJET'].
    generators_heavy: [str]
        Hadron generators for the heavy B-meson channels. Defaults to ['NLO-P8', 'NLO-P8-Max', 'NLO-P8-Min'].
    mass_DarkHiggs: float
        Dark Higgs mass [GeV]. Fixes the chain-decay resonance mass and the
        upper edge of each 3-body massrange (mass_DarkHiggs/2).
    theta_DarkHiggs: float
        Dark Higgs mixing angle, entering the B -> X_s phi branching prefactor.
    brem_configurations: [str]
        Bremsstrahlung spectrum columns to consume as the channel's variants.
        Defaults to ["Brem_QRA_L1.5", "Brem_QRA_L1.0", "Brem_QRA_L2.0"].

    Returns
    -------
        Model
    """
    if generators_light is None:
        generators_light = ["EPOSLHC", "SIBYLL", "QGSJET"]
    if generators_heavy is None:
        generators_heavy = ["NLO-P8", "NLO-P8-Max", "NLO-P8-Min"]
    if brem_configurations is None:
        brem_configurations = ["Brem_QRA_L1.5", "Brem_QRA_L1.0", "Brem_QRA_L2.0"]

    ensure_model_layout(path, link_direct=True, direct_name="DarkPhoton")

    model = Model("DarkPhoton+DarkHiggs", path=path)

    model.add_production_2bodydecay(
        pid0="111",
        pid1="22",
        br="2.*0.99 * coupling**2 * (1-mass**2/self.masses(pid0)**2)**3 ",
        generator=generators_light,
        energy=energy,
        nsample=nsample_2body,
    )
    model.add_production_2bodydecay(
        pid0="221",
        pid1="22",
        br="2.*0.39 * coupling**2 * (1-mass**2/self.masses(pid0)**2)**3 ",
        generator=generators_light,
        energy=energy,
        nsample=nsample_2body,
    )
    model.add_production_2bodydecay(
        pid0="331",
        pid1="22",
        br="2.*0.023 * coupling**2 * (1-mass**2/self.masses(pid0)**2)**3 ",
        generator=generators_light,
        energy=energy,
        nsample=nsample_2body,
    )
    model.add_production_2bodydecay(
        pid0="113",
        pid1="111",
        label="113_111",
        br=" coupling**2 * 4.7e-4 * ((self.masses('pid0')**2 - (self.masses('pid1') + mass)**2)*(self.masses('pid0')**2 - (self.masses('pid1') - mass)**2)/(self.masses('pid0')**2 - self.masses('pid1')**2)**2)**(3/2)",
        generator=generators_light,
        energy=energy,
        nsample=nsample_2body,
    )
    model.add_production_2bodydecay(
        pid0="113",
        pid1="221",
        label="113_221",
        br=" coupling**2 * 3.0e-4 * ((self.masses('pid0')**2 - (self.masses('pid1') + mass)**2)*(self.masses('pid0')**2 - (self.masses('pid1') - mass)**2)/(self.masses('pid0')**2 - self.masses('pid1')**2)**2)**(3/2)",
        generator=generators_light,
        energy=energy,
        nsample=nsample_2body,
    )
    model.add_production_2bodydecay(
        pid0="223",
        pid1="111",
        label="223_111",
        br=" coupling**2 * 8.33e-2 * ((self.masses('pid0')**2 - (self.masses('pid1') + mass)**2)*(self.masses('pid0')**2 - (self.masses('pid1') - mass)**2)/(self.masses('pid0')**2 - self.masses('pid1')**2)**2)**(3/2)",
        generator=generators_light,
        energy=energy,
        nsample=nsample_2body,
    )
    model.add_production_2bodydecay(
        pid0="223",
        pid1="221",
        label="223_221",
        br=" coupling**2 * 4.5e-4 * ((self.masses('pid0')**2 - (self.masses('pid1') + mass)**2)*(self.masses('pid0')**2 - (self.masses('pid1') - mass)**2)/(self.masses('pid0')**2 - self.masses('pid1')**2)**2)**(3/2)",
        generator=generators_light,
        energy=energy,
        nsample=nsample_2body,
    )
    model.add_production_2bodydecay(
        pid0="333",
        pid1="221",
        br=" coupling**2 * 1.306e-2 * ((self.masses('pid0')**2 - (self.masses('pid1') + mass)**2)*(self.masses('pid0')**2 - (self.masses('pid1') - mass)**2)/(self.masses('pid0')**2 - self.masses('pid1')**2)**2)**(3/2)",
        generator=generators_light,
        energy=energy,
        nsample=nsample_2body,
    )

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

    DARKHIGGS_CHANNELS = [
        ("511", "130"),
        ("-511", "130"),
        ("521", "321"),
        ("-521", "-321"),
    ]
    for pid0, pid1 in DARKHIGGS_CHANNELS:
        model.add_production_3bodydecay(
            pid0=pid0,
            pid1=pid1,
            pid2="0",
            integration="chain_decay",
            br=[
                "2*5.6 * " + str(theta_DarkHiggs)
                + "**2 * pow(1.-pow(mI/self.masses(" + pid0 + "),2),2)",
                str(mass_DarkHiggs),
            ],
            generator=generators_heavy,
            energy=energy,
            nsample=nsample_3body,
            massrange=[0, mass_DarkHiggs / 2],
            scaling=0,
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
