import numpy as np

from src.foresee import Model
from src.utils.utility import ensure_model_layout
from src.utils.detectors import default_detectors


BR_2BODY_KAON = (
    "{prefactor} * coupling**2 * np.sqrt("
    "(1-(mass+self.masses('pid1'))**2/self.masses('pid0')**2)*"
    "(1-(mass-self.masses('pid1'))**2/self.masses('pid0')**2)"
    ")"
)
BR_2BODY_B = "1.6e5 * coupling**2 * (1-(mass/self.masses('pid0'))**2)**2"

BR_3BODY = "{prefactor}*coupling**2*(energy**2-mass**2)**(3/2)"


def build_model(
    path,
    energy="14",
    nsample_2body=100,
    nsample_3body=100,
    generators_light=None,
    generators_heavy=None,
):
    """
    Build and return the FORESEE model object for this model.

    Parameters
    ----------
    path: str
        Model directory (Models/ALP-e/). Supplied by Foresee.load_model.
    energy: str
        Collider energy in TeV.
    nsample_2body, nsample_3body: int
        Sampling per production channel. Applied uniformly across all 16
        production channels.
    generators_light: [str]
        Generators for light-meson production. Defaults to ['EPOSLHC', 'SIBYLL', 'QGSJET'].
    generators_heavy: [str]
        Generators for heavy-meson production. Defaults to ['NLO-P8', 'NLO-P8-Max', 'NLO-P8-Min'].

    Returns
    -------
        Model
    """
    if generators_light is None:
        generators_light = ['EPOSLHC', 'SIBYLL', 'QGSJET']
    if generators_heavy is None:
        generators_heavy = ['NLO-P8', 'NLO-P8-Max', 'NLO-P8-Min']

    ensure_model_layout(path, link_direct=False)

    model = Model("ALP-e", path=path)

    for label, pid0, pid1, prefactor in [
        ("2body_321_211",  "321",  "211", "45"),
        ("2body_-321_-211", "-321", "-211", "45"),
        ("2body_130_111",  "130",  "111", "27"),
        ("2body_310_111",  "310",  "111", "0.3"),
    ]:
        model.add_production_2bodydecay(
            label=label,
            pid0=pid0,
            pid1=pid1,
            br=BR_2BODY_KAON.format(prefactor=prefactor),
            generator=generators_light,
            energy=energy,
            nsample=nsample_2body,
        )

    for label, pid0, pid1 in [
        ("2body_511_130",   "511",  "130"),
        ("2body_-511_130",  "-511", "130"),
        ("2body_521_321",   "521",  "321"),
        ("2body_-521_-321", "-521", "-321"),
        ("2body_531_333",   "531",  "333"),
        ("2body_-531_333",  "-531", "333"),
    ]:
        model.add_production_2bodydecay(
            label=label,
            pid0=pid0,
            pid1=pid1,
            br=BR_2BODY_B,
            generator=generators_heavy,
            energy=energy,
            nsample=nsample_2body,
        )

    for label, pid0, pid1, pid2, prefactor, gens in [
        ("3body_211_-11_-12",  "211",  "-11", "-12", "7.6e6", generators_light),
        ("3body_-211_11_12",   "-211", "11",  "12",  "7.6e6", generators_light),
        ("3body_321_-11_-12",  "321",  "-11", "-12", "9.9e4", generators_light),
        ("3body_-321_11_12",   "-321", "11",  "12",  "9.9e4", generators_light),
        ("3body_411_-11_-12",  "411",  "-11", "-12", "5.5e2", generators_heavy),
        ("3body_-411_11_12",   "-411", "11",  "12",  "5.5e2", generators_heavy),
        ("3body_431_-11_-12",  "431",  "-11", "-12", "7.9e3", generators_heavy),
        ("3body_-431_11_12",   "-431", "11",  "12",  "7.9e3", generators_heavy),
    ]:
        model.add_production_3bodydecay(
            label=label,
            pid0=pid0,
            pid1=pid1,
            pid2=pid2,
            br=BR_3BODY.format(prefactor=prefactor),
            generator=gens,
            energy=energy,
            nsample=nsample_3body,
            integration="dE",
        )

    model.set_ctau_1d(
        filename="model/ctau_total.txt",
        coupling_ref=1,
    )

    model.set_br_1d(
        modes=["e_e"],
        finalstates=[[11, -11]],
        filenames=["model/br/e_e.txt"],
    )

    return model
