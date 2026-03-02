#Contains tests for the Foresee class, inheriting Utility

#To run the tests, make sure pytest is installed:
#  python3 -m pip install pytest
#Then do
#  pytest test_Foresee.py

import sys, os
src_path = "../"
sys.path.append(src_path)

from src.foresee import Utility,Model,Foresee,LorentzVector,Decay
import pytest
import numpy as np

foresee = Foresee(path=src_path)
foresee.rng.seed(137)

#@pytest.mark.skip  #Uncomment decorator to disable this test
def test_threebody_decay_pure_phase_space():
    ref = [{'px':-4.524630, 'py': 37.565497, 'pz': 12.005886, 'e':60.508054},\
           {'px':11.575480, 'py':  5.151148, 'pz':  6.659295, 'e':26.948718},\
           {'px':-7.050850, 'py':-42.716645, 'pz':-18.665182, 'e':49.543227},]
    m0 = 137.
    p0=LorentzVector(px=0., py=0., pz=0., e=m0)
    ret = foresee.threebody_decay_pure_phase_space(p0=p0, m0=m0, m1=m0/3., m2=m0/6., m3=m0/9.)
    assert len(ret)==len(ref)
    for a,b in zip(ref,ret):
        assert np.isclose(a['px'], b.px, rtol=0.001)
        assert np.isclose(a['py'], b.py, rtol=0.001)
        assert np.isclose(a['pz'], b.pz, rtol=0.001)
        assert np.isclose(a['e' ], b.e,  rtol=0.001)
        
