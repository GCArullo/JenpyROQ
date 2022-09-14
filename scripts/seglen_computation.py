# Copied from https://pypi.org/project/gw-frequencies/#description, see its documentation for a description of the code.

import numpy as np

SEGLEN_20_HZ = 157.86933774

def seglen_from_freq(f_0,
                     m_tot=2.8,
                     maximum_mass_ratio=4.0,
                     power_of_two=False,
                     margin_percent=5.0,
                     ):
    
    eta = maximum_mass_ratio / (1 + maximum_mass_ratio) ** 2
    
    seglen = (
              SEGLEN_20_HZ * (f_0 / 20) ** (-8 / 3) * (m_tot / 2.8) ** (5 / 3) / (4 * eta)
              ) * (1 + margin_percent / 100)
              
    return 2 ** (np.ceil(np.log2(seglen))) if power_of_two else seglen


print(seglen_from_freq(20, m_tot= 2.8, margin_percent=0.0, maximum_mass_ratio=3.0))
