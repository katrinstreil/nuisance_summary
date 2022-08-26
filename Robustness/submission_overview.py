import os
import numpy as np
decade = 80 # 0, 20,30,40,..
amplitude = str(1e-14)
rnds = np.linspace(0,9,10)
for rnd in rnds:
    rnd = int(decade + rnd)
    os.system(f"python  /home/hpc/caph/mppi045h/3D_analysis/N_parameters_in_L/nuisance_summary/Robustness/2-Fit_Mock_Dataset.py --rnd {rnd} --amplitude {amplitude} --false_est False ")