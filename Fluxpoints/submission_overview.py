import os
import numpy as np
decade = 0 # 0, 20,30,40,..
amplitude = str(1e-10)
rnds = np.linspace(0,1,1)
rnds = [2.]
for rnd in rnds:
    rnd = int(decade + rnd)
    os.system(f"python  /home/hpc/caph/mppi045h/3D_analysis/N_parameters_in_L/nuisance_summary/Fluxpoints/1-Compute_Fluxpoints.py --rnd {rnd} --amplitude {amplitude} --false_est False ")