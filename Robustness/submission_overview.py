import os
import numpy as np
decade = 0 # 0, 20,30,40,..
amplitude = str(1e-13)
rnds =  np.linspace(0,19,20)
for rnd in rnds:
    rnd = int(decade + rnd)
    os.system(f"python 2-Fit_Mock_Dataset.py --rnd {rnd} --amplitude {amplitude} --false_est False")