import gammapy 
print(f'loaded gammapy version: {gammapy.__version__} ' )
print(f'Supposed to be 1.0 (21-12-2022)' )
#get_ipython().system('jupyter nbconvert --to script 1-Nui_Par_Fitting.ipynb')
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import gammapy

# from gammapy.datasets import MapDataset
from gammapy.maps import Map
from astropy.coordinates import SkyCoord, Angle
from gammapy.modeling import Fit, Parameter, Parameters, Covariance
from gammapy.datasets import MapDataset #, MapDatasetNuisance
from gammapy.modeling.models import (
    PowerLawSpectralModel,
    create_crab_spectral_model,
    SkyModel,
    PointSpatialModel,
    ShellSpatialModel,
    GeneralizedGaussianSpatialModel,
    TemplateSpatialModel,
    LogParabolaSpectralModel,
    GaussianSpatialModel,
    DiskSpatialModel,
    PowerLawNormSpectralModel,
    Models,
    SpatialModel,
    FoVBackgroundModel,
)
from gammapy.estimators import TSMapEstimator, ExcessMapEstimator
from gammapy.maps import MapAxis
from gammapy.modeling.models.spectral import scale_plot_flux
from gammapy.estimators import  FluxPointsEstimator
    
from regions import CircleSkyRegion, RectangleSkyRegion
import yaml
import sys

#sys.path.append(
#    "/home/hpc/caph/mppi045h/3D_analysis/N_parameters_in_L/syserror_3d_bkgmodel/4-Fitting_nuisance_and_model_parameters"
#)
#from my_dataset_maps_20 import MapDatasetNuisance
#from MapDatasetNuisanceE import MapDatasetNuisanceE
#from  my_fit_20 import Fit
from Dataset_Creation import sys_dataset

source = 'Crab'
path = '/home/vault/caph/mppi062h/repositories/HESS_3Dbkg_syserror/2-error_in_dataset'
path_crab = '/home/hpc/caph/mppi045h/3D_analysis/N_parameters_in_L/nuisance_summary/Crab'
path_local = '/home/katrin/Documents'

try:
    models = Models.read(f"{path_crab}/standard_model.yml")
    dataset_standard = MapDataset.read(f'{path}/{source}/stacked.fits')

except:
    models = Models.read(f"{path_local}/{source}/standard_model.yml")
    dataset_standard = MapDataset.read(f'{path_local}/{source}/stacked.fits')

dataset_standard = dataset_standard.downsample(4)
model_spectrum  = PowerLawSpectralModel(
    index=2.3,
    amplitude="1e-12 TeV-1 cm-2 s-1",    )
source_model = SkyModel(spatial_model = models['main source'].spatial_model ,
                       spectral_model = model_spectrum,
                       name = "Source")    
models = Models(source_model)

bkg_model = FoVBackgroundModel(dataset_name=dataset_standard.name)
bkg_model.parameters['tilt'].frozen  = False
models.append(bkg_model)
dataset_standard.models = models

dataset_asimov = dataset_standard.copy()
dataset_asimov.counts = dataset_standard.npred()
models = Models(source_model.copy())
bkg_model = FoVBackgroundModel(dataset_name=dataset_asimov.name)
bkg_model.parameters['tilt'].frozen  = False
models.append(bkg_model)
dataset_asimov.models = models

    
dataset_standard.counts.sum_over_axes().plot(add_cbar=1)
binsize = dataset_standard.geoms["geom"].width[1] / dataset_standard.geoms["geom"].data_shape[1]
print(
    "spatial binsize = ",
    binsize
)

print(dataset_standard)


bias = 0.1
resolution = 0.8


N = 500
shift = 0
tilt = 0

save = True
save_flux = True
rnd = False



for n in range(N):
    print(n)
    resolution_rnd = np.random.uniform(resolution, 3*resolution, 1)
    bias_rnd = np.random.uniform(-3*bias, -bias, 1)
    print(f"bias_rnd:, {bias_rnd}, resolution_rnd: {resolution_rnd}" )
    sys_d_cor = sys_dataset(dataset_asimov= dataset_asimov,
                    shift = 0, 
                    tilt = 0,
                    bias = bias_rnd,
                    sigma = resolution_rnd,
                    rnd = rnd)
    dataset = sys_d_cor.create_dataset()
    dataset_N = sys_d_cor.create_dataset_N()
    zero = 1e-2
    penalising_invcovmatrix = np.zeros((4, 4))
    np.fill_diagonal(penalising_invcovmatrix, [1/zero**2, 1/zero**2, 1/bias_rnd**2, 1/resolution_rnd**2])
    dataset_N.penalising_invcovmatrix= penalising_invcovmatrix
    fit_cor = Fit(store_trace=False)
    if resolution_rnd >0:
        result_cor = fit_cor.run([dataset])
        result_cor = fit_cor.run([dataset_N])


    if save:
        with open("data/7a_P_draw_info.txt", "a") as myfile:
            myfile.write(str(float(resolution_rnd)) + '    '+ str(float(bias_rnd)) + '    ' +  str(float(dataset.stat_sum())) + '\n')

    stri = ""
    
    for p in ['amplitude', 'index', 'norm', 'tilt',]:
        stri += str(dataset.models.parameters[p].value)  + '   ' +  str(dataset.models.parameters[p].error)  + '   '
    print(stri)
    if save:
        with open("data/7a_P_draw_par.txt", "a") as myfile:
            myfile.write(stri + '\n')

    stri = ""
    for p in ['amplitude', 'index', 'norm', 'tilt', 'bias', 'resolution']:
        stri += str(dataset_N.models.parameters[p].value)  + '   ' +  str(dataset_N.models.parameters[p].error)  + '   '
    print(stri)
    if save:
        with open("data/7a_P_draw_par_N.txt", "a") as myfile:
            myfile.write(stri + '\n')

        
