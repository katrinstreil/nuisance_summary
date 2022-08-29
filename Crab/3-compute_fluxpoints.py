import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from gammapy.maps import Map
from astropy.coordinates import SkyCoord #, Angle
from gammapy.modeling import Fit, Parameter, Parameters#, Covariance
from gammapy.datasets import MapDataset #, MapDatasetNuisance
from gammapy.modeling.models import (
    PowerLawSpectralModel,
    SkyModel,
    PointSpatialModel,
    GaussianSpatialModel,
    Models,
    FoVBackgroundModel,
)


import yaml
import sys
import json

sys.path.append(
    "/home/hpc/caph/mppi045h/3D_analysis/N_parameters_in_L/syserror_3d_bkgmodel/4-Fitting_nuisance_and_model_parameters"
)
from my_dataset_maps_19 import MapDatasetNuisance
from  my_fit_19 import Fit
from gammapy.estimators import FluxPointsEstimator
from my_estimator_points_sed_19 import My_FluxPointsEstimator
from my_estimator_points_core_19 import My_FluxPoints
#definitons

source = 'Crab'
path = '/home/vault/caph/mppi062h/repositories/HESS_3Dbkg_syserror/2-error_in_dataset'
#path = '/home/hpc/caph/mppi045h/3D_analysis/N_parameters_in_L/syserror_3d_bkgmodel/2-source_dataset'
path_local_repo = '/home/hpc/caph/mppi045h/3D_analysis/N_parameters_in_L/syserror_3d_bkgmodel/2-source_dataset'

if source == "Crab":

    dataset_standard = MapDataset.read(f'{path}/{source}/stacked.fits')
    dataset_standard = dataset_standard.downsample(4)

    #dataset_standard = MapDataset.read(f'{path}/{source}/Crab_fullsys.fits')
    

    models = Models.read(f"standard_model.yml")
    
    bkg_model = FoVBackgroundModel(dataset_name=dataset_standard.name)
    bkg_model.parameters['tilt'].frozen  = False

    models.parameters['lon_0'].frozen = True
    models.parameters['lat_0'].frozen = True
    
    models.append(bkg_model)
    dataset_standard.models = models
    ebins_display = 6,9
    print(dataset_standard)

    

dataset_standard.counts.sum_over_axes().plot(add_cbar=1)
binsize = dataset_standard.geoms["geom"].width[1] / dataset_standard.geoms["geom"].data_shape[1]
print(
    "spatial binsize = ",
    binsize
)



added = "_008_624"
dataset_N= MapDatasetNuisance.read(f'{path_local_repo}/{source}/nui_dataset{added}.fits')
dataset_N.N_parameters
with open(f'{path_local_repo}/{source}/nui_par{added}.yml', "r") as ymlfile:
    nui_par = yaml.load(ymlfile, Loader=yaml.FullLoader)
dataset_N.N_parameters = Parameters.from_dict(nui_par )
bkg_model = FoVBackgroundModel(dataset_name=dataset_N.name)
with open(f'{path_local_repo}/{source}/nui_model{added}.yml', "r") as ymlfile:
    model_dict = yaml.load(ymlfile, Loader=yaml.FullLoader)
models = Models.from_dict(model_dict)
models.append(bkg_model)
dataset_N.models =models
print(dataset_N)

i_start, i_end = 6,9   
def plot_residual(dataset, max_ = None):
    res_standard = (
        dataset.residuals("diff/sqrt(model)")
        #.slice_by_idx(dict(energy=slice(6, 9)))
        .smooth(0.1 * u.deg)
        )
    if max_ is None:
        vmax = np.nanmax(np.abs(res_standard.data))
    else:
        vmax = max_
    res_standard.slice_by_idx(dict(energy=slice(i_start,i_end))).plot_grid(add_cbar=1, 
                                                                  vmax=vmax, vmin=-vmax, cmap="coolwarm")
    return res_standard            

plot_residual(dataset_standard)
plot_residual(dataset_N)
fluxpoint_bins = dataset_standard.geoms['geom'].axes[0].edges[6::5].value
fluxpoint_bins
print("Fluxpoint Standard" )
fluxpointsestimator_standard = FluxPointsEstimator(
        energy_edges=fluxpoint_bins * u.TeV ,
        source=0,
        norm_min=0.8,
        norm_max=1.2,
        norm_n_values=11,
        norm_values=None,
        n_sigma=1,
        n_sigma_ul=2,
        reoptimize=True,
        selection_optional=["errn-errp", "ul","scan"],
    )

fluxpoints_standard = fluxpointsestimator_standard.run(dataset_standard)
fluxfilename = f"{path_local_repo}/{source}/Flux_st.fits"
fluxpoints_standard.write(fluxfilename,overwrite= True )


    
fluxpointsestimator_N = My_FluxPointsEstimator(
        energy_edges=fluxpoint_bins * u.TeV ,
        source=0,
        norm_min=0.8,
        norm_max=1.2,
        norm_n_values=11,
        norm_values=None,
        n_sigma=1,
        n_sigma_ul=2,
        reoptimize=True,
        selection_optional=["errn-errp","ul", ],
    )

#dataset_N.N_parameters.freeze_all()
#for i in [145,146,147]:
#    dataset_N.N_parameters[i].frozen = False
fluxpoints_N = fluxpointsestimator_N.run([dataset_N])
    
fluxfilename = f"{path_local_repo}/{source}/Flux_N.fits"

fluxpoints_N.write(fluxfilename,overwrite= True )


