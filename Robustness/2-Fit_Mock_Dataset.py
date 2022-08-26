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
#definitons


import argparse
CLI=argparse.ArgumentParser()
CLI.add_argument( "--rnd" )  
CLI.add_argument( "--amplitude" )  
CLI.add_argument( "--false_est" )   


args = CLI.parse_args()
input_ = dict()
input_['rnd'] = int(args.rnd)
input_['amplitude'] = float(args.amplitude)
if str(args.false_est) == "False":
    input_['false_est']   = False
if str(args.false_est) == "True":
    input_['false_est']   = True
    
rnd = input_['rnd']
amplitude =input_['amplitude']* u.Unit('cm-2 s-1 TeV-1')
false_est = input_['false_est']

print("...." * 20)
print("...." * 20)
print("...." * 20)
print("RND.................", input_['rnd'])
print("amplitude...........", amplitude)
print("false_est...........", false_est)

print("...." * 20)
print("...." * 20)
print("...." * 20)


pos_frozen = True
spatial_model_type = "pointsource_center"
outputfolder = '/home/hpc/caph/mppi045h/3D_analysis/N_parameters_in_L/nuisance_summary/Robustness/output/data_asimov_tests'
inputfolder = '/home/hpc/caph/mppi045h/3D_analysis/N_parameters_in_L/nuisance_summary/Robustness'


if spatial_model_type == "pointsource_center":
    outputfile = '/OOutput'+str(amplitude.value)+'.json'
    
with open(outputfolder+outputfile, 'r') as f:
    data = json.load(f)
j = 0
rnds = list(data.keys()) 

print("setting to started ..")
print("fitting rnd dataset number:", rnd)
data[str(rnd)]['started'] = True

with open(outputfolder+outputfile, 'w') as fp:
    json.dump(data, fp, indent=4)

def plot_nui_distribution(nui_values_co, sys, mus, stds):
    ii = i_end - i_start
    ii = ii // 3 + 1
    fig, axs = plt.subplots(ii,3, figsize =(10,ii* 3) )
    for i,e in enumerate(xaxis[i_start:i_end]):
        ax = axs.flatten()[i]
        mu, sigma_ = mus[i+i_start], stds[i+i_start]  # mean and standard deviation
        s = nui_values_co[i*amount_free_par : (i+1)* amount_free_par]
        count, bins, ignored = ax.hist(s, 20, density=False, alpha = 0.3, color = 'red',)
        #ax.set_xlim(-10,10)
        ax.set_title(f'Energy: {xaxis[i+i_start].value:.2} TeV')
        ax.plot(bins,   max(count) *
                                np.exp( - (bins - mu)**2 / (2 * sigma_**2) ),
                          linewidth=2, color='r')
        ylim = ax.get_ylim()
        
        ax.vlines(0 +sys[i+i_start] *100, ylim[0], ylim[1],color = 'red' )
        ax.vlines(0 -sys[i+i_start] *100 , ylim[0], ylim[1],color = 'red')
    plt.tight_layout()
        
def plot_corr_matrix(dataset):
    M = np.linalg.inv(dataset.inv_corr_matrix)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(M)  # interpolation='nearest')
    fig.colorbar(cax);
    print("Maximal expected sys amplitude in % of bg:", np.sqrt( M.max() ) * 100)

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

def compute_K_matrix(l_deg, sigma, ndim_spatial_nui, ndim_spectral_nui,geom_down ):
    helper_map = Map.from_geom(geom_down).slice_by_idx(dict(energy=slice(0, 1)))
    helper_map2 = helper_map.copy()
    ndim_spatial_nui_1D = int(np.sqrt(ndim_spatial_nui))
    corr_matrix_spatial = np.identity(ndim_spatial_nui)
    for b_0 in range(ndim_spatial_nui_1D):
        for l_0 in range(ndim_spatial_nui_1D):
            i = b_0 * ndim_spatial_nui_1D + l_0
            C = SkyCoord(
                helper_map.geom.pix_to_coord((l_0, b_0, 0))[0],
                helper_map.geom.pix_to_coord((l_0, b_0, 0))[1],
                frame=geom_down.frame,
            )
            helper_map.data[0, :, :] = C.separation(
                geom_down.to_image().get_coord().skycoord
            ).value
            helper_map2.data = np.zeros(ndim_spatial_nui_1D ** 2).reshape(
                helper_map2.geom.data_shape
            )
            helper_map2.data[0, :, :] = np.exp(
                -0.5 * helper_map.data[0, :, :] ** 2 / l_deg ** 2
            )
            corr_matrix_spatial[i, :] = helper_map2.data.flatten()

    corr_matrix_spectral = np.identity(ndim_spectral_nui)
    for e in range((ndim_spectral_nui)):
        corr_matrix_spectral[e, e] = sigma[e] ** 2
    return np.kron(corr_matrix_spectral, corr_matrix_spatial)

with open(inputfolder+'/0_estimate_sys_per_ebin.yml', "r") as ymlfile:
    sys_read = yaml.load(ymlfile, Loader=yaml.FullLoader)
mus = sys_read['mus']
stds = sys_read['stds']

path_local_repo = '/home/saturn/caph/mppi045h/Nuisance_Asimov_Datasets'

def read_mock_dataset(ad ):
    added = "_" + str(ad)
    dataset_N_sys = MapDatasetNuisance.read(f'{path_local_repo}/nui_dataset{added}.fits')
    with open(f'{path_local_repo}/nui_par{added}.yml', "r") as ymlfile:
        nui_par = yaml.load(ymlfile, Loader=yaml.FullLoader)
    dataset_N_sys.N_parameters = Parameters.from_dict(nui_par )
    bkg_model = FoVBackgroundModel(dataset_name=dataset_N_sys.name)
    models = Models([])
    models.append(bkg_model)
    dataset_N_sys.models =models
    return dataset_N_sys

dataset_N_sys_ex = read_mock_dataset(0)
emask = dataset_N_sys_ex.nuisance_mask.data.sum(axis=2).sum(axis=1) >0
xaxis = dataset_N_sys_ex.geoms['geom'].axes[0].center
i_start = 6
i_end = i_start + sum(emask)

downsampling_factor = 10
ndim_spatial_nui = dataset_N_sys_ex.geoms['geom_down'].data_shape[1] **2
print("# of Nuis per ebin:", ndim_spatial_nui)
l_corr, ndim_spectral_nui = 0.08,  i_end -i_start 
print("# of Ebins with Nuis:", ndim_spectral_nui)
bg = dataset_N_sys_ex.background
bg_e = bg.data.sum(axis=2).sum(axis=1)
amount_free_par = ndim_spatial_nui

models = Models.read(inputfolder+"/1a-Source_models.yaml")
model_asimov = models[spatial_model_type]
model_asimov.parameters['amplitude'].value = amplitude.value
print(model_asimov)

def create_dataset_fitting(dataset, mus, stds, falseningfactor):
    Nuisance_parameters_correct = Parameters([Parameter(name = par.name, value =0 ,frozen = par.frozen)  
      for  par in dataset.N_parameters])

    dataset_fitting = MapDatasetNuisance(
        background=dataset.background,
        exposure=dataset.exposure,
        psf=dataset.psf,
        edisp=dataset.edisp,
        mask_fit=dataset.mask_fit,
        mask_safe=dataset.mask_safe,
        inv_corr_matrix=dataset.inv_corr_matrix,
        N_parameters=Nuisance_parameters_correct,
        nuisance_mask=dataset.nuisance_mask,
    )

    bkg_model = FoVBackgroundModel(dataset_name=dataset_fitting.name)
    bkg_model.parameters["tilt"].frozen = False
    models = Models(model_asimov) 
    models.append(bkg_model)
    dataset_fitting.models = models
    
    if pos_frozen:
        dataset_fitting.models.parameters['lon_0'].frozen = True
        dataset_fitting.models.parameters['lat_0'].frozen = True
    
    ## Add systematic 
    sys_map = dataset.N_map().copy()
    for e in range(24):
        ex = dataset.exposure
        ex_ = ex.slice_by_idx(dict(energy_true= slice(e, e+1)))
        ex_.data = ex_.data / np.max(ex_.data)
        sys_map.slice_by_idx(dict(energy= slice(e, e+1))).data *= ex_.data
    sys_map.plot_grid(add_cbar = 1)
    dataset_fitting.counts = Map.from_geom(dataset_fitting.geoms['geom'])
    dataset_fitting.counts.data =  dataset.background.data * (1+sys_map.data) 
    
    ## Add Source
    dataset_fitting.counts.data += dataset_fitting.npred_signal()
    
    sys = (np.abs(mus) + falseningfactor* np.array(stds)) /100
    print(sys)
    correlation_matrix_co = compute_K_matrix(l_corr, np.array(sys[i_start:i_end]), 
                                             ndim_spatial_nui,
                                             ndim_spectral_nui, 
                                      dataset_N_sys_ex.geoms['geom_down'])
    dataset_fitting.inv_corr_matrix=np.linalg.inv(correlation_matrix_co)
    
    return dataset_fitting

dataset_N_sys = read_mock_dataset(rnd)
dataset_A_fitting = create_dataset_fitting (dataset_N_sys, mus, stds, 1)

plot_residual(dataset_A_fitting)

print("Fitting Standard" )
# "semi standard"
fit_N = Fit(store_trace=False)
with dataset_A_fitting.N_parameters.restore_status():
    dataset_A_fitting.N_parameters.freeze_all()
    result_standard = fit_N.run([dataset_A_fitting])

L_statsum_standard= dataset_A_fitting.stat_sum()
print(f"best fit {L_statsum_standard}")
with dataset_A_fitting.models.parameters.restore_status():
    dataset_A_fitting.models.parameters['amplitude'].value = 0 
    L_statsum_0_standard = dataset_A_fitting.stat_sum()
    TS_standard =   L_statsum_0_standard - L_statsum_standard
print(f"0        {L_statsum_0_standard}")
print(f"TS: {TS_standard}")
best_fit_model_standard = dataset_A_fitting.models[0].copy()
best_fit_bgmodel_standard = dataset_A_fitting.models[1].copy(name='bestfit')

print(best_fit_model_standard.parameters.to_table())
print(best_fit_bgmodel_standard.parameters.to_table())

print()
print()
print()

print("Fitting Corr Est." )
fit_N = Fit(store_trace=False)
#dataset_A_fitting.N_parameters.freeze_all()
#for i in [145,146,147]:
#    dataset_A_fitting.N_parameters[i].frozen = False
result_N = fit_N.run([dataset_A_fitting])
L_statsum_N= dataset_A_fitting.stat_sum()
print(f"best fit {L_statsum_N}")
with dataset_A_fitting.models.parameters.restore_status():
    dataset_A_fitting.models.parameters['amplitude'].value = 0 
    L_statsum_0_N = dataset_A_fitting.stat_sum()
    TS_N =   L_statsum_0_N - L_statsum_N
print(f"0        {L_statsum_0_N}")
print(f"TS: {TS_N}")

best_fit_model_N = dataset_A_fitting.models[0].copy()
best_fit_bgmodel_N = dataset_A_fitting.models[1].copy(name='bestfit')

print(best_fit_model_N.parameters.to_table())
print(best_fit_bgmodel_N.parameters.to_table())
print()
print()
print()


dict_nuis = dict()
dict_nuis['corr'] = dataset_A_fitting.N_parameters.to_dict()

with open(f'{path_local_repo}/OOutput{amplitude.value}/nui_par_{rnd}.yml', 'w') as outfile:
        yaml.dump(dict_nuis, outfile, default_flow_style=False)

        
import json

with open(outputfolder+outputfile, 'r') as f:
    data = json.load(f)
    
result = dict()


result["L_statsum_N"] = L_statsum_N
result["L_statsum_0_N"] = L_statsum_0_N
result["TS_N"] =  TS_N

result["L_statsum_standard"] = L_statsum_standard
result["L_statsum_0_standard"] = L_statsum_0_standard
result["TS_standard"] =  TS_standard


result["success_standard"] = result_standard.success
result["success_N"] =result_N.success


par_names = best_fit_model_standard.parameters.names
for par_name in par_names:
    result["best_fit_"+par_name+"_standard"] = best_fit_model_standard.parameters[par_name].value
    result["best_fit_"+par_name+"_N"] = best_fit_model_N.parameters[par_name].value

    result["best_fit_"+par_name+"_error_standard"] = best_fit_model_standard.parameters[par_name].error
    result["best_fit_"+par_name+"_error_N"] = best_fit_model_N.parameters[par_name].error

    
par_names = best_fit_bgmodel_standard.parameters.names
for par_name in par_names:
    result["best_fit_"+par_name+"_standard"] = best_fit_bgmodel_standard.parameters[par_name].value
    result["best_fit_"+par_name+"_N"] = best_fit_bgmodel_N.parameters[par_name].value

    result["best_fit_"+par_name+"_error_standard"] = best_fit_bgmodel_standard.parameters[par_name].error
    result["best_fit_"+par_name+"_error_N"] = best_fit_bgmodel_N.parameters[par_name].error


data[str(rnd)]['result'] = result
with open(outputfolder+outputfile, 'w') as fp:
    json.dump(data, fp, indent=4)
    
    
    
if false_est:
    dataset_A_fitting_under = create_dataset_fitting (dataset_N_sys, mus, stds, 0.5)
    dataset_A_fitting_over = create_dataset_fitting (dataset_N_sys, mus, stds, 2)
    print("Fitting Under Est." )
    fit_N = Fit(store_trace=False)
    #dataset_A_fitting_under.N_parameters.freeze_all()
    #for i in [145,146,147]:
    #    dataset_A_fitting_under.N_parameters[i].frozen = False
    result_N_under = fit_N.run([dataset_A_fitting_under])
    L_statsum_N_under= dataset_A_fitting_under.stat_sum()
    print(f"best fit {L_statsum_N_under}")
    with dataset_A_fitting_under.models.parameters.restore_status():
        dataset_A_fitting_under.models.parameters['amplitude'].value = 0 
        L_statsum_0_N_under = dataset_A_fitting_under.stat_sum()
        TS_N_under =   L_statsum_0_N_under - L_statsum_N_under
    print(f"0        {L_statsum_0_N_under}")
    print(f"TS: {TS_N_under}")

    best_fit_model_N_under = dataset_A_fitting_under.models[0].copy()
    best_fit_bgmodel_N_under = dataset_A_fitting_under.models[1].copy(name='bestfit')

    print(best_fit_model_N_under.parameters.to_table())
    print(best_fit_bgmodel_N_under.parameters.to_table())
    print()
    print()
    print()



    print("Fitting Over Est." )
    fit_N = Fit(store_trace=False)
    #dataset_A_fitting_over.N_parameters.freeze_all()
    #for i in [145,146,147]:
    #    dataset_A_fitting_over.N_parameters[i].frozen = False
    result_N_over = fit_N.run([dataset_A_fitting_over])
    L_statsum_N_over= dataset_A_fitting_over.stat_sum()
    print(f"best fit {L_statsum_N_over}")
    with dataset_A_fitting_over.models.parameters.restore_status():
        dataset_A_fitting_over.models.parameters['amplitude'].value = 0 
        L_statsum_0_N_over = dataset_A_fitting_over.stat_sum()
        TS_N_over =   L_statsum_0_N_over - L_statsum_N_over
    print(f"0        {L_statsum_0_N_over}")
    print(f"TS: {TS_N_over}")

    best_fit_model_N_over = dataset_A_fitting_over.models[0].copy()
    best_fit_bgmodel_N_over = dataset_A_fitting_over.models[1].copy(name='bestfit')

    print(best_fit_model_N_over.parameters.to_table())
    print(best_fit_bgmodel_N_over.parameters.to_table())


    dict_nuis['under'] = dataset_A_fitting_under.N_parameters.to_dict()
    dict_nuis['over'] = dataset_A_fitting_over.N_parameters.to_dict()

    with open(f'{path_local_repo}/OOutput{amplitude.value}/nui_par_{rnd}.yml', 'w') as outfile:
            yaml.dump(dict_nuis, outfile, default_flow_style=False)


    with open(outputfolder+outputfile, 'r') as f:
        data = json.load(f)
    
    result["L_statsum_N_under"] = L_statsum_N_under
    result["L_statsum_0_N_under"] = L_statsum_0_N_under
    result["TS_N_under"] =  TS_N_under

    result["L_statsum_N_over"] = L_statsum_N_over
    result["L_statsum_0_N_over"] = L_statsum_0_N_over
    result["TS_N_over"] =  TS_N_over

    result["success_N_under"] =result_N_under.success
    result["success_N_over"] =result_N_over.success


    par_names = best_fit_model_standard.parameters.names
    for par_name in par_names:
        result["best_fit_"+par_name+"_N_under"] = best_fit_model_N_under.parameters[par_name].value
        result["best_fit_"+par_name+"_N_over"] = best_fit_model_N_over.parameters[par_name].value


        result["best_fit_"+par_name+"_error_N_under"] = best_fit_model_N_under.parameters[par_name].error
        result["best_fit_"+par_name+"_error_N_over"] = best_fit_model_N_over.parameters[par_name].error


    par_names = best_fit_bgmodel_standard.parameters.names
    for par_name in par_names:
        result["best_fit_"+par_name+"_N_under"] = best_fit_bgmodel_N_under.parameters[par_name].value
        result["best_fit_"+par_name+"_N_over"] = best_fit_bgmodel_N_over.parameters[par_name].value


        result["best_fit_"+par_name+"_error_N_under"] = best_fit_bgmodel_N_under.parameters[par_name].error
        result["best_fit_"+par_name+"_error_N_over"] = best_fit_bgmodel_N_over.parameters[par_name].error


    data[str(rnd)]['result'] = result
    with open(outputfolder+outputfile, 'w') as fp:
        json.dump(data, fp, indent=4)
    
    