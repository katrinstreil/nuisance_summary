#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gammapy 
print(f'loaded gammapy version: {gammapy.__version__} ' )
print(f'Supposed to be 1.0 (21-12-2022)' )


# In[2]:


#get_ipython().system('jupyter nbconvert --to script 1-Nui_Par_Fitting.ipynb')
import pyximport

pyximport.install()
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


# In[ ]:





# ## Read dataset

# In[3]:


source = 'Crab'
path = '/home/vault/caph/mppi062h/repositories/HESS_3Dbkg_syserror/2-error_in_dataset'
path_crab = '/home/hpc/caph/mppi045h/3D_analysis/N_parameters_in_L/nuisance_summary/Crab'


dataset_standard = MapDataset.read(f'{path}/{source}/stacked.fits')
dataset_standard = dataset_standard.downsample(4)
models = Models.read(f"{path_crab}/standard_model.yml")
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


# ## Varying the Exposure

# In[4]:


factor = +0.1
N = 100
sigma = 0.1


# In[5]:


datasets = []
for n in range(N):
    sys_d_cor = sys_dataset(dataset_asimov, factor, True)
    dataset = sys_d_cor.create_dataset()
    datasets.append(dataset)
sys_d_cor = sys_dataset(dataset_asimov, factor, False)
dataset_asimov = sys_d_cor.create_dataset()
dataset_asimov_N = sys_d_cor.create_dataset_N(sigma)


# In[ ]:


for n in range(N):
    print(n)
    fit_cor = Fit(store_trace=False)
    result_cor = fit_cor.run([datasets[n]])
result_cor = fit_cor.run([dataset_asimov])    
result_cor = fit_cor.run([dataset_asimov_N])    


dataset_asimov_N.models[0].parameters['amplitude_nuisance'].frozen = True


# In[ ]:


print("best fit amplitude")
amplitudes = []
print('====')
print('Input')
print('====')
for d in datasets:
    m = d.models['Source']
    value = m.parameters['amplitude'].value 
    error = m.parameters['amplitude'].error
    amplitudes.append([value, error])
    print(f'{value} pm {error}')
print()
print("best fit Index")
indices = []
print('====')
print('Input')
print('====')
for d in datasets:
    m = d.models['Source']
    value = m.parameters['index'].value 
    error = m.parameters['index'].error
    indices.append([value, error])
    print(f'{value} pm {error}')
print()

print("Best Fit bkg Norm")
norms = []
print('====')
print('Input')
print('====')
for d in datasets:
    value = d.background_model.parameters['norm'].value 
    error = d.background_model.parameters['norm'].error
    print(f'{value} pm {error}')
    norms.append([value, error])
    
norms = np.array(norms)
indices = np.array(indices)
amplitudes = np.array(amplitudes)
    
with open('data/norms_poission.yaml', 'w') as file:
    documents = yaml.dump(norms, file)
with open('data/amplitudes_poission.yaml', 'w') as file:
    documents = yaml.dump(amplitudes, file)
with open('data/indices_poission.yaml', 'w') as file:
    documents = yaml.dump(indices, file)
    
#dataset_neg.models.write("data/1_model_neg.yml", overwrite= True)
#dataset_pos.models.write("data/1_model_pos.yml", overwrite= True)
#dataset_cor.models.write("data/1_model_cor.yml", overwrite= True)


# In[ ]:


fig, axs = plt.subplots(3,1)

axs[0].set_title("BKG Norm")
axs[1].set_title("Signal Amplitude")
axs[2].set_title("Signal Index")


labels = [ '-10%', '0%', '10%']


for d in [dataset_asimov, dataset_asimov_N]:
    
    best = d.models[1].parameters['norm'].value
    er = d.models[1].parameters['norm'].error
    axs[0].plot( [best] * N, color  = 'blue', alpha = 0.3, label = "Asimov")
    axs[0].fill_betweenx( [best-er , best + er], 0.5, N+0.5, color  = 'blue', alpha = 0.3,
                        label = "Asimov error")
    
    best = d.models[0].parameters['amplitude'].value
    er = d.models[0].parameters['amplitude'].error
    axs[1].fill_betweenx( [best-er , best + er], 0.5, N+0.5, color  = 'blue', alpha = 0.3)
    axs[1].plot( [best] * N, color  = 'blue', alpha = 0.3, label = "Asimov")

    best = d.models[0].parameters['index'].value
    er = d.models[0].parameters['index'].error
    axs[2].fill_betweenx( [best-er , best + er], 0.5, N+0.5, color  = 'blue', alpha = 0.3)
    axs[2].plot( [best] * N, color  = 'blue', alpha = 0.3, label = "Asimov")

for i, a in enumerate(norms[1:]):
    axs[0].errorbar(x= i+1, y = a[0], yerr = a[1], fmt= 'o', color =  'red',markersize=7)
axs[0].hlines( norms[0][0], 1, N, label = "input", color = 'black')


for i, a in enumerate(amplitudes[1:]):
    axs[1].errorbar(x= i+1, y = a[0], yerr = a[1], fmt= 'o', color =  'red',markersize=7)
axs[1].hlines( amplitudes[0][0], 1, N, label = "input", color ='black')


for i, a in enumerate(indices[1:]):
    axs[2].errorbar(x= i+1, y = a[0], yerr = a[1], fmt= 'o', color =  'red',markersize=7)
axs[2].hlines( indices[0][0], 1, N, label = "input", color ='black')



plt.tight_layout()
for a in axs:
    a.legend(); 
    a.set_ylabel('Best Fit'); 

plt.tight_layout()    
fig.savefig(f"plots/6_best_fit_{factor}_{N}.pdf")   


# In[ ]:



valuies = [norms, amplitudes, indices]

valuies_asimov = [[dataset_asimov.models[1].parameters['norm'].value, 
                   dataset_asimov.models[1].parameters['norm'].error],
                  [dataset_asimov.models[0].parameters['amplitude'].value, 
                   dataset_asimov.models[0].parameters['amplitude'].error],
                  [dataset_asimov.models[0].parameters['index'].value, 
                   dataset_asimov.models[0].parameters['index'].error],]


valuies_asimov_N = [[dataset_asimov_N.models[1].parameters['norm'].value, 
                   dataset_asimov_N.models[1].parameters['norm'].error],
                  [dataset_asimov_N.models[0].parameters['amplitude'].value, 
                   dataset_asimov_N.models[0].parameters['amplitude'].error],
                  [dataset_asimov_N.models[0].parameters['index'].value, 
                   dataset_asimov_N.models[0].parameters['index'].error],]
is_within_norm = []
is_within_amplitude = []
is_within_index = []

i = 0
for n in norms:
    larger = np.all(n[0] >= (valuies_asimov[0][0] - valuies_asimov[0][1]))
    smaller = np.all(n[0] <= (valuies_asimov[0][0] + valuies_asimov[0][1]))
    is_within_norm.append(larger and smaller)
    
for a in amplitudes:
    larger = np.all(a[0] >= (valuies_asimov[1][0] - valuies_asimov[1][1]))
    smaller = np.all(a[0] <= (valuies_asimov[1][0] + valuies_asimov[1][1]))
    is_within_amplitude.append(larger and smaller)
    
for i in indices:
    larger = np.all(i[0] >= (valuies_asimov[2][0] - valuies_asimov[2][1]))
    smaller = np.all(i[0] <= (valuies_asimov[2][0] + valuies_asimov[2][1]))
    is_within_index.append(larger and smaller)
print(is_within_norm )


# In[ ]:


fig, axs = plt.subplots(3,1)

axs[0].set_title(f"BKG Norm: {np.count_nonzero(is_within_norm)/N}")
axs[1].set_title(f"Signal Amplitude: {np.count_nonzero(is_within_amplitude)/N}")
axs[2].set_title(f"Signal Index: {np.count_nonzero(is_within_index)/N}")




for i, v in enumerate(valuies):
    axs[i].hist(v[1:,0], color ='red')
    ylim = axs[i].get_ylim()
    
    axs[i].vlines(valuies_asimov_N[i][0], ylim[0], ylim[1], color = 'green')
    axs[i].fill_between([valuies_asimov_N[i][0] - valuies_asimov_N[i][1],
                         valuies_asimov_N[i][0] + valuies_asimov_N[i][1]]
                        , ylim[0], ylim[1],
                       alpha = 0.4, color = 'green')
    
    axs[i].vlines(valuies_asimov[i][0], ylim[0], ylim[1], color = 'red')
    axs[i].fill_between([valuies_asimov[i][0] - valuies_asimov[i][1],
                         valuies_asimov[i][0] + valuies_asimov[i][1]]
                        , ylim[0], ylim[1],
                       alpha = 0.4 , color = 'red')
    
    
    
    
    
plt.tight_layout()
fig.savefig(f"plots/6_histo_{factor}_{N}.pdf")


# In[ ]:


mask = datasets[0].mask.data[:,60,60]

ebins = datasets[0].counts.geom.axes[0].center[mask]
print(len(ebins))
asimov, asimov_errors = dataset_asimov.models[0].spectral_model.evaluate_error(ebins)
asimov_N, asimov_errors_N = dataset_asimov_N.models[0].spectral_model.evaluate_error(ebins)

asimov_errors


# In[ ]:


withine = []
withine_N = []

for ie, e in enumerate(ebins):
    xs = []
    xs_N = []
    for d in datasets:
        value = d.models[0].spectral_model(e)   
        lowerlim = (np.all(value <= asimov[ie] + asimov_errors[ie]))
        upperlim = (np.all(value >= asimov[ie] - asimov_errors[ie]))
        x = lowerlim & upperlim
        xs.append(x)
        
        value = d.models[0].spectral_model(e)   
        lowerlim = (np.all(value <= asimov_N[ie] + asimov_errors_N[ie]))
        upperlim = (np.all(value >= asimov_N[ie] - asimov_errors_N[ie]))
        x = lowerlim & upperlim
        xs_N.append(x)
        
        
    withine.append(np.count_nonzero(xs) / len(datasets))
    withine_N.append(np.count_nonzero(xs_N) / len(datasets))
    
withine    


# In[ ]:


fig, ax = plt.subplots()
plt.plot(ebins, withine, label = "st")
plt.plot(ebins, withine_N, label = "nui")

plt.xscale('log')
plt.legend()
fig.savefig("plots/6_fraction_wihine.pdf")


# In[ ]:


from gammapy.maps import MapAxis
from gammapy.modeling.models.spectral import scale_plot_flux
energy_bounds = (ebins[0], ebins[-1] ) * u.TeV


energy_min, energy_max = energy_bounds
energy = MapAxis.from_energy_bounds(
    energy_min,
    energy_max,
    18,
)

fluxes = []
for i, d in enumerate(datasets):
    flux, _ = d.models[0].spectral_model._get_plot_flux(sed_type='dnde', energy=energy)

    flux = scale_plot_flux(flux, energy_power=2)
    flux = flux.quantity[:, 0, 0]
    fluxes.append(flux)
fluxes = np.array(fluxes)

mean = fluxes.mean(axis =0)
std = fluxes.std(axis =0)


# In[ ]:


true_energy = datasets[0].exposure.geom.axes[0].center.value

fig, axs = plt.subplots()
for i, d in enumerate(datasets):
    d.models[0].spectral_model.plot(energy_bounds,  energy_power = 2, ax = axs,
                                label = "Best-Fit; sys = -10%", color = 'grey',
                                   alpha = 0.2)
    
dataset_asimov_N.models[0].spectral_model.plot(energy_bounds,  energy_power = 2, ax = axs,
                                label = "Input", color = "red")    

dataset_asimov_N.models[0].spectral_model.plot_error(energy_bounds,  energy_power = 2, ax = axs,
                                label = "Input", facecolor = "red")    
    
dataset_asimov.models[0].spectral_model.plot(energy_bounds,  energy_power = 2, ax = axs,
                                label = "Input", color = "blue")    

dataset_asimov.models[0].spectral_model.plot_error(energy_bounds,  energy_power = 2, ax = axs,
                                label = "Input", facecolor = "blue")

dataset_standard.models[0].spectral_model.plot(energy_bounds, linestyle=':', energy_power = 2, ax = axs,
                                label = "Input", color = "black", )

axs.errorbar(energy.center.value, mean, yerr = std, fmt='x')

fig.savefig(f"plots/6_spectra_{factor}_{N}.pdf")


# In[ ]:


Ls = []
for d in datasets:
    Ls.append(d.stat_sum())


# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(indices[:,0], amplitudes[:,0], Ls, marker='o')
fig.savefig(f"plots/6_Ls_{factor}_{N}.pdf")


# In[ ]:




