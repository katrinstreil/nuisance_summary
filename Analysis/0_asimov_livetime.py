"""
Runnning with gammapy-dev/IRF_model
-----------------------------------

Fitting asimov datasets with nuisance parameters based on the different
livetimes

"""


######################################################################
# Setup
# ~~~~~
# 

import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from gammapy.maps import Map
from astropy.coordinates import SkyCoord, Angle
from gammapy.modeling import Fit,  Parameters, Covariance , Parameter
from gammapy.datasets import MapDataset ,Datasets, FluxPointsDataset
from gammapy.modeling.models import (
    PowerLawSpectralModel,
    SkyModel,
    PointSpatialModel,
    PowerLawNormSpectralModel,
    Models,
    SpatialModel,
    FoVBackgroundModel,
    PiecewiseNormSpectralModel,
)
from gammapy.estimators import TSMapEstimator, ExcessMapEstimator
from gammapy.estimators import FluxPoints, FluxPointsEstimator

from regions import CircleSkyRegion, RectangleSkyRegion
import yaml
import sys
sys.path.append('../')
import Dataset_load 

from  Dataset_Setup import Setup, GaussianCovariance_matrix


######################################################################
# Read dataset
# ------------
# 

c = Dataset_load.load_config()
awo, aw, ewo, ew = c['_colors']  

livetimes = c['livetimes']
zero = c['zero'] 
norm = c['norm'] 
tilt = c['tilt'] 
bias = c['bias'] 
resolution = c['resolution'] 
magnitude = c['magnitude'] 
corrlength = c['corrlength']
sys = c['sys'] 
folder = c['folder']
nbidx = 0

livetimes = livetimes[9:]
# %%time
datasets_input = []
for l in livetimes:
    datasets_i  = Dataset_load.create_asimov(model = c['model'], source = c['source'], 
                                                   livetime = f"{l}-hr",
                                            parameters = None)
    datasets_input.append(datasets_i)


######################################################################
# Datasets
# --------
# 

def set_up(dataset_input):
    
    setup = Setup(dataset_input=dataset_input)
    #setup.set_up_irf_sys(bias, resolution, norm, tilt)
    dataset_asimov, dataset_asimov_N = setup.run()
    # irf model
    setup.set_irf_model(dataset_asimov_N)
    if sys == "Eff_area":
        dataset_asimov_N.models.parameters['resolution'].frozen = True
        dataset_asimov_N.irf_model.parameters['tilt'].frozen = False
        dataset_asimov_N.irf_model.parameters['bias'].frozen = True
        setup.set_irf_prior(dataset_asimov_N, bias, resolution, norm, tilt)
        
    if sys == "E_reco":
        dataset_asimov_N.models.parameters['resolution'].frozen = True
        dataset_asimov_N.irf_model.parameters['tilt'].frozen = True
        dataset_asimov_N.irf_model.parameters['bias'].frozen = False
        dataset_asimov_N.irf_model.parameters['norm'].frozen = True
        setup.set_irf_prior(dataset_asimov_N, bias, resolution, norm, tilt)
        
    if sys == "Combined":
        dataset_asimov_N.models.parameters['resolution'].frozen = True
        dataset_asimov_N.irf_model.parameters['tilt'].frozen = False
        dataset_asimov_N.irf_model.parameters['bias'].frozen = False
        setup.set_irf_prior(dataset_asimov_N, bias, resolution, norm, tilt)

    # piece wise model
    # remove old bkg model
    #setup.unset_model(dataset_asimov_N, FoVBackgroundModel)
    #setup.set_piecewise_bkg_model(dataset_asimov_N)
    # energy of the following parameters smaller than ethrshold
    #dataset_asimov_N.background_model.parameters['norm0'].frozen = True
    #dataset_asimov_N.background_model.parameters['norm1'].frozen = True
    #dataset_asimov_N.background_model.parameters['norm2'].frozen = True
    #dataset_asimov_N.background_model.parameters['norm3'].frozen = True
    #setup.set_bkg_prior(dataset_asimov_N, magnitude, corrlength)

    
       
    return dataset_asimov_N, dataset_asimov

datasets_N = []
datasets = []
for d in datasets_input:
    dataset_asimov_N, dataset_asimov = set_up(d)
    datasets_N.append(dataset_asimov_N)
    datasets.append(dataset_asimov)

print(datasets[0].models)

# %%time
fitting = 0
if fitting:
    for i, d in enumerate(datasets):
        print(i)
        fit_cor = Fit(store_trace=0)
        result_cor = fit_cor.run(d)
        print(result_cor)
        print("saving")
        path = f'../{folder}/data/0_model_livetime_{livetimes[i]}.yml'
        d.models.write(path,overwrite=True)
    
else:
    print("read")
    for i, d in enumerate(datasets):
        print(i)
        path = f'../{folder}/data/0_model_livetime_{livetimes[i]}.yml'
        datasets[i].models = Models.read(path)


# %%time
fitting_N = 1

if fitting_N:
    
    for i, d in enumerate(datasets_N):
        print(i)
        fit_cor = Fit(store_trace=0)
        result_cor = fit_cor.run(d)
        print(result_cor)
        print("saving")
        path = f'../{folder}/data/0_model_nui_livetime_{livetimes[i]}.yml'
        d.models.write(path,overwrite=True)
    
else:
    print("read")
    for i, d in enumerate(datasets_N):
        print(i)
        path = f'../{folder}/data/0_model_nui_livetime_{livetimes[i]}.yml'
        print(path)
        datasets_N[i] = Dataset_load.load_dataset_N(d, path,bkg_sys = False)        
        datasets_N[i].models.parameters['tilt'].frozen = False 
    

#39min 32

labels = [str(int(l)) for l in livetimes]
models_list = [ d.irf_model for d in datasets_N]
pars =   [('bias', 'resolution'), ('norm', 'tilt')]

colors_default =  plt.rcParams['axes.prop_cycle'].by_key()['color'] 
colors_default = colors_default* 5

i = 0
fig,axs = plt.subplots(2,1)
for l, m in zip(livetimes, models_list):
    axs[0].plot(l, m.parameters[pars[0][0]].error, 'x', label = "bias", color = colors_default[i])
    axs[0].plot(l, m.parameters[pars[0][1]].error, 'o' , label = "resolution", color = colors_default[i])
    axs[1].plot(l, m.parameters[pars[1][0]].error, 'x' , label = "norm", color = colors_default[i])
    axs[1].plot(l, m.parameters[pars[1][1]].error, 'o' , label = "tilt", color = colors_default[i])
    i +=1
    
    
axs[0].plot([livetimes[0],livetimes[-1]], [bias, bias])
axs[0].plot([livetimes[0],livetimes[-1]], [resolution, resolution])
axs[1].plot([livetimes[0],livetimes[-1]], [norm, norm])
axs[1].plot([livetimes[0],livetimes[-1]], [tilt, tilt])  

for a in axs:
    a.set_xscale("log")
    a.legend()


    from collections import OrderedDict

    handles, labels = a.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    a.legend(by_label.values(), by_label.keys())
    
axs[0].set_ylim(-0.01, 0.2)
fig.savefig(f'../{folder}/plots/{nbidx}_irf_fit_results.pdf')

inputvalues = {}
for p in datasets_input[0].models[0].parameters:
    inputvalues[p.name]  = p.value

def plotting(parname = "amplitude"):
    fig = plt.figure(figsize  = (8,3))
    inputvalue = inputvalues[parname]
    
    errs = []
    errs_N = []
    livetimes_ext = np.array(livetimes).copy(); livetimes_ext2 = np.array(livetimes).copy(); 
    livetimes_ext[-1] *=1.1;
    livetimes_ext2[0:,] *=1.1; 
    livetimes_ext2 = np.insert(livetimes_ext2, 0, values =  livetimes_ext2[0]* 0.9)


    for i, d in enumerate(datasets_N):
        par = datasets[i].models[0].parameters[parname]
        val, err = par.value, par.error
        errs.append( err/ inputvalue)
        plt.errorbar(livetimes[i], val/ inputvalue, yerr = err/ inputvalue,  fmt='x', 
                     color = awo[0])

        par = d.models[0].parameters[parname]
        val, err = par.value, par.error
        errs_N.append( err/ inputvalue)
        plt.errorbar( 1.1* livetimes[i] , val/ inputvalue, yerr = err/ inputvalue,  fmt='x', 
                     color = aw[0])

    

    errs_N = np.insert(errs_N, 0, values =  errs_N[0])
    plt.fill_between(  livetimes_ext2, 1- np.array(errs_N), 1+np.array(errs_N) 
                     ,color = aw[1], alpha = 0.8, label = '')
    plt.fill_between( livetimes_ext, 1- np.array(errs), 1+np.array(errs)
                     ,color = awo[1], alpha = 0.8, label = '')

    plt.errorbar( np.nan, np.nan , yerr = np.nan, fmt='x', color = aw[0],
                 label = 'fit with systematics')
    plt.errorbar( np.nan, np.nan , yerr = np.nan, fmt='x', color = awo[0], 
                 label = 'fit w/o systematics')
    ax1 = plt.gca()
    xlim = ax1.get_xlim()
    plt.hlines(1, xlim[0], xlim[1], color = 'lightgrey')
    #plt.title("Best Fit Amplitude")
    plt.legend()
    plt.xscale('log')
    plt.xlabel("Livetime [hr]")
    plt.ylabel(f"Best-fit {parname} \n[Input {parname}]")

    ax2 = ax1.twiny()
    ax2.set_xscale("log")
    npred = [d.npred_signal().data.sum() for d in datasets]
    
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(livetimes)
    clabels = [ f'{int(c)}' for c in npred]
    ax2.set_xticklabels(clabels)
    ax2.set_xlabel(r"Signal Counts")
    plt.tight_layout()
    return fig

fig = plotting(parname = "amplitude")
fig.savefig(f'../{folder}/plots/{nbidx}_best_fit_vs_livetime_amplitude.svg')
fig.savefig(f'../{folder}/plots/{nbidx}_best_fit_vs_livetime_amplitude.pdf')
fig.savefig(f'../{folder}/plots/{nbidx}_best_fit_vs_livetime_amplitude.png')

fig = plotting(parname = "index1")
fig.savefig(f'../{folder}/plots/{nbidx}_best_fit_vs_livetime_index.svg')
fig.savefig(f'../{folder}/plots/{nbidx}_best_fit_vs_livetime_index.pdf')
fig.savefig(f'../{folder}/plots/{nbidx}_best_fit_vs_livetime_index.png')

fig = plotting(parname = "index2")
fig.savefig(f'../{folder}/plots/{nbidx}_best_fit_vs_livetime_index2.svg')
fig.savefig(f'../{folder}/plots/{nbidx}_best_fit_vs_livetime_index2.pdf')
fig.savefig(f'../{folder}/plots/{nbidx}_best_fit_vs_livetime_index2.png')

fig = plotting(parname = "ebreak")
fig.savefig(f'../{folder}/plots/{nbidx}_best_fit_vs_livetime_ebreak.svg')
fig.savefig(f'../{folder}/plots/{nbidx}_best_fit_vs_livetime_ebreak.pdf')
fig.savefig(f'../{folder}/plots/{nbidx}_best_fit_vs_livetime_ebreak.png')




######################################################################
# RND Fluxpoints
# --------------
# 

for i, d in enumerate(datasets):
    l = livetimes[i]
    print()
    print(l)
    dataset = datasets[i]
    dataset_N = datasets_N[i]
    parameters = dataset.models[0].parameters.names
    for p in parameters:
        
        par = dataset.models[0].parameters[p]
        per = par.error /par.value * 100 
        print(f"{p}:")
        print(f"without {per:.4} %")
        par = dataset_N.models[0].parameters[p]
        per = par.error /par.value * 100 
        
        print(f"with    {per:.4} %")        

def plot_spectra():
    fig,axs = plt.subplots(len(datasets), 1, figsize = (6, 5 * len(datasets)))
    axs= axs.flatten()
    energy_power = 2
    for i, d in enumerate(datasets):
        ax = axs[i]
        live = livetimes[i]
        dataset = datasets[i]
        dataset_N = datasets_N[i]

        dataset_N.models[0].spectral_model.plot((0.1, 100) *u.TeV, energy_power = energy_power,
                                                     color = aw[0], ax = ax)    
        par = dataset_N.models[0].parameters['amplitude']
        per = par.error /par.value
        dataset_N.models[0].spectral_model.plot_error((0.1, 100) *u.TeV,energy_power = energy_power, ax = ax, 
                                                   facecolor= aw[1], alpha = 0.8, 
                                                     label = f'With nui par.\n amplitude error =  {per*100:.3}% ')
        dataset.models[0].spectral_model.plot((0.1, 100) *u.TeV,energy_power = energy_power, ax = ax, 
                                                   color= awo[0], linestyle = 'dashed')
        par = dataset.models[0].parameters['amplitude']
        per = par.error /par.value
        dataset.models[0].spectral_model.plot_error((0.1, 100) *u.TeV,energy_power = energy_power, ax = ax, 
                                                   facecolor= awo[1], alpha = 0.8,
                                                   label = f'W/o nui par.\n amplitude error =  {per*100:.3}%')
        ax.set_title(f"Livetime: {float(live):.2} hr")
        ax.set_ylim(1e-12, 1e-10)

        if i <2:
            ax.set_ylim(6e-13, 6e-9)
            
        
        #if live in lives:
        #    idx = np.where(live == lives)[0][-1]
        #    fpd_Ns[idx].data.plot(ax = ax, energy_power = energy_power, color = purple[0],
        ##                         label = "Example fluxpoint with nui par.")
        #    fpd_s[idx].data.plot(ax = ax, energy_power = energy_power, color = orange[0],
        #                        label = "Example fluxpoint w/o nui par.")
        ax.legend()
    plt.tight_layout()
    return fig
        
fig = plot_spectra()
fig.savefig(f'../{folder}/plots/{nbidx}_spectra_with_rnd_fp.pdf')

par_names = ['index', 'amplitude', 'lambda_', 'bias', 'resolution', 'norm' , 
            'tilt']
names =  ['index', 'amplitude',  
        'bkg norm', 'bkg tilt',
          'bias', 'resolution',
          'eff. area norm' , 
            'eff. area tilt']
for i in range(len(livetimes)):

    #datasets_N[i].models.read_covariance("data", f"8_model_nui_livetime_{livetimes[i]}_covariance.dat", format="ascii.fixed_width")
    parameters = Parameters([_ for _ in datasets_N[i].models.parameters if _.name in par_names])
    cov_par = datasets_N[i].models.covariance.get_subcovariance(parameters)
    ax = cov_par.plot_correlation()
    ax.set_title(str(livetimes[i]) + " hr")
    fig = plt.gcf()
    fig.savefig(f'../{folder}/plots/{nbidx}_covmatrix_livetime_{livetimes[i]}.svg')
    fig.savefig(f'../{folder}/plots/{nbidx}_covmatrix_livetime_{livetimes[i]}.pdf')
    


