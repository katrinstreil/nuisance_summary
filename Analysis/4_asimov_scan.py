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
from scipy.interpolate import interp2d

from regions import CircleSkyRegion, RectangleSkyRegion
import yaml
import sys
sys.path.append('../')
import Dataset_load 

from  Dataset_Setup import Setup, GaussianCovariance_matrix

from matplotlib import rc

rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)


######################################################################
# Read dataset
# ------------
# 

c = Dataset_load.load_config()
awo, aw, ewo, ew = c['_colors']  

livetime = c['livetime']
zero = c['zero'] 
norm = c['norm'] 
tilt = c['tilt'] 
bias = c['bias'] 
resolution = c['resolution'] 
magnitude = c['magnitude'] 
corrlength = c['corrlength']
sys = c['sys'] 
folder = c['folder']
parameter_names = c['parameter_names']        
nbidx = 0
print(sys)

parameter_names_1  = set(list(np.array(parameter_names).ravel()))
for p in parameter_names_1:
    print(p)

# %%time
dataset_input  = Dataset_load.create_asimov(model = c['model'], source = c['source'], 
                                               livetime = f"{livetime}-hr",
                                        parameters = None)



######################################################################
# Datasets
# --------
# 

    
setup = Setup(dataset_input=dataset_input)
#setup.set_up_irf_sys(bias, resolution, norm, tilt)
dataset_asimov, dataset_asimov_N = setup.run()
# irf model
setup.set_irf_model(dataset_asimov_N)
if  "Eff_area" in sys:
    dataset_asimov_N.models.parameters['resolution'].frozen = True
    dataset_asimov_N.irf_model.parameters['tilt'].frozen = False
    dataset_asimov_N.irf_model.parameters['bias'].frozen = True
    setup.set_irf_prior(dataset_asimov_N, bias, resolution, norm, tilt)
    e_reco_n = 10
    
if sys == "E_reco":
    dataset_asimov_N.models.parameters['resolution'].frozen = True
    dataset_asimov_N.irf_model.parameters['tilt'].frozen = True
    dataset_asimov_N.irf_model.parameters['bias'].frozen = False
    dataset_asimov_N.irf_model.parameters['norm'].frozen = True
    setup.set_irf_prior(dataset_asimov_N, bias, resolution, norm, tilt)
    e_reco_n = 2000
    
    
if  "Combined" in sys:
    dataset_asimov_N.models.parameters['resolution'].frozen = True
    dataset_asimov_N.irf_model.parameters['tilt'].frozen = False
    dataset_asimov_N.irf_model.parameters['bias'].frozen = False
    dataset_asimov_N.irf_model.parameters['norm'].frozen = False
    setup.set_irf_prior(dataset_asimov_N, bias, resolution, norm, tilt)
    e_reco_n = 2000

    
if sys == "BKG":
        
    # piece wise model
    # remove old bkg model
    setup.set_up_bkg_sys_V( breake = 10,
                        index1 = 2,
                        index2 = 1.5, 
                        magnitude = magnitude )

    dataset_asimov, dataset_asimov_N = setup.run()

    setup.unset_model(dataset_asimov_N, FoVBackgroundModel)
    setup.set_piecewise_bkg_model(dataset_asimov_N)
    # energy of the following parameters smaller than ethrshold
    dataset_asimov_N.background_model.parameters['norm0'].frozen = True
    dataset_asimov_N.background_model.parameters['norm1'].frozen = True
    dataset_asimov_N.background_model.parameters['norm2'].frozen = True
    dataset_asimov_N.background_model.parameters['norm3'].frozen = True
    setup.set_bkg_prior(dataset_asimov_N, magnitude, corrlength)
    frozen_pos = 1
    if frozen_pos:
        dataset_asimov.models.parameters['lon_0'].frozen = True
        dataset_asimov.models.parameters['lat_0'].frozen = True
        dataset_asimov_N.models.parameters['lon_0'].frozen = True
        dataset_asimov_N.models.parameters['lat_0'].frozen = True




######################################################################
# Scan
# ----
# 

def computing_scan(dataset, note):
        
    fit_cor = Fit(store_trace=False)
    result_cor = fit_cor.run(dataset)
    print(dataset_asimov.models)
    
    results = []
    for parname1 in parameter_names_1 :
        print("scanning",  parname1)
        dataset.models.parameters[parname1].scan_n_values=numpoints
        result = fit_cor.stat_profile(dataset,
                             dataset.models.parameters[parname1],
                            reoptimize = True
                            )

        contour_write = dict()
        for k in result.keys():
            print(k)
            if k != "fit_results":
                contour_write[k] = [float(_) for _ in result[k]]#.tolist()
        print(contour_write)
        with open(f"../{c['folder']}/data/4_scan_{note}_{parname1}_{numpoints}.yml", "w") as outfile:
            yaml.dump(contour_write, outfile, default_flow_style=False)

        results.append(result)
    return results
        
def read_in_scan(note):
    results = []
    for parname1 in parameter_names_1 :
        try:
            with open(f"../{c['folder']}/data/4_scan_{note}_{parname1}_{numpoints}.yml", "r") as stream:
                contour = yaml.safe_load(stream)
        except:
            with open(f"../{c['folder']}/data/4_scan_{note}_{parname1}.yml", "r") as stream:
                contour = yaml.safe_load(stream)
        results.append(contour)
    return results


# %%time
numpoints = 5
computing = 1
if computing:
    results = computing_scan(dataset_asimov, "2.15h")
else:
    results = read_in_scan("2.15h")
    path = f'../{folder}/data/0_model_livetime_{livetime}.yml'
    dataset_asimov.models = Models.read(path)
    


# %%time
computing = 1

if computing:
    dataset_asimov_N.models.parameters['lon_0'].frozen = True
    dataset_asimov_N.models.parameters['lat_0'].frozen = True
    
    results_N = computing_scan(dataset_asimov_N, "N_2.15h")
else:
    results_N = read_in_scan("N_2.15h")
    path = f'../{folder}/data/0_model_nui_livetime_{livetime}.yml'
    dataset_asimov_N = Dataset_load.load_dataset_N(dataset_asimov_N, path,bkg_sys = False)        
print(results_N)

import colors as s
s.blue

import upper_limit_18_02

colors_ = [s.blue, s.orange,
          s.lblue, s.lorange]

colors_ = [awo[0] , aw[0],
           awo[1] , aw[1]]

for i, p in enumerate(parameter_names_1):
    #if p == 'index':
    if True:
        stat_profile_N = results_N[i]
        stat_profile = results[i]

        stat_profile_N['stat_scan'] -= np.min(stat_profile_N['stat_scan'])
        stat_profile['stat_scan'] -= np.min(stat_profile['stat_scan'])

        fig = plt.figure()
        ll_N_a = stat_profile_N.copy()
        case = 'spectral'
        if p == 'sigma':
            case = 'spatial'
            
        amplitude_err = dataset_asimov.models[0].parameters[p].error
        amplitude = dataset_asimov.models[0].parameters[p].value
        amplitude_err_N = dataset_asimov_N.models[0].parameters[p].error
        amplitude_N = dataset_asimov_N.models[0].parameters[p].value

        fig, ax = plt.subplots(1,1)
            
        ### LIKELIHOOD
        # scan
        ll_a = stat_profile.copy()
        ul_a = upper_limit_18_02.upper_limit(ll_a,0,0,  name=f'{dataset_asimov.models[0].name}.{case}.{p}_scan')
        L_a, x_a = ul_a.interpolate()
        plt.plot(x_a, L_a(x_a),label = "-2log (L)", linestyle = 'dashed', color = colors_[0])
    
        ylim= ax.get_ylim()
        ymax = 2#ylim[1]
        min_, er_neg, er_pos = ul_a.likelihood_error_asymmetric()
        min_ = min_[0]; er_neg = er_neg[0]; er_pos = er_pos[0]; 
        dataset_asimov.models.parameters[p].error_n = er_neg
        dataset_asimov.models.parameters[p].error_p = er_pos
        
        factor = 1
        if p == 'amplitude':
            factor = 1e11

        ax.fill_between(  [np.nan, np.nan], ylim[0], ymax,  alpha = 0.5, color=colors_[2],
                        label = f'1$\sigma$ error (Minos): -{er_neg*factor:.2} +{er_pos*factor:.2} ')
     
        
        ax.vlines(amplitude-amplitude_err, ylim[0], ymax, color = colors_[0], linestyle ='dotted')
        ax.vlines(amplitude+amplitude_err, ylim[0], ymax, color = colors_[0], linestyle ='dotted',
                 label =  f'1$\sigma$ error (Minuit): {amplitude_err*factor:.2}')

           
        ### POSTERIOR
        # scan
        ul_N_a = upper_limit_18_02.upper_limit(ll_N_a,0,0, 
                                               name=f'{dataset_asimov.models[0].name}.{case}.{p}_scan')
        L_N_a, x_N_a = ul_N_a.interpolate()
        plt.plot(x_N_a, L_N_a(x_N_a),label = "-2log (P)", color = colors_[1])
        
        
        min_N, er_negN, er_posN = ul_N_a.likelihood_error_asymmetric()
        min_N = min_N[0]; er_negN = er_negN[0]; er_posN = er_posN[0]; 
        dataset_asimov_N.models.parameters[p].error_n = er_negN
        dataset_asimov_N.models.parameters[p].error_p = er_posN


        ax.fill_between(  [min_N-er_negN, min_N+ er_posN], ylim[0], ymax, alpha = 0.5, color = colors_[3],
                        label = f'1$\sigma$ error (Minos): -{er_negN*factor:.2} +{er_posN*factor:.2} ')
        ax.vlines(amplitude_N-amplitude_err_N, ylim[0], ymax,color = colors_[1] ,
                  linestyles='dotted'
                 )
        ax.vlines(amplitude_N+amplitude_err_N, ylim[0], ymax,color = colors_[1],
                  linestyles='dotted',
                    label = f'1$\sigma$ error (Minuit): $\pm${amplitude_err_N*factor:.2}')
                 
        nn = 2
        ax.set_xlim(amplitude_N-amplitude_err_N*nn, 
                   amplitude_N+amplitude_err_N*nn)
        ax.set_ylim(np.min(stat_profile['stat_scan'])-0.5,
                    np.min(stat_profile['stat_scan'])+ 3)

        
        
        ax.fill_between(  [min_-er_neg, min_+ er_pos], ylim[0], ymax,  alpha = 0.5, color=colors_[2],
                        label = f'')
       
    
    
        xx = ax.get_xlim()
        alpha = 0.6
        ax.hlines(0, xx[0], xx[1], color = 'grey', alpha = alpha)
        ax.hlines(1, xx[0], xx[1], color = 'grey', alpha = alpha)
        if p == 'amplitude':
            str_= "[$\\mathrm{TeV^{-1}\\,s^{-1}\\,cm^{-2}}$]"
            plt.xlabel(f"Source strength " + str_) 
        else:
            plt.xlabel(p)
        plt.ylabel("-2log (L) [arb. unit]")
        plt.legend(ncol = 2)

    fig.savefig(f"../{c['folder']}/plots/4_scan_{p}.pdf")
    
    

path = f'../{folder}/data/0_model_nui_livetime_{livetime}_np.yml'
dataset_asimov_N.models.write(path, overwrite = 1)

path = f'../{folder}/data/0_model_livetime_{livetime}_np.yml'
dataset_asimov.models.write(path, overwrite = 1)

print(dataset_asimov.models.parameters['index'].error)
print(dataset_asimov.models.parameters['index'].error_n)
print(dataset_asimov.models.parameters['index'].error_p)
print(dataset_asimov.models.parameters['index'].value)

print(dataset_asimov.models.parameters['amplitude'].error)
print(dataset_asimov.models.parameters['amplitude'].error_n)
print(dataset_asimov.models.parameters['amplitude'].error_p)
print(dataset_asimov.models.parameters['amplitude'].value)



######################################################################
# Minos
# -----
# 

# %%time
fit_cor = Fit(store_trace=False)
result_cor = fit_cor.run(dataset_asimov)
result_cor.minuit.minos()


# %%time
fit_cor_N = Fit(store_trace=False)
result_cor_N = fit_cor_N.run(dataset_asimov_N)
result_cor_N.minuit.minos()


lt = c['livetime']

print(dataset_asimov_N.models)

# %%time
compute_minos = True
if compute_minos :
       
    for p in result_cor_N.minuit.parameters:
        p_ = p[8:]
        print(p_)
        factor = 1 
        if p_ == "amplitude":
            factor = dataset_asimov.models.parameters['amplitude'].scale
        minos_model_N = Models(dataset_asimov_N.models.copy() )
        minos_model_N.parameters[p_].error_n = fit_cor_N.minuit.merrors[p].lower* factor
        minos_model_N.parameters[p_].error_p = fit_cor_N.minuit.merrors[p].upper* factor
        print(fit_cor_N.minuit.merrors[p].lower* factor)
    minos_model_N.write(f'../{folder}/data/4_minos_error_{lt}_nui.yaml', overwrite = True)
    
    for p in result_cor.minuit.parameters:
        p_ = p[8:]
        factor = 1 
        if p_ == "amplitude":
            factor = dataset_asimov.models.parameters['amplitude'].scale
        minos_model = Models(dataset_asimov.models.copy() )
        
        minos_model.parameters[p_].error_n = fit_cor.minuit.merrors[p].lower* factor
        minos_model.parameters[p_].error_p = fit_cor.minuit.merrors[p].upper* factor
    minos_model.write(f'../{folder}/data/4_minos_error_{lt}.yaml', overwrite = True)
    
    
else:
    minos_model_N = Models.read(f'../{folder}/data/4_minos_error_{lt}_nui.yaml')    
    minos_model = Models.read(f'../{folder}/data/4_minos_error_{lt}.yaml')

minos_model_N.parameters['index'].error_n

import upper_limit_18_02

colors_ = [s.blue, s.orange,
          s.lblue, s.lorange]

colors_ = [awo[0] , aw[0],
           awo[1] , aw[1]]

for i, p in enumerate(parameter_names_1):
    #if p == 'index':
    if True:
        print(p)
        stat_profile_N = results_N[i]
        stat_profile = results[i]

        stat_profile_N['stat_scan'] -= np.min(stat_profile_N['stat_scan'])
        stat_profile['stat_scan'] -= np.min(stat_profile['stat_scan'])

        fig = plt.figure()
        ll_N_a = stat_profile_N.copy()
        case = 'spectral'
        if p == 'sigma':
            case = 'spatial'
            
        amplitude_err = dataset_asimov.models[0].parameters[p].error
        amplitude = dataset_asimov.models[0].parameters[p].value
        amplitude_err_N = dataset_asimov_N.models[0].parameters[p].error
        amplitude_N = dataset_asimov_N.models[0].parameters[p].value

        fig, ax = plt.subplots(1,1)
            
        ### LIKELIHOOD
        # scan
        ll_a = stat_profile.copy()
        ul_a = upper_limit_18_02.upper_limit(ll_a,0,0,  name=f'{dataset_asimov.models[0].name}.{case}.{p}_scan')
        L_a, x_a = ul_a.interpolate()
        plt.plot(x_a, L_a(x_a),label = "-2log (L)", linestyle = 'dashed', color = colors_[0])
    
        ylim= ax.get_ylim()
        ymax = 2#ylim[1]
        min_, er_neg, er_pos = ul_a.likelihood_error_asymmetric()
        min_ = min_[0]; er_neg = er_neg[0]; er_pos = er_pos[0]; 
        dataset_asimov.models.parameters[p].error_n = er_neg
        dataset_asimov.models.parameters[p].error_p = er_pos
        
        factor = 1
        if p == 'amplitude':
            factor = 1e11

        ax.fill_between(  [np.nan, np.nan], ylim[0], ymax,  alpha = 0.5, color=colors_[2],
                        label = f'1$\sigma$ error (Scan): -{er_neg*factor:.2} +{er_pos*factor:.2} ')
     
        
        ax.vlines(amplitude-amplitude_err, ylim[0], ymax, color = colors_[0], linestyle ='dotted')
        ax.vlines(amplitude+amplitude_err, ylim[0], ymax, color = colors_[0], linestyle ='dotted',
                 label =  f'1$\sigma$ error (Minuit): {amplitude_err*factor:.2}')
         ## minos 
        # without nui      
        par = minos_model.parameters[p]
        value, error_n, error_p  = par.value, par.error_n, par.error_p
        print(value)
        
        ax.vlines(value-error_n, ylim[0], ymax,color = 'tab:orange' ,
                  linestyles='dashed'
                 )
        ax.vlines(value+error_p, ylim[0], ymax,color = 'tab:orange',
                  linestyles='dashed',
                    label = f'1$\sigma$ error (Minos): -{error_n*factor:.2} +{error_p*factor:.2} ')
        
           
        ### POSTERIOR
        # scan
        ul_N_a = upper_limit_18_02.upper_limit(ll_N_a,0,0, 
                                               name=f'{dataset_asimov.models[0].name}.{case}.{p}_scan')
        L_N_a, x_N_a = ul_N_a.interpolate()
        plt.plot(x_N_a, L_N_a(x_N_a),label = "-2log (P)", color = colors_[1])
        
        
        min_N, er_negN, er_posN = ul_N_a.likelihood_error_asymmetric()
        min_N = min_N[0]; er_negN = er_negN[0]; er_posN = er_posN[0]; 
        dataset_asimov_N.models.parameters[p].error_n = er_negN
        dataset_asimov_N.models.parameters[p].error_p = er_posN


        ax.fill_between(  [min_N-er_negN, min_N+ er_posN], ylim[0], ymax, alpha = 0.5, color = colors_[3],
                        label = f'1$\sigma$ error (Scan): -{er_negN*factor:.2} +{er_posN*factor:.2} ')
        ax.vlines(amplitude_N-amplitude_err_N, ylim[0], ymax,color = colors_[1] ,
                  linestyles='dotted'
                 )
        ax.vlines(amplitude_N+amplitude_err_N, ylim[0], ymax,color = colors_[1],
                  linestyles='dotted',
                    label = f'1$\sigma$ error (Minuit): $\pm${amplitude_err_N*factor:.2}')
        ## minos 
        # with nui
        par = minos_model_N.parameters[p]
        value, error_n, error_p  = par.value, par.error_n, par.error_p
        print(value)
        ax.vlines(value-error_n, ylim[0], ymax,color = 'purple' ,
                  linestyles='dashed'
                 )
        ax.vlines(value+error_p, ylim[0], ymax,color = 'purple',
                  linestyles='dashed',
                    label = f'1$\sigma$ error (Minos): -{error_n*factor:.2} +{error_p*factor:.2} ')
                   
              
            
        nn = 2
        ax.set_xlim(amplitude_N-amplitude_err_N*nn, 
                   amplitude_N+amplitude_err_N*nn)
        ax.set_ylim(np.min(stat_profile['stat_scan'])-0.5,
                    np.min(stat_profile['stat_scan'])+ 3)

        
        
        ax.fill_between(  [min_-er_neg, min_+ er_pos], ylim[0], ymax,  alpha = 0.5, color=colors_[2],
                        label = f'')
       
    
    
        xx = ax.get_xlim()
        alpha = 0.6
        ax.hlines(0, xx[0], xx[1], color = 'grey', alpha = alpha)
        ax.hlines(1, xx[0], xx[1], color = 'grey', alpha = alpha)
        if p == 'amplitude':
            str_= "[$\\mathrm{TeV^{-1}\\,s^{-1}\\,cm^{-2}}$]"
            plt.xlabel(f"Source strength " + str_) 
        else:
            plt.xlabel(p)
        plt.ylabel("-2log (L) [arb. unit]")
        plt.legend(ncol = 2)

    fig.savefig(f"../{c['folder']}/plots/4_scan_{p}_minos.pdf")
    
    


######################################################################
# Minos Errors
# ------------
# 

unit = "[10^{-11}/\\text{cm}^2 \\text{s} \\text{TeV}] "
unit2 = "[\\text{TeV}^{-1}]"
print(f"& &  $\Phi _0  {unit} $  & $\Lambda$ & $\lambda {unit2}$ \\\ \hline \hline")

for i, m in enumerate([setup.dataset_helper.models[0],minos_model , minos_model_N]):
    if i == 0:
        str_  = f" & Input &" 
    if i == 1:
        str_  = f" Bias  &Without fitting &" 
        
    if i == 2:
        str_  = f" & With fitting  &" 
    for j, p in enumerate(['amplitude', 'index', 'lambda_']):
        factor = 1
        if p == 'amplitude':
            factor = 1e11
            
        if i == 0:
            str_  += f" ${ m.parameters[p].value*factor:.5} $  &" 
        if i > 0:
            str_  += " \error {" + f"{m.parameters[p].value*factor:.5}" + '} { ' + f'{m.parameters[p].error*factor:.3}' + '}  {' + f'{m.parameters[p].error_p*factor:.3}' + '}  {' + f'{m.parameters[p].error_n*factor:.3}' +"} &" 
            
    str_ = str_[:-1]
    str_ += "\\\  "
    str_ += "\hline"
    print(f"{str_}")
    str_ = ""
    print()


######################################################################
# Scan Errors
# -----------
# 

unit = "[10^{-11}/\\text{cm}^2 \\text{s} \\text{TeV}] "
unit2 = "[\\text{TeV}^{-1}]"
print(f"& &  $\Phi _0  {unit} $  & $\Lambda$ & $\lambda {unit2}$ \\\ \hline \hline")

for i, m in enumerate([setup.dataset_helper.models[0],dataset_asimov.models[0] , 
                       dataset_asimov_N.models[0]]):
    if i == 0:
        str_  = f" & Input &" 
    if i == 1:
        str_  = f" Bias  &Without fitting &" 
        
    if i == 2:
        str_  = f" & With fitting  &" 
    for j, p in enumerate(['amplitude', 'index', 'lambda_']):
        factor = 1
        if p == 'amplitude':
            factor = 1e11
            
        if i == 0:
            str_  += f" ${ m.parameters[p].value*factor:.5} $  &" 
        if i > 0:
            str_  += " \error {" + f"{m.parameters[p].value*factor:.5}" + '} { ' + f'{m.parameters[p].error*factor:.3}' + '}  {' + f'{m.parameters[p].error_p*factor:.3}' + '}  {' + f'{m.parameters[p].error_n*factor:.3}' +"} &" 
            
    str_ = str_[:-1]
    str_ += "\\\  "
    str_ += "\hline"
    print(f"{str_}")
    str_ = ""
    print()

